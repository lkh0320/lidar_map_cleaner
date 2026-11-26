"""Offline: build clean navigation layers from a saved keyframe map.

  ros2 run lidar_map_cleaner build_layers <map_dir>                  # moving objects removed first
  ros2 run lidar_map_cleaner build_layers <map_dir> --keep-dynamic

Outputs in <map_dir> (previous files are moved to layers_old/):
  octomap.bt                      3D voxels (octomap_server)
  grid.pgm, grid.yaml             2D occupancy, Nav2 map_server format
  elevation.npz/.pgm/.yaml        2.5D elevation (max / min / count layers)
  elevation.png                   colour elevation image (turbo colormap, transparent = no data)
  voxels.pcd                      static map as voxel centres coloured by height above the floor (XYZRGB)
  voxels.ply                      (--voxel-ply) the same as coloured cubes, a mesh for MeshLab / CloudCompare

Compared with layers accumulated live during SLAM:
  - keyframes are inserted at their *final* (loop-closed) poses, so loop closures are reflected
  - points classified as dynamic (remove_dynamic) are left out
  - rays start at the LiDAR origin (not the robot base)
The elevation map is computed directly from the static points: a cell is an obstacle (highest point) only when enough
of its points stand clearly above the local floor, otherwise it is floor (median height of its floor points), so LiDAR
noise and ring patterns do not make a flat floor blotchy.
octomap_server is started in a private ROS domain (localhost only), so this can run next to a live robot stack.
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

from .colorize import height_colors, write_pcd_rgb, write_png, write_voxel_ply
from .pcd_io import Keyframes
from .remove_dynamic import add_args, classify, local_floor, sensor_from_args

OUT_FILES = ['octomap.bt', 'grid.pgm', 'grid.yaml', 'elevation.npz', 'elevation.pgm', 'elevation.yaml',
             'elevation.png', 'voxels.pcd', 'voxels.ply']


def write_pgm(path, img):
    h, w = img.shape
    with open(path, 'wb') as f:
        f.write(f'P5\n{w} {h}\n255\n'.encode())
        f.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())


def save_grid(m, out):
    g = np.array(m.data, dtype=np.int16).reshape(m.info.height, m.info.width)
    img = np.full(g.shape, 205, np.uint8)
    img[g == 0] = 254
    img[g >= 65] = 0
    write_pgm(os.path.join(out, 'grid.pgm'), img[::-1])
    o, q = m.info.origin.position, m.info.origin.orientation
    yaw = float(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
    with open(os.path.join(out, 'grid.yaml'), 'w') as f:
        f.write(f'image: grid.pgm\nmode: trinary\nresolution: {m.info.resolution}\n'
                f'origin: [{o.x}, {o.y}, {yaw}]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n')
    return f'{m.info.width}x{m.info.height} @ {m.info.resolution:.2f} m, {(g >= 65).sum()} occupied cells'


def save_elevation(m, out):
    """from a grid_map_msgs/GridMap (elevation_mapper)"""
    layers = {}
    for name, arr in zip(m.layers, m.data):
        cols, rows = arr.layout.dim[0].size, arr.layout.dim[1].size
        layers[name] = np.array(arr.data, dtype=np.float32).reshape((rows, cols), order='F')
    img_layers = {k: v[::-1, :].T for k, v in layers.items()}   # image: +x right, +y up
    return save_elevation_layers(img_layers, m.info.resolution, m.info.pose.position.x - m.info.length_x / 2,
                                 m.info.pose.position.y - m.info.length_y / 2, m.header.frame_id, out)


def robust_elevation(P, floor, res=0.1, length_x=40.0, length_y=40.0, max_h=1.2, min_h=-1.5, obj=0.06,
                     min_points=2):
    """2.5D elevation straight from the static points (image layout: row 0 = +y end, col 0 = x0).

    A cell is an obstacle cell when at least max(3, 10 %) of its points are more than `obj` above the local floor;
    then elevation = highest point. Otherwise it is floor and elevation = median of its floor points, so LiDAR noise and
    ring patterns (a few points a few cm high) no longer make the floor blotchy.
    """
    rel = P[:, 2] - floor
    keep = (rel > min_h) & (rel < max_h)
    P, rel = P[keep], rel[keep]
    if len(P):                                    # grow the grid (centred on the origin) when the map is larger
        length_x = max(length_x, 2 * np.abs(P[:, 0]).max() + 2 * res)
        length_y = max(length_y, 2 * np.abs(P[:, 1]).max() + 2 * res)
    nx, ny = int(np.ceil(length_x / res)), int(np.ceil(length_y / res))
    x0, y0 = -nx * res / 2, -ny * res / 2
    col = np.floor((P[:, 0] - x0) / res).astype(np.int64)
    row = np.floor((y0 + ny * res - P[:, 1]) / res).astype(np.int64)
    ok = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
    cell, z, r = row[ok] * nx + col[ok], P[ok, 2].astype(np.float64), rel[ok]
    N = nx * ny
    cnt = np.bincount(cell, minlength=N)
    zmax = np.full(N, np.nan)
    np.fmax.at(zmax, cell, z)
    zmin = np.full(N, np.nan)
    np.fmin.at(zmin, cell, z)
    up = np.bincount(cell, weights=(r > obj).astype(float), minlength=N)
    is_obj = up >= np.maximum(3, 0.1 * cnt)
    lo = r <= obj
    o = np.lexsort((z[lo], cell[lo]))
    cs, zs = cell[lo][o], z[lo][o]
    uc, st, ct = np.unique(cs, return_index=True, return_counts=True)
    med = np.full(N, np.nan)
    med[uc] = zs[st + ct // 2]
    elev = np.where(is_obj | np.isnan(med), zmax, med)
    sparse = cnt < min_points
    elev[sparse], zmin[sparse] = np.nan, np.nan
    shape = (ny, nx)
    return dict(elevation=elev.reshape(shape).astype(np.float32), min=zmin.reshape(shape).astype(np.float32),
                count=np.where(cnt > 0, cnt, np.nan).reshape(shape).astype(np.float32)), x0, y0


def save_elevation_layers(img_layers, res, x0, y0, frame, out):
    np.savez_compressed(os.path.join(out, 'elevation.npz'), resolution=res, origin=np.array([x0, y0]),
                        frame=frame,
                        note='array[row, col]: row 0 = top (+y end). cell centre = '
                             '(x0 + (col+0.5)*res, y0 + (rows-1-row+0.5)*res)', **img_layers)
    e = img_layers['elevation']
    valid = ~np.isnan(e)
    if not valid.any():
        raise RuntimeError('elevation map is empty')
    zmin, zmax = float(np.nanmin(e)), float(np.nanmax(e))
    img = np.zeros(e.shape, np.uint8)
    img[valid] = (1 + (e[valid] - zmin) / max(zmax - zmin, 1e-6) * 254).astype(np.uint8)
    write_pgm(os.path.join(out, 'elevation.pgm'), img)
    rgba = np.zeros(e.shape + (4,), np.uint8)
    rgba[valid, :3] = height_colors(e[valid], zmin, zmax)
    rgba[valid, 3] = 255
    write_png(os.path.join(out, 'elevation.png'), rgba)
    with open(os.path.join(out, 'elevation.yaml'), 'w') as f:
        f.write(f'image: elevation.pgm   # 1..255 = z_min..z_max, 0 = no data (raw values in elevation.npz)\n'
                f'color_image: elevation.png   # turbo colormap, blue = z_min .. red = z_max, transparent = no data\n'
                f'resolution: {res}\norigin: [{x0}, {y0}, 0.0]\nz_min: {zmin:.3f}\nz_max: {zmax:.3f}\n'
                f'frame: {frame}\n')
    return f'{int(valid.sum())} cells, z {zmin:.2f} .. {zmax:.2f} m'


def save_octomap(out):
    path = os.path.join(out, 'octomap.bt')
    p = subprocess.run(['ros2', 'run', 'octomap_server', 'octomap_saver_node', '--ros-args', '-p', f'octomap_path:={path}'],
                       capture_output=True, text=True, timeout=60)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return f'{os.path.getsize(path) / 1e3:.0f} KB'
    raise RuntimeError((p.stderr or p.stdout).strip().splitlines()[-1] if (p.stderr or p.stdout) else 'octomap save failed')


def save_voxels(P, floor, out, size=0.1, top=2.0, ply=False, min_points=2):
    """voxel centres of the static map coloured by height above the floor (blue 0 m .. red `top` m)"""
    k = np.floor(P / size).astype(np.int64)
    uk, inv, cnt = np.unique(k, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    fl = np.bincount(inv, weights=floor) / cnt
    keep = cnt >= min_points
    c = (uk[keep] + 0.5) * size
    rgb = height_colors(c[:, 2] - fl[keep], 0.0, top)
    write_pcd_rgb(os.path.join(out, 'voxels.pcd'), c, rgb)
    msg = f'{len(c)} voxels of {size} m, coloured 0..{top} m above the floor'
    if ply:
        write_voxel_ply(os.path.join(out, 'voxels.ply'), c, size * 0.98, rgb)
        msg += ' (+ voxels.ply)'
    return msg


def main(argv=None):
    ap = argparse.ArgumentParser(description='Build octomap / 2D occupancy / elevation from a keyframe map')
    ap.add_argument('map_dir')
    ap.add_argument('--keep-dynamic', action='store_true', help='do not remove moving objects')
    ap.add_argument('--domain', type=int, default=87, help='private ROS domain id (<= 232)')
    ap.add_argument('--rate', type=float, default=10.0, help='keyframes inserted per second')
    ap.add_argument('--octomap-config', default=None, help='octomap_server params yaml (default: package config)')
    ap.add_argument('--elevation-config', default=None,
                    help='elevation params yaml: resolution, length_x/y, max/min_height_above_robot, min_points '
                         '(default: package config)')
    ap.add_argument('--elevation-obstacle', type=float, default=0.06,
                    help='points more than this above the local floor count as obstacle (m)')
    ap.add_argument('--voxel-size', type=float, default=0.1, help='voxels.pcd cube size (m)')
    ap.add_argument('--voxel-color-max', type=float, default=2.0, help='height above floor shown red in voxels (m)')
    ap.add_argument('--voxel-ply', action='store_true', help='also write voxels.ply (coloured cube mesh)')
    add_args(ap)
    a = ap.parse_args(argv)
    d = os.path.abspath(a.map_dir)

    os.environ['ROS_DOMAIN_ID'] = str(a.domain)
    os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'LOCALHOST'
    os.environ.pop('ROS_LOCALHOST_ONLY', None)
    os.environ.pop('CYCLONEDDS_URI', None)

    kf = Keyframes(d)
    sensor = sensor_from_args(a)
    P, I, K, dyn = classify(kf, sensor, a)
    if a.keep_dynamic:
        dyn[:] = False

    import rclpy
    from ament_index_python.packages import get_package_share_directory
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import OccupancyGrid
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header
    from tf2_ros import TransformBroadcaster

    share = get_package_share_directory('lidar_map_cleaner')
    oct_cfg = a.octomap_config or os.path.join(share, 'config', 'octomap.yaml')
    ele_cfg = a.elevation_config or os.path.join(share, 'config', 'elevation.yaml')
    import yaml
    with open(ele_cfg) as f:
        ep = yaml.safe_load(f) or {}
    ep = ep.get('elevation_mapper', ep).get('ros__parameters', ep)
    procs = [
        subprocess.Popen(['ros2', 'run', 'octomap_server', 'octomap_server_node', '--ros-args', '-r', '__node:=octomap_server',
                          '--params-file', oct_cfg, '-r', 'cloud_in:=/build_layers/cloud',
                          '-p', 'frame_id:=map', '-p', 'base_frame_id:=base_link'],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True),
    ]
    rclpy.init()
    node = rclpy.create_node('build_layers')
    pub = node.create_publisher(PointCloud2, '/build_layers/cloud', 10)
    tfb = TransformBroadcaster(node)
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    got = {}
    node.create_subscription(OccupancyGrid, '/projected_map', lambda m: got.__setitem__('grid', m), latched)
    fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate('xyz')]
    fields.append(PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1))

    def tf_msg(stamp, child, t, q):
        m = TransformStamped()
        m.header.stamp, m.header.frame_id, m.child_frame_id = stamp, 'map', child
        m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = map(float, t)
        m.transform.rotation.x, m.transform.rotation.y, m.transform.rotation.z, m.transform.rotation.w = map(float, q)
        return m

    def spin_for(sec):
        end = time.time() + sec
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.02)

    ok = True
    try:
        end = time.time() + 20
        while time.time() < end and pub.get_subscription_count() < 1:
            rclpy.spin_once(node, timeout_sec=0.2)
        if pub.get_subscription_count() < 1:
            print('octomap_server did not start (is octomap_server installed?)', file=sys.stderr)
            return 1
        spin_for(1.0)
        print(f'inserting {len(kf)} keyframes ({int((~dyn).sum())} static points)...', flush=True)
        for k in range(len(kf)):
            R, t = kf.R[k], kf.t[k]
            R_ws, o_ws = sensor.sensor_pose_in_world(R, t)
            sel = (K == k) & ~dyn
            ps = R_ws.inv().apply(P[sel] - o_ws)               # sensor frame -> rays start at the LiDAR
            arr = np.zeros((len(ps), 4), np.float32)
            arr[:, :3], arr[:, 3] = ps, I[sel]
            stamp = node.get_clock().now().to_msg()
            tfb.sendTransform([tf_msg(stamp, 'base_link', t, R.as_quat()),
                               tf_msg(stamp, 'build_layers_lidar', o_ws, R_ws.as_quat())])
            time.sleep(0.02)
            pub.publish(PointCloud2(header=Header(stamp=stamp, frame_id='build_layers_lidar'), height=1, width=len(ps),
                                    fields=fields, is_bigendian=False, point_step=16, row_step=16 * len(ps),
                                    is_dense=True, data=arr.tobytes()))
            spin_for(1.0 / a.rate)
            if (k + 1) % 100 == 0:
                print(f'  {k + 1}/{len(kf)}', flush=True)
        # octomap re-publishes the 2D map after every insertion (latched):
        # wait a few seconds so the last message (after the final keyframe) is the one kept in `got`
        spin_for(4.0)
        end = time.time() + 10
        while time.time() < end and 'grid' not in got:
            rclpy.spin_once(node, timeout_sec=0.1)

        old = os.path.join(d, 'layers_old')
        os.makedirs(old, exist_ok=True)
        for f in OUT_FILES:
            src = os.path.join(d, f)
            if os.path.exists(src):
                shutil.move(src, os.path.join(old, f))
        st = ~dyn
        Ps = P[st]
        floor = local_floor(Ps, kf.t)

        def elevation():
            layers, x0, y0 = robust_elevation(
                Ps, floor, float(ep.get('resolution', 0.1)), float(ep.get('length_x', 40.0)),
                float(ep.get('length_y', 40.0)), float(ep.get('max_height_above_robot', 1.2)),
                float(ep.get('min_height_above_robot', -1.5)), a.elevation_obstacle, int(ep.get('min_points', 2)))
            return save_elevation_layers(layers, float(ep.get('resolution', 0.1)), x0, y0, 'map', d)
        for name, fn in (('octomap.bt', lambda: save_octomap(d)),
                         ('grid.pgm/yaml', lambda: save_grid(got['grid'], d)),
                         ('elevation (+ elevation.png)', elevation),
                         ('voxels.pcd', lambda: save_voxels(Ps, floor, d, a.voxel_size, a.voxel_color_max,
                                                            a.voxel_ply))):
            try:
                print(f'  [OK] {name}: {fn()}')
            except Exception as e:  # noqa: BLE001
                ok = False
                print(f'  [FAILED] {name}: {e}')
        return 0 if ok else 1
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)


if __name__ == '__main__':
    raise SystemExit(main())
