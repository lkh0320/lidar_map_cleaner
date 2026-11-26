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
from .pcd_io import Keyframes, read_pcd
from .remove_dynamic import add_args, classify, local_floor, sensor_from_args

OUT_FILES = ['octomap.bt', 'grid.pgm', 'grid.yaml', 'elevation.npz', 'elevation.pgm', 'elevation.yaml',
             'elevation.png', 'voxels.pcd', 'voxels.ply']


def write_pgm(path, img):
    h, w = img.shape
    with open(path, 'wb') as f:
        f.write(f'P5\n{w} {h}\n255\n'.encode())
        f.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())


def prefer_grid(m_ref, m_all):
    """overwrite m_all (OccupancyGrid, same resolution) with every known cell of m_ref
    -> (number of cells taken, bool mask of those cells in m_all's image layout (row 0 = top))"""
    res = m_all.info.resolution
    ga = np.array(m_all.data, dtype=np.int8).reshape(m_all.info.height, m_all.info.width)
    gr = np.array(m_ref.data, dtype=np.int8).reshape(m_ref.info.height, m_ref.info.width)
    di = int(round((m_ref.info.origin.position.y - m_all.info.origin.position.y) / res))
    dj = int(round((m_ref.info.origin.position.x - m_all.info.origin.position.x) / res))
    r0, c0 = max(di, 0), max(dj, 0)
    r1, c1 = min(di + gr.shape[0], ga.shape[0]), min(dj + gr.shape[1], ga.shape[1])
    sub = gr[r0 - di:r1 - di, c0 - dj:c1 - dj]
    known = sub >= 0
    ga[r0:r1, c0:c1][known] = sub[known]
    m_all.data = ga.ravel().tolist()
    mask = np.zeros(ga.shape, bool)
    mask[r0:r1, c0:c1] = known
    return int(known.sum()), mask[::-1]


def point_obstacles(shape, res, origin, P, floor, step=0.06, top=0.8, min_frac=0.1):
    """grid cells (image layout, row 0 = top) the robot cannot drive over, judged from the points.

    OctoMap's projection only counts voxels above a fixed height (occupancy_min_z), so low platforms, boards and door
    thresholds below it look free. Here a cell is an obstacle when at least max(3, min_frac of the cell's points) lie
    between floor + `step` and floor + `top` (same support rule as the elevation map, so floor noise does not count).
    """
    h, w = shape
    rel = P[:, 2] - floor
    col = np.floor((P[:, 0] - origin[0]) / res).astype(np.int64)
    row = h - 1 - np.floor((P[:, 1] - origin[1]) / res).astype(np.int64)
    ok = (col >= 0) & (col < w) & (row >= 0) & (row < h) & (rel > -0.3) & (rel < top)
    cell = row[ok] * w + col[ok]
    cnt = np.bincount(cell, minlength=h * w)
    up = np.bincount(cell, weights=(rel[ok] > step).astype(float), minlength=h * w)
    return (up >= np.maximum(3, min_frac * cnt)).reshape(shape)


def save_grid(m, out, P=None, floor=None, despeckle=(12, 20), extra=None):
    """extra: callable (shape, res, origin) -> bool mask of additional obstacle cells (image layout)"""
    g = np.array(m.data, dtype=np.int16).reshape(m.info.height, m.info.width)
    img = np.full(g.shape, 205, np.uint8)
    img[g == 0] = 254
    img[g >= 65] = 0
    img = img[::-1].copy()
    o, q = m.info.origin.position, m.info.origin.orientation
    freed = 0
    if P is not None and despeckle and despeckle[0] > 0:
        freed = despeckle_grid(img, m.info.resolution, (o.x, o.y), P, floor, *despeckle)
    added = 0
    if extra is not None:
        add = extra(img.shape, m.info.resolution, (o.x, o.y)) & (img != 0)
        # tiny isolated additions (<= 3 cells, not touching an existing obstacle) are more likely noise than a platform
        from scipy import ndimage
        lab, n = ndimage.label(add | (img == 0), np.ones((3, 3)))
        if n:
            idx = np.arange(1, n + 1)
            size = ndimage.sum(np.ones(img.shape), lab, idx)
            had_occ = ndimage.maximum(img == 0, lab, idx)
            tiny = np.zeros(n + 1, bool)
            tiny[1:] = (size <= 3) & ~had_occ.astype(bool)
            add &= ~tiny[lab]
        img[add] = 0
        added = int(add.sum())
    write_pgm(os.path.join(out, 'grid.pgm'), img)
    yaw = float(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
    with open(os.path.join(out, 'grid.yaml'), 'w') as f:
        f.write(f'image: grid.pgm\nmode: trinary\nresolution: {m.info.resolution}\n'
                f'origin: [{o.x}, {o.y}, {yaw}]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n')
    return (f'{m.info.width}x{m.info.height} @ {m.info.resolution:.2f} m, {int((img == 0).sum())} occupied cells'
            + (f' ({freed} speck cells freed)' if freed else '')
            + (f' ({added} low-obstacle cells added from points)' if added else ''))


def despeckle_grid(img, res, origin, P, floor, max_cells=12, min_points=20, band=(0.1, 2.0)):
    """remove small occupied blobs of a grid image (row 0 = top, 0 = occupied) that the 3D map does not support.

    A blob (8-connected occupied cells) of at most `max_cells` cells is set free when fewer than `min_points` map
    points between floor + band[0] and floor + band[1] fall into its cells (octomap specks from noise / leftovers).
    Real thin objects (table legs, poles) are seen from many keyframes and keep their points. -> number of cells freed
    """
    from scipy import ndimage
    h, w = img.shape
    rel = P[:, 2] - floor
    s = (rel > band[0]) & (rel < band[1])
    col = np.floor((P[s, 0] - origin[0]) / res).astype(np.int64)
    row = h - 1 - np.floor((P[s, 1] - origin[1]) / res).astype(np.int64)
    ok = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    cnt = np.zeros(img.shape, np.int64)
    np.add.at(cnt, (row[ok], col[ok]), 1)
    lab, n = ndimage.label(img == 0, np.ones((3, 3)))
    if n == 0:
        return 0
    idx = np.arange(1, n + 1)
    size = ndimage.sum(np.ones_like(cnt), lab, idx)
    pts = ndimage.sum(cnt, lab, idx)
    drop = np.zeros(n + 1, bool)
    drop[1:] = (size <= max_cells) & (pts < min_points)
    m = drop[lab]
    img[m] = 254
    return int(m.sum())


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


def load_grid(d):
    """grid.pgm/yaml in d -> (image row 0 = top, resolution, (x0, y0)) or None"""
    try:
        import yaml
        with open(os.path.join(d, 'grid.yaml')) as f:
            y = yaml.safe_load(f)
        with open(os.path.join(d, y.get('image', 'grid.pgm')), 'rb') as f:
            f.readline()
            line = f.readline()
            while line.startswith(b'#'):
                line = f.readline()
            w, h = map(int, line.split())
            f.readline()
            img = np.frombuffer(f.read(w * h), np.uint8).reshape(h, w)
        return img, float(y['resolution']), (float(y['origin'][0]), float(y['origin'][1]))
    except (OSError, KeyError, ValueError):
        return None


def save_voxels(P, floor, out, size=0.1, top=2.0, ply=False, min_points=2, obj=0.06, grid_top=0.8):
    """voxels of the static map coloured by height above the floor (blue 0 m .. red `top` m).

    Layers follow the local floor (layer 0 = floor +-size/2), so floor noise does not spill into the layer above.
    A low voxel (layer 1) with nothing above it is kept only if its column is an obstacle by the elevation rule
    (at least max(3, 10 %) of the column's points more than `obj` above the floor); otherwise it is floor noise / a ring
    pattern and dropped.
    If grid.pgm/yaml exist in `out`, voxels between the floor layer and `grid_top` (the 2D map's height band) standing on
    cells the 2D map marks free (and not next to an obstacle) are dropped too, so the voxels agree with the 2D map (with --prefer: the reference
    session decides, leftovers of the other session go).
    """
    rel = P[:, 2] - floor
    kxy = np.floor(P[:, :2] / size).astype(np.int64)
    kz = np.round(rel / size).astype(np.int64)
    uk, inv, cnt = np.unique(np.c_[kxy, kz], axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    fl = np.bincount(inv, weights=floor) / cnt
    keep = cnt >= min_points
    # column statistics
    ucol, cinv = np.unique(kxy, axis=0, return_inverse=True)
    cinv = cinv.ravel()
    ccnt = np.bincount(cinv)
    up = np.bincount(cinv, weights=(rel > obj).astype(float), minlength=len(ucol))
    col_obj = up >= np.maximum(3, 0.1 * ccnt)
    vcol = np.zeros(len(uk), np.int64)
    vcol[inv] = cinv
    has_above = np.zeros(len(ucol), bool)
    has_above[vcol[keep & (uk[:, 2] >= 2)]] = True
    noise = keep & (uk[:, 2] == 1) & ~has_above[vcol] & ~col_obj[vcol]
    keep &= ~noise
    free = np.zeros(len(uk), bool)
    g = load_grid(out)
    if g is not None:
        img, res, (ox, oy) = g
        from scipy.ndimage import binary_dilation
        # only cells at least one cell away from any obstacle: voxel and grid cells are not aligned, so the edges of
        # walls and furniture must not be cut
        sure_free = (img == 254) & ~binary_dilation(img == 0, np.ones((3, 3), bool))
        cx, cy = (uk[:, 0] + 0.5) * size, (uk[:, 1] + 0.5) * size
        col = np.floor((cx - ox) / res).astype(np.int64)
        row = img.shape[0] - 1 - np.floor((cy - oy) / res).astype(np.int64)
        ok = (col >= 0) & (col < img.shape[1]) & (row >= 0) & (row < img.shape[0])
        val = np.zeros(len(uk), bool)
        val[ok] = sure_free[row[ok], col[ok]]
        free = keep & val & (uk[:, 2] >= 1) & (uk[:, 2] * size < grid_top)
        keep &= ~free
    c = np.c_[(uk[keep, :2] + 0.5) * size, fl[keep] + uk[keep, 2] * size]
    rgb = height_colors(uk[keep, 2] * size, 0.0, top)
    write_pcd_rgb(os.path.join(out, 'voxels.pcd'), c, rgb)
    msg = (f'{len(c)} voxels of {size} m ({int(noise.sum())} floor bumps, {int(free.sum())} on free 2D cells dropped), '
           f'coloured 0..{top} m above the floor')
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
    ap.add_argument('--grid-despeckle', type=int, default=12,
                    help='free occupied blobs of at most this many cells that map.pcd does not support; 0 = off')
    ap.add_argument('--grid-min-points', type=int, default=20, help='map points needed to keep such a blob')
    ap.add_argument('--grid-step', type=float, default=0.06,
                    help='cells with enough points this far above the floor are obstacles in grid.pgm (low platforms, '
                         'thresholds); 0 = octomap only')
    ap.add_argument('--grid-top', type=float, default=0.8, help='... up to this height (robot height + margin)')
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
        # --prefer: reference session first; its 2D map is kept wherever it observed (free or occupied), the other
        # session only fills cells the reference never saw -> leftovers of the other session do not reach grid.pgm
        ref = np.array([n.startswith(a.prefer) for n in kf.names]) if a.prefer else np.zeros(len(kf), bool)
        order = list(np.nonzero(ref)[0]) + list(np.nonzero(~ref)[0])
        for i, k in enumerate(order):
            if a.prefer and i == ref.sum():
                spin_for(4.0)
                got['grid_ref'] = got.get('grid')
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
            if (i + 1) % 100 == 0:
                print(f'  {i + 1}/{len(kf)}', flush=True)
        # octomap re-publishes the 2D map after every insertion (latched):
        # wait a few seconds so the last message (after the final keyframe) is the one kept in `got`
        spin_for(4.0)
        end = time.time() + 10
        while time.time() < end and 'grid' not in got:
            rclpy.spin_once(node, timeout_sec=0.1)

        if got.get('grid_ref') is not None and 'grid' in got:
            n_ref, got['ref_known'] = prefer_grid(got['grid_ref'], got['grid'])
            print(f'2D: {n_ref} cells from the reference session, the rest from the other session', flush=True)
        old = os.path.join(d, 'layers_old')
        os.makedirs(old, exist_ok=True)
        for f in OUT_FILES:
            src = os.path.join(d, f)
            if os.path.exists(src):
                shutil.move(src, os.path.join(old, f))
        st = ~dyn
        Ps = P[st]
        floor = local_floor(Ps, kf.t)
        ref_pt = ref[K[st]]

        def low_obstacles(shape, res, origin):
            if a.grid_step <= 0:
                return np.zeros(shape, bool)
            f = lambda sel: point_obstacles(shape, res, origin, Ps[sel], floor[sel], a.grid_step, a.grid_top)  # noqa: E731
            if not a.prefer:
                return f(np.ones(len(Ps), bool))
            known = got.get('ref_known')
            other = f(~ref_pt)
            if known is not None and known.shape == shape:
                other &= ~known                      # the other session only where the reference saw nothing
            return f(ref_pt) | other

        ev = {}

        def evidence():
            # final point-cloud map (after clean_floor if it was run): decides which small grid blobs are real and
            # is what voxels.pcd shows
            if not ev:
                mp = os.path.join(d, 'map.pcd')
                if os.path.isfile(mp):
                    Q, _ = read_pcd(mp)
                    ev['v'] = (Q, local_floor(Q, kf.t))
                else:
                    ev['v'] = (Ps, floor)
            return ev['v']

        def elevation():
            layers, x0, y0 = robust_elevation(
                Ps, floor, float(ep.get('resolution', 0.1)), float(ep.get('length_x', 40.0)),
                float(ep.get('length_y', 40.0)), float(ep.get('max_height_above_robot', 1.2)),
                float(ep.get('min_height_above_robot', -1.5)), a.elevation_obstacle, int(ep.get('min_points', 2)))
            return save_elevation_layers(layers, float(ep.get('resolution', 0.1)), x0, y0, 'map', d)
        for name, fn in (('octomap.bt', lambda: save_octomap(d)),
                         ('grid.pgm/yaml', lambda: save_grid(got['grid'], d, *evidence(),
                                                             (a.grid_despeckle, a.grid_min_points),
                                                             low_obstacles)),
                         ('elevation (+ elevation.png)', elevation),
                         ('voxels.pcd', lambda: save_voxels(*evidence(), d, a.voxel_size, a.voxel_color_max,
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
