"""Offline: remove moving objects from a saved keyframe map (visibility / "seen-through" test).

  ros2 run lidar_map_cleaner remove_dynamic <map_dir>             # writes map_static.pcd, map_dynamic.pcd
  ros2 run lidar_map_cleaner remove_dynamic <map_dir> --replace   # map.pcd <- static map (original kept as map_raw.pcd)

Every map voxel is projected into every keyframe's range image. If a keyframe's beam passed clearly beyond
the voxel ("seen-through") more often than it hit it, the voxel was empty at that time -> it held a moving object.
Only the band [floor + floor_margin, floor + max_height] is judged, where floor = z of the horizontally nearest
keyframe (body frame should be on the ground, e.g. base_footprint). Grazing floor/ceiling returns cause
false see-throughs, so they are never removed. Objects never re-observed from another viewpoint cannot be judged.
"""
import argparse
import os
import shutil

import numpy as np
from scipy.spatial import cKDTree

from .pcd_io import Keyframes, write_pcd
from .sensor_model import PRESETS, SensorModel


def add_args(ap):
    ap.add_argument('--preset', default='vlp16', help=f'beam layout preset: {sorted(PRESETS)}')
    ap.add_argument('--sensor-xyz', default='0,0,0', help='sensor position in the body frame x,y,z (m)')
    ap.add_argument('--sensor-rpy', default='0,0,0', help='sensor orientation in the body frame r,p,y (rad)')
    ap.add_argument('--max-range', type=float, default=15.0)
    ap.add_argument('--floor-margin', type=float, default=0.15)
    ap.add_argument('--max-height', type=float, default=2.2)
    ap.add_argument('--min-see', type=int, default=3, help='min number of see-through observations')
    ap.add_argument('--ratio', type=float, default=0.6, help='see / (see + hit) needed to call a voxel dynamic')
    ap.add_argument('--voxel', type=float, default=0.1)


def sensor_from_args(a):
    return SensorModel(**PRESETS[a.preset], xyz=[float(v) for v in a.sensor_xyz.split(',')],
                       rpy=[float(v) for v in a.sensor_rpy.split(',')])


def classify(kf: Keyframes, sensor: SensorModel, a, log=print):
    """-> P (map points), I, K (keyframe index per point), dyn (bool per point)"""
    P, I, K, imgs, origins, Rws, floor = [], [], [], [], [], [], []
    for k in range(len(kf)):
        xyz, inten = kf.cloud(k)
        R, t = kf.R[k], kf.t[k]
        P.append(R.apply(xyz) + t)
        I.append(inten)
        K.append(np.full(len(xyz), k, np.int32))
        imgs.append(sensor.range_images(sensor.body_to_sensor(xyz), erode_cols=0)[0])
        R_ws, o_ws = sensor.sensor_pose_in_world(R, t)
        Rws.append(R_ws)
        origins.append(o_ws)
        floor.append(t[2])
    P, I, K = np.concatenate(P), np.concatenate(I), np.concatenate(K)
    origins, floor = np.array(origins), np.array(floor)
    log(f'{len(kf)} keyframes, {len(P)} points')

    vkey = np.floor(P / a.voxel).astype(np.int64)
    _, vid, vinv = np.unique(vkey, axis=0, return_index=True, return_inverse=True)
    vinv = vinv.ravel()
    V = P[vid]
    _, nk = cKDTree(kf.t[:, :2]).query(V[:, :2], workers=-1)
    rel = V[:, 2] - floor[nk]
    cand = (rel > a.floor_margin) & (rel < a.max_height)
    Vc = V[cand]
    log(f'{len(V)} voxels, {len(Vc)} in the judged height band')

    see = np.zeros(len(Vc), np.int32)
    hit = np.zeros(len(Vc), np.int32)
    tree = cKDTree(Vc)
    for k in range(len(kf)):
        idx = np.array(tree.query_ball_point(origins[k], a.max_range), dtype=np.int64)
        if len(idx) == 0:
            continue
        r, row, col, ok = sensor.project(Rws[k].inv().apply(Vc[idx] - origins[k]), el_tol=min(0.5, 0.25 * sensor.v_res))
        ok &= r > 1.0
        rs = np.where(ok, imgs[k][row, col], np.inf)
        valid = ok & np.isfinite(rs)
        m = 0.15 + 0.02 * r
        see[idx[valid & (rs > r + m)]] += 1
        hit[idx[valid & (np.abs(rs - r) <= m)]] += 1
    dyn_c = (see >= a.min_see) & (see >= a.ratio * (see + hit))
    dyn_v = np.zeros(len(V), bool)
    dyn_v[np.nonzero(cand)[0][dyn_c]] = True
    dyn = dyn_v[vinv]
    log(f'dynamic points: {dyn.sum()} ({100 * dyn.mean():.2f} %)')
    return P, I, K, dyn


def main(argv=None):
    ap = argparse.ArgumentParser(description='Remove moving objects from a saved keyframe map')
    ap.add_argument('map_dir')
    add_args(ap)
    ap.add_argument('--replace', action='store_true', help='overwrite map.pcd with the static map (keeps map_raw.pcd)')
    a = ap.parse_args(argv)
    d = a.map_dir
    P, I, _, dyn = classify(Keyframes(d), sensor_from_args(a), a)
    write_pcd(os.path.join(d, 'map_dynamic.pcd'), P[dyn], I[dyn])
    if a.replace:
        if os.path.exists(os.path.join(d, 'map.pcd')) and not os.path.exists(os.path.join(d, 'map_raw.pcd')):
            shutil.copy(os.path.join(d, 'map.pcd'), os.path.join(d, 'map_raw.pcd'))
        write_pcd(os.path.join(d, 'map.pcd'), P[~dyn], I[~dyn])
        print('wrote map.pcd (static; original in map_raw.pcd), map_dynamic.pcd')
    else:
        write_pcd(os.path.join(d, 'map_static.pcd'), P[~dyn], I[~dyn])
        print('wrote map_static.pcd, map_dynamic.pcd')


if __name__ == '__main__':
    main()
