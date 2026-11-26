"""Offline: remove moving objects from a saved keyframe map (visibility / "seen-through" test).

  ros2 run lidar_map_cleaner remove_dynamic <map_dir>             # writes map_static.pcd, map_dynamic.pcd
  ros2 run lidar_map_cleaner remove_dynamic <map_dir> --replace   # map.pcd <- static map (original kept as map_raw.pcd)

Every map voxel is projected into every keyframe's range image. If a keyframe's beam passed clearly beyond
the voxel ("seen-through") more often than it hit it, the voxel was empty at that time -> it held a moving object.
Only the band [floor + floor_margin, floor + max_height] is judged, where floor = z of the horizontally nearest
keyframe (body frame should be on the ground, e.g. base_footprint). Grazing floor/ceiling returns cause
false see-throughs, so they are never removed. Objects never re-observed from another viewpoint cannot be judged.

Merged multi-session maps (merge_maps names keyframes a_* / b_*): --prefer a makes session A the reference.
  - A's points are judged with A's keyframes only (only A's own moving objects are removed)
  - other points are removed when A saw their place as free (e.g. chairs that were tidied away before recording A)
    or by the normal all-keyframe test; where A never looked they are kept, so B fills A's gaps.

Near-floor band (floor + 0.04 m .. floor_margin), which the visibility test cannot judge (--no-low-band to skip):
  - "feet": in a 0.1 m cell whose 0.15-0.6 m part was removed entirely, the same session's low points go too
    (feet of people, bottoms of moved objects)
  - with --prefer: other-session low points are removed where the reference session saw the floor around them but has
    no structure (> 0.08 m above floor) within 0.08 m (chair bases and floor duplicates of the other session)
  The floor height is estimated per cell from the points themselves (low percentile), not from keyframe heights.

Leftovers (--no-remnants to skip): static points 3 cm .. 1 m above the floor are grouped into 26-connected 5 cm voxel
clusters. A cluster at most 0.6 m wide and 0.5 m tall is removed when >= 30 % of its points lie within 0.15 m of
removed points (a remnant of a person / moved object). Isolated objects that were never next to anything removed
(bins, extinguishers, sparsely seen far structure) are kept, whatever their size. This runs inside the classification, so build_layers' 2D and
elevation layers get it too.
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
    ap.add_argument('--prefer', default=None, help='keyframe name prefix of the reference session (e.g. a) in a merged map')
    ap.add_argument('--prefer-min-see', type=int, default=2, help='see-throughs by the reference session to remove other points')
    ap.add_argument('--prefer-ratio', type=float, default=0.5)
    ap.add_argument('--no-low-band', action='store_true', help='do not clean the near-floor band')
    ap.add_argument('--no-remnants', action='store_true', help='do not remove small leftover clusters')


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

    prefer = getattr(a, 'prefer', None)
    ref_kf = np.array([n.startswith(prefer) for n in kf.names]) if prefer else np.zeros(len(kf), bool)
    if prefer:
        log(f'reference session "{prefer}*": {ref_kf.sum()} of {len(kf)} keyframes')
        if not ref_kf.any():
            raise ValueError(f'no keyframe name starts with "{prefer}"')
    G = ref_kf[K]                              # point comes from the reference session
    vkey = np.c_[np.floor(P / a.voxel).astype(np.int64), G]     # voxels per session
    _, vid, vinv = np.unique(vkey, axis=0, return_index=True, return_inverse=True)
    vinv = vinv.ravel()
    V = P[vid]
    _, nk = cKDTree(kf.t[:, :2]).query(V[:, :2], workers=-1)
    rel = V[:, 2] - floor[nk]
    cand = (rel > a.floor_margin) & (rel < a.max_height)
    Vc = V[cand]
    log(f'{len(V)} voxels, {len(Vc)} in the judged height band')

    Gc = G[vid][cand]
    see = np.zeros(len(Vc), np.int32)
    hit = np.zeros(len(Vc), np.int32)
    see_r = np.zeros(len(Vc), np.int32)        # evidence from reference-session keyframes only
    hit_r = np.zeros(len(Vc), np.int32)
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
        s_, h_ = idx[valid & (rs > r + m)], idx[valid & (np.abs(rs - r) <= m)]
        see[s_] += 1
        hit[h_] += 1
        if ref_kf[k]:
            see_r[s_] += 1
            hit_r[h_] += 1
    if prefer:
        own = (see_r >= a.min_see) & (see_r >= a.ratio * (see_r + hit_r))
        by_ref = (see_r >= a.prefer_min_see) & (see_r >= a.prefer_ratio * (see_r + hit_r))
        normal = (see >= a.min_see) & (see >= a.ratio * (see + hit))
        dyn_c = np.where(Gc, own, by_ref | normal)
        log(f'other-session voxels removed because the reference saw free space: {int((by_ref & ~Gc).sum())}')
    else:
        dyn_c = (see >= a.min_see) & (see >= a.ratio * (see + hit))
    dyn_v = np.zeros(len(V), bool)
    dyn_v[np.nonzero(cand)[0][dyn_c]] = True
    dyn = dyn_v[vinv]
    rel = None
    if not getattr(a, 'no_low_band', False):
        dyn, rel = clean_low_band(P, dyn, G, kf, a, prefer, log)
    if not getattr(a, 'no_remnants', False):
        if rel is None:
            rel = P[:, 2] - local_floor(P, kf.t)
        dyn = remove_remnants(P, dyn, rel, log)
    log(f'dynamic points: {dyn.sum()} ({100 * dyn.mean():.2f} %)')
    return P, I, K, dyn


def local_floor(P, kf_xyz, res=0.2):
    """per-point floor height: 10th percentile of points near keyframe height in each cell, min over 3x3 cells"""
    from scipy.ndimage import minimum_filter
    _, nk = cKDTree(kf_xyz[:, :2]).query(P[:, :2], workers=-1)
    near = np.abs(P[:, 2] - kf_xyz[nk, 2]) < 0.3
    ij = np.floor(P[:, :2] / res).astype(np.int64)
    ij -= ij.min(0)
    shp = tuple(ij.max(0) + 1)
    grid = np.full(shp, np.inf)
    idx = np.nonzero(near)[0]
    idx = idx[np.lexsort((P[idx, 2], ij[idx, 1], ij[idx, 0]))]
    key = ij[idx, 0] * shp[1] + ij[idx, 1]
    uk, st, cnt = np.unique(key, return_index=True, return_counts=True)
    grid.flat[uk] = P[idx[st + (cnt * 0.1).astype(int)], 2]
    grid = minimum_filter(grid, size=3)
    f = grid[ij[:, 0], ij[:, 1]]
    return np.where(np.isfinite(f), f, kf_xyz[nk, 2])


def clean_low_band(P, dyn, G, kf, a, prefer, log):
    from scipy.ndimage import uniform_filter
    rel = P[:, 2] - local_floor(P, kf.t)
    low = (rel > 0.04) & (rel < a.floor_margin) & ~dyn
    c = np.floor(P[:, :2] / 0.1).astype(np.int64)
    key = c[:, 0] * 2000003 + c[:, 1]
    upper = (rel >= a.floor_margin) & (rel < 0.6)
    feet = np.zeros(len(P), bool)
    for g in (True, False):
        sel = G == g
        gone = np.setdiff1d(np.unique(key[upper & dyn & sel]), np.unique(key[upper & ~dyn & sel]))
        feet |= low & sel & np.isin(key, gone)
    out = dyn | feet
    n_ref = 0
    if prefer:
        ref_struct = P[G & ~out & (rel > 0.08)]
        cand = np.nonzero(low & ~G & ~feet)[0]
        if len(ref_struct) and len(cand):
            d, _ = cKDTree(ref_struct).query(P[cand], workers=-1)
            ij = np.floor((P[:, :2] - P[:, :2].min(0) + 1.0) / 0.1).astype(np.int64)
            grid = np.zeros(tuple(ij.max(0) + 4), np.float32)
            fl = G & (np.abs(rel) < 0.03)
            np.add.at(grid, (ij[fl, 0], ij[fl, 1]), 1.0)
            cov = uniform_filter(grid, size=7) * 49           # reference floor points within ~0.35 m
            r = cand[(d >= 0.08) & (cov[ij[cand, 0], ij[cand, 1]] >= 3)]
            out[r] = True
            n_ref = len(r)
    log(f'near-floor band: {int(feet.sum())} feet/bottom points' +
        (f', {n_ref} other-session points the reference saw as floor' if prefer else ''))
    return out, rel


def remove_remnants(P, dyn, rel, log, voxel=0.05, max_width=0.6, max_height=0.5, near=0.15, frac=0.3):
    from scipy import ndimage
    idx = np.nonzero(~dyn & (rel > 0.03) & (rel < 1.0))[0]
    if len(idx) == 0 or not dyn.any():
        return dyn
    ij = np.floor(P[idx] / voxel).astype(np.int64)
    ij -= ij.min(0)
    occ = np.zeros(tuple(ij.max(0) + 1), bool)
    occ[ij[:, 0], ij[:, 1], ij[:, 2]] = True
    lab, _ = ndimage.label(occ, structure=np.ones((3, 3, 3)))
    pl = lab[ij[:, 0], ij[:, 1], ij[:, 2]]
    sizes = np.bincount(lab.ravel())
    small = []
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        w = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) * voxel
        if w <= max_width and (sl[2].stop - sl[2].start) * voxel <= max_height:
            small.append(k)
    small = np.array(small, np.int64)
    out = dyn.copy()
    if len(small) == 0:
        return out
    in_small = np.isin(pl, small)
    d, _ = cKDTree(P[dyn]).query(P[idx[in_small]], distance_upper_bound=near, workers=-1)
    close = np.zeros(len(sizes))
    np.add.at(close, pl[in_small], np.isfinite(d).astype(float))
    npts = np.bincount(pl[in_small], minlength=len(sizes)).astype(float)
    ratio = close / np.maximum(npts, 1)
    kill = small[ratio[small] >= frac]
    out[idx[np.isin(pl, kill)]] = True
    log(f'leftover clusters removed: {len(kill)} ({int(np.isin(pl, kill).sum())} points), '
        f'kept {len(small) - len(kill)} small standalone objects')
    return out


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
