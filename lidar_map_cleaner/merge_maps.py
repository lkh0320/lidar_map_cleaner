"""Offline: merge two mapping sessions (keyframe map directories) of the same place into one.

  ros2 run lidar_map_cleaner merge_maps <map_A> <map_B> <out_dir>            # B is aligned to A
  ros2 run lidar_map_cleaner merge_maps <map_A> <map_B> <out_dir> --check-only
  then:  remove_dynamic <out_dir> --replace  &&  build_layers <out_dir>

1. Global alignment (no initial guess needed): wall slices (z band, default 0.3-1.8 m) of both maps are rasterised
   to 2D occupancy images; B is rotated in yaw steps and the best translation per yaw is found by FFT
   cross-correlation. The best (yaw, x, y) seeds a point-to-plane ICP (6 DoF) against A.
2. Levelling (ground robots, --no-level to skip): if a session started with a wrong gravity estimate the whole map is
   tilted. A plane is fitted to all keyframe positions (a ground-level body frame touches the floor) and the merged
   map is rotated so that the floor is horizontal at z = 0.
3. Merging: B poses <- T_AB * poses; keyframe clouds are hard-linked (copied across file systems) as a_* / b_*.
   The output is a normal map directory (poses.txt + patches/).
4. Check: distance from B's wall points to A's local planes in the overlap.
Assumes the two maps overlap substantially and are each internally consistent (loop-closed); the merge is rigid.
"""
import argparse
import os
import shutil

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

from .pcd_io import read_pcd


def voxel(p, s):
    _, i = np.unique(np.floor(p / s).astype(np.int64), axis=0, return_index=True)
    return p[i]


def _raster(p2, lo, shape, res):
    img = np.zeros(shape, np.float32)
    ij = np.floor((p2 - lo) / res).astype(int)
    ok = (ij >= 0).all(1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    np.add.at(img, (ij[ok, 0], ij[ok, 1]), 1.0)
    return np.minimum(img, 3.0)


def global_align_2d(A, B, res=0.1, yaw_step=1.0, pad=25.0):
    """2D (yaw, x, y) aligning wall points B onto A -> (Rotation, t (3,), score)"""
    a2 = A[:, :2]
    lo = a2.min(0) - pad
    shape = tuple(np.ceil((a2.max(0) + pad - lo) / res).astype(int))
    FA = np.fft.rfft2(_raster(a2, lo, shape, res))
    cb, ca = B[:, :2].mean(0), a2.mean(0)
    best = (-1.0, 0.0, np.zeros(2))
    for yaw in np.arange(0.0, 360.0, yaw_step):
        R2 = Rot.from_euler('z', yaw, degrees=True).as_matrix()[:2, :2]
        FB = np.fft.rfft2(_raster((B[:, :2] - cb) @ R2.T + ca, lo, shape, res))
        corr = np.fft.irfft2(FA * np.conj(FB), s=shape)
        k = int(np.argmax(corr))
        if corr.flat[k] > best[0]:
            di, dj = np.unravel_index(k, shape)
            di = di - shape[0] if di > shape[0] // 2 else di
            dj = dj - shape[1] if dj > shape[1] // 2 else dj
            best = (float(corr.flat[k]), yaw, np.array([di, dj]) * res)
    score, yaw, d = best
    R = Rot.from_euler('z', yaw, degrees=True)
    return R, np.r_[ca - R.as_matrix()[:2, :2] @ cb + d, 0.0], score


def icp_point_to_plane(src, dst, R, t, iters=30, max_dist=(1.0, 0.5, 0.3, 0.2)):
    """refine (R, t) mapping src onto dst; returns R, t, dst normals, dst kd-tree"""
    tree = cKDTree(dst)
    _, nb = tree.query(dst, k=12, workers=-1)
    X = dst[nb] - dst[nb].mean(1, keepdims=True)
    N = np.linalg.eigh(np.einsum('nki,nkj->nij', X, X))[1][:, :, 0]
    for md in max_dist:
        for _ in range(iters):
            p = R.apply(src) + t
            d, j = tree.query(p, distance_upper_bound=md, workers=-1)
            v = np.isfinite(d)
            if v.sum() < 50:
                break
            n = N[j[v]]
            r = np.einsum('ij,ij->i', n, p[v] - dst[j[v]])
            J = np.hstack([np.cross(p[v], n), n])
            dx = -np.linalg.solve(J.T @ J + 1e-6 * np.eye(6), J.T @ r)
            Rd = Rot.from_rotvec(dx[:3])
            R, t = Rd * R, Rd.apply(t) + dx[3:]
            if np.linalg.norm(dx) < 1e-6:
                break
    return R, t, N, tree


def align(A, B, band=(0.3, 1.8), yaw_step=1.0, log=print):
    """full maps A, B (N,3) -> (R, t) with A ~ R B + t, plus overlap statistics"""
    A, B = voxel(A, 0.1), voxel(B, 0.1)
    wa = A[(A[:, 2] > band[0]) & (A[:, 2] < band[1])]
    wb = B[(B[:, 2] > band[0]) & (B[:, 2] < band[1])]
    R, t, score = global_align_2d(wa, wb, yaw_step=yaw_step)
    log(f'global: yaw {R.as_euler("zyx", degrees=True)[0]:.1f} deg, t ({t[0]:.2f}, {t[1]:.2f}), score {score:.0f}')
    R, t, N, tree = icp_point_to_plane(voxel(B, 0.2), A, R, t)
    e = R.as_euler('zyx', degrees=True)
    log(f'ICP: yaw {e[0]:.2f} pitch {e[1]:.2f} roll {e[2]:.2f} deg, t ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}) m')
    p = R.apply(wb) + t
    d, j = tree.query(p, distance_upper_bound=0.5, workers=-1)
    v = np.isfinite(d)
    pl = np.abs(np.einsum('ij,ij->i', N[j[v]], p[v] - tree.data[j[v]])) if v.any() else np.array([np.inf])
    stats = dict(overlap=float(v.mean()), median=float(np.median(pl)), p90=float(np.percentile(pl, 90)))
    log(f'overlap: {100 * stats["overlap"]:.0f} % of B wall points within 0.5 m of A, point-to-plane median '
        f'{100 * stats["median"]:.1f} cm, 90 % {100 * stats["p90"]:.1f} cm')
    return R, t, stats


def read_poses(d):
    out = []
    with open(os.path.join(d, 'poses.txt')) as f:
        for line in f:
            s = line.split()
            if len(s) == 8:
                out.append((s[0], np.array([float(v) for v in s[1:4]]),
                            Rot.from_quat([float(s[5]), float(s[6]), float(s[7]), float(s[4])])))
    return out


def level_from_keyframes(positions, log=print):
    """rotation + z shift that makes the plane through ground-level keyframe positions horizontal at z = 0"""
    A = np.c_[positions[:, :2], np.ones(len(positions))]
    c = np.linalg.lstsq(A, positions[:, 2], rcond=None)[0]
    res = positions[:, 2] - A @ c
    n = np.array([-c[0], -c[1], 1.0])
    n /= np.linalg.norm(n)
    axis = np.cross(n, [0.0, 0.0, 1.0])
    ang = np.arccos(np.clip(n[2], -1.0, 1.0))
    R = Rot.from_rotvec(axis / np.linalg.norm(axis) * ang) if ang > 1e-9 else Rot.identity()
    t = np.array([0.0, 0.0, -R.apply([0.0, 0.0, c[2]])[2]])
    log(f'levelling: floor tilt {np.degrees(ang):.2f} deg corrected, keyframe height residual RMS {100 * np.sqrt((res ** 2).mean()):.1f} cm')
    return R, t


def main(argv=None):
    ap = argparse.ArgumentParser(description='Merge two keyframe maps of the same place')
    ap.add_argument('map_a', help='reference session')
    ap.add_argument('map_b', help='session aligned onto A')
    ap.add_argument('out_dir')
    ap.add_argument('--map-file', default='map.pcd', help='merged cloud of each session used for alignment')
    ap.add_argument('--yaw-step', type=float, default=1.0)
    ap.add_argument('--band', default='0.3,1.8', help='z band (m) of the wall slice used for 2D alignment')
    ap.add_argument('--no-level', action='store_true', help='do not level the floor (non-ground-robot data)')
    ap.add_argument('--check-only', action='store_true', help='only align and report')
    a = ap.parse_args(argv)

    A = read_pcd(os.path.join(a.map_a, a.map_file))[0]
    B = read_pcd(os.path.join(a.map_b, a.map_file))[0]
    R, t, _ = align(A, B, band=tuple(float(v) for v in a.band.split(',')), yaw_step=a.yaw_step)
    if a.check_only:
        return 0

    PA, PB = read_poses(a.map_a), read_poses(a.map_b)
    LR, Lt = Rot.identity(), np.zeros(3)
    if not a.no_level:
        LR, Lt = level_from_keyframes(np.array([p for _, p, _ in PA] + [R.apply(p) + t for _, p, _ in PB]))
    os.makedirs(os.path.join(a.out_dir, 'patches'), exist_ok=True)
    lines = []
    for tag, src, poses, T in (('a', a.map_a, PA, None), ('b', a.map_b, PB, (R, t))):
        for fname, pos, q in poses:
            if T is not None:
                pos, q = T[0].apply(pos) + T[1], T[0] * q
            pos, q = LR.apply(pos) + Lt, LR * q
            name = f'{tag}_{fname}'
            dst = os.path.join(a.out_dir, 'patches', name)
            if not os.path.exists(dst):
                try:
                    os.link(os.path.join(src, 'patches', fname), dst)
                except OSError:
                    shutil.copy(os.path.join(src, 'patches', fname), dst)
            qx = q.as_quat()
            lines.append(f'{name} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {qx[3]:.8f} {qx[0]:.8f} {qx[1]:.8f} {qx[2]:.8f}\n')
    with open(os.path.join(a.out_dir, 'poses.txt'), 'w') as f:
        f.writelines(lines)
    fmt = lambda v: ', '.join(f'{x:.8f}' for x in v)
    with open(os.path.join(a.out_dir, 'merge_info.yaml'), 'w') as f:
        f.write(f'map_a: {os.path.abspath(a.map_a)}\nmap_b: {os.path.abspath(a.map_b)}\n'
                f'T_ab_quat_xyzw: [{fmt(R.as_quat())}]\nT_ab_t: [{fmt(t)}]\n'
                f'level_quat_xyzw: [{fmt(LR.as_quat())}]\nlevel_t: [{fmt(Lt)}]\n')
    print(f'merged {len(lines)} keyframes -> {a.out_dir}  (next: remove_dynamic --replace, build_layers)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
