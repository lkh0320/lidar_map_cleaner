"""Offline: tidy the floor of a cleaned point-cloud map (map.pcd).

  ros2 run lidar_map_cleaner clean_floor <map_dir>          # map.pcd <- tidied (original kept as map_before_floor.pcd)
  ros2 run lidar_map_cleaner clean_floor <map_dir> --out tidy.pcd

What remains after remove_dynamic is mostly at floor level, where nothing can be decided from visibility:
caster wheels, shoe soles, a few cm of floor mismatch between sessions (and LiDAR ring patterns on the floor).
  1. floor band (default): the floor surface (< --floor-keep above the local floor) is kept as recorded, including ring
     patterns; the thin band between --floor-keep and --floor-height, where the small leftovers sit, is emptied.
     --flatten instead replaces everything below --floor-height by one point per --cell at the floor height (also
     removes ring patterns; cells with fewer than --min-points points are dropped).
  2. despeckle: points between --floor-height and --band-top are grouped into 26-connected 5 cm voxel clusters;
     clusters of at most --despeckle voxels are removed (small isolated leftovers).
Trade-off: objects lower than --floor-height (cables, thresholds) lose their top / merge into the floor, and tiny real
objects in the band are removed. Only map.pcd is changed; build_layers works from the keyframes and is not affected.
"""
import argparse
import os
import shutil

import numpy as np
from scipy import ndimage

from .pcd_io import Keyframes, read_pcd, write_pcd
from .remove_dynamic import local_floor


def tidy(P, I, floor, floor_height=0.05, cell=0.05, min_points=2, despeckle=20, band_top=0.5, voxel=0.05,
         flatten=False, floor_keep=0.025):
    """-> (points, intensity, stats)"""
    rel = P[:, 2] - floor
    fl = rel < floor_height
    if not flatten:
        surf = rel < floor_keep
        rest, rest_i, rest_rel = P[~fl], I[~fl], rel[~fl]
        drop = _despeckle(rest, rest_rel, despeckle, band_top, voxel)
        out = np.vstack([rest[~drop], P[surf]])
        out_i = np.r_[rest_i[~drop], I[surf]].astype(np.float32)
        return out, out_i, dict(floor_in=int(fl.sum()), floor_out=int(surf.sum()), specks=int(drop.sum()))
    c = np.floor(P[fl, :2] / cell).astype(np.int64)
    _, inv, cnt = np.unique(c, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    n = len(cnt)
    xy = np.zeros((n, 2))
    np.add.at(xy, inv, P[fl, :2])
    fz = np.zeros(n)
    np.add.at(fz, inv, floor[fl])
    it = np.zeros(n)
    np.add.at(it, inv, I[fl])
    keep = cnt >= min_points
    flat = np.c_[xy[keep] / cnt[keep, None], fz[keep] / cnt[keep]]
    flat_i = (it[keep] / cnt[keep]).astype(np.float32)

    rest, rest_i, rest_rel = P[~fl], I[~fl], rel[~fl]
    drop = _despeckle(rest, rest_rel, despeckle, band_top, voxel)
    out = np.vstack([rest[~drop], flat])
    out_i = np.r_[rest_i[~drop], flat_i].astype(np.float32)
    return out, out_i, dict(floor_in=int(fl.sum()), floor_out=int(keep.sum()), specks=int(drop.sum()))


def _despeckle(rest, rest_rel, despeckle, band_top, voxel):
    drop = np.zeros(len(rest), bool)
    band = rest_rel < band_top
    if despeckle > 0 and band.any():
        ij = np.floor(rest[band] / voxel).astype(np.int64)
        ij -= ij.min(0)
        occ = np.zeros(tuple(ij.max(0) + 1), bool)
        occ[ij[:, 0], ij[:, 1], ij[:, 2]] = True
        lab, _ = ndimage.label(occ, structure=np.ones((3, 3, 3)))
        size = np.bincount(lab.ravel())[lab[ij[:, 0], ij[:, 1], ij[:, 2]]]
        drop[np.nonzero(band)[0][size <= despeckle]] = True
    return drop


def main(argv=None):
    ap = argparse.ArgumentParser(description='Flatten the floor and remove small leftovers of map.pcd')
    ap.add_argument('map_dir')
    ap.add_argument('--map-file', default='map.pcd')
    ap.add_argument('--out', default=None, help='write here instead of replacing map.pcd')
    ap.add_argument('--floor-height', type=float, default=0.05, help='top of the near-floor band handled here (m)')
    ap.add_argument('--floor-keep', type=float, default=0.025, help='floor surface kept as recorded below this (m)')
    ap.add_argument('--flatten', action='store_true', help='replace the whole band by an even floor (removes ring patterns)')
    ap.add_argument('--cell', type=float, default=0.05, help='flattened floor resolution (m)')
    ap.add_argument('--min-points', type=int, default=2)
    ap.add_argument('--despeckle', type=int, default=20, help='max cluster size (5 cm voxels) removed; 0 = off')
    ap.add_argument('--band-top', type=float, default=0.5, help='despeckle only below this height (m)')
    a = ap.parse_args(argv)
    src = os.path.join(a.map_dir, a.map_file)
    P, I = read_pcd(src)
    floor = local_floor(P, Keyframes(a.map_dir).t)
    out, out_i, st = tidy(P, I, floor, a.floor_height, a.cell, a.min_points, a.despeckle, a.band_top,
                          flatten=a.flatten, floor_keep=a.floor_keep)
    print(f'near-floor points {st["floor_in"]} -> {st["floor_out"]} ({"flattened" if a.flatten else "surface kept"}), '
          f'small clusters removed: {st["specks"]} points; '
          f'{len(P)} -> {len(out)} points')
    if a.out:
        write_pcd(a.out, out, out_i)
        print(f'wrote {a.out}')
    else:
        backup = os.path.join(a.map_dir, 'map_before_floor.pcd')
        # refresh the backup when map.pcd was regenerated (e.g. by remove_dynamic --replace) after the last run
        if not os.path.exists(backup) or os.path.getmtime(src) > os.path.getmtime(backup):
            shutil.copy2(src, backup)
        write_pcd(src, out, out_i)
        os.utime(backup)                                   # backup is now newer than map.pcd's source state
        print(f'wrote {src} (original in map_before_floor.pcd)')


if __name__ == '__main__':
    main()
