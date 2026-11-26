"""Offline: tidy the floor of a cleaned point-cloud map (map.pcd).

  ros2 run lidar_map_cleaner clean_floor <map_dir>          # map.pcd <- tidied (original kept as map_before_floor.pcd)
  ros2 run lidar_map_cleaner clean_floor <map_dir> --out tidy.pcd

What remains after remove_dynamic is mostly at floor level, where nothing can be decided from visibility:
LiDAR ring patterns, caster wheels, shoe soles, a few cm of floor mismatch between sessions. Two steps:
  1. flatten: points lower than --floor-height above the local floor are replaced by one point per --cell (xy) at the
     floor height (cells with fewer than --min-points points are dropped). The floor becomes an even surface.
  2. despeckle: with the floor noise gone, points between --floor-height and --band-top are grouped into 26-connected
     5 cm voxel clusters; clusters of at most --despeckle voxels are removed (small isolated leftovers).
Trade-off: objects lower than --floor-height (cables, thresholds) merge into the floor, and tiny real objects in the
band are removed. Only map.pcd is changed; build_layers works from the keyframes and is not affected.
"""
import argparse
import os
import shutil

import numpy as np
from scipy import ndimage

from .pcd_io import Keyframes, read_pcd, write_pcd
from .remove_dynamic import local_floor


def tidy(P, I, floor, floor_height=0.05, cell=0.05, min_points=2, despeckle=20, band_top=0.5, voxel=0.05):
    """-> (points, intensity, stats)"""
    rel = P[:, 2] - floor
    fl = rel < floor_height
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
    drop = np.zeros(len(rest), bool)
    if despeckle > 0:
        band = rest_rel < band_top
        if band.any():
            ij = np.floor(rest[band] / voxel).astype(np.int64)
            ij -= ij.min(0)
            occ = np.zeros(tuple(ij.max(0) + 1), bool)
            occ[ij[:, 0], ij[:, 1], ij[:, 2]] = True
            lab, _ = ndimage.label(occ, structure=np.ones((3, 3, 3)))
            size = np.bincount(lab.ravel())[lab[ij[:, 0], ij[:, 1], ij[:, 2]]]
            drop[np.nonzero(band)[0][size <= despeckle]] = True
    out = np.vstack([rest[~drop], flat])
    out_i = np.r_[rest_i[~drop], flat_i].astype(np.float32)
    return out, out_i, dict(floor_in=int(fl.sum()), floor_out=int(keep.sum()), specks=int(drop.sum()))


def main(argv=None):
    ap = argparse.ArgumentParser(description='Flatten the floor and remove small leftovers of map.pcd')
    ap.add_argument('map_dir')
    ap.add_argument('--map-file', default='map.pcd')
    ap.add_argument('--out', default=None, help='write here instead of replacing map.pcd')
    ap.add_argument('--floor-height', type=float, default=0.05, help='points below this above the floor are flattened (m)')
    ap.add_argument('--cell', type=float, default=0.05, help='flattened floor resolution (m)')
    ap.add_argument('--min-points', type=int, default=2)
    ap.add_argument('--despeckle', type=int, default=20, help='max cluster size (5 cm voxels) removed; 0 = off')
    ap.add_argument('--band-top', type=float, default=0.5, help='despeckle only below this height (m)')
    a = ap.parse_args(argv)
    src = os.path.join(a.map_dir, a.map_file)
    P, I = read_pcd(src)
    floor = local_floor(P, Keyframes(a.map_dir).t)
    out, out_i, st = tidy(P, I, floor, a.floor_height, a.cell, a.min_points, a.despeckle, a.band_top)
    print(f'floor points {st["floor_in"]} -> {st["floor_out"]} (flattened), small clusters removed: {st["specks"]} points; '
          f'{len(P)} -> {len(out)} points')
    if a.out:
        write_pcd(a.out, out, out_i)
        print(f'wrote {a.out}')
    else:
        backup = os.path.join(a.map_dir, 'map_before_floor.pcd')
        if not os.path.exists(backup):
            shutil.copy(src, backup)
        write_pcd(src, out, out_i)
        print(f'wrote {src} (original in map_before_floor.pcd)')


if __name__ == '__main__':
    main()
