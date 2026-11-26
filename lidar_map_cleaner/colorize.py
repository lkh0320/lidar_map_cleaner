"""Dependency-free colour output: turbo colormap, PNG writer, RGB point clouds and voxel cube meshes."""
import struct
import zlib

import numpy as np


# Google's Turbo colormap sampled at 33 points (Apache-2.0), linearly interpolated
_TURBO = np.array([
    [48, 18, 59],
    [57, 42, 115],
    [64, 64, 162],
    [68, 86, 199],
    [70, 107, 227],
    [70, 128, 246],
    [66, 148, 255],
    [55, 168, 250],
    [40, 188, 235],
    [28, 205, 216],
    [24, 221, 194],
    [31, 233, 175],
    [50, 242, 152],
    [78, 249, 125],
    [109, 254, 98],
    [139, 255, 75],
    [164, 252, 60],
    [185, 246, 53],
    [205, 236, 52],
    [223, 223, 55],
    [238, 207, 58],
    [248, 190, 57],
    [253, 172, 52],
    [254, 150, 43],
    [251, 126, 33],
    [244, 102, 23],
    [235, 80, 14],
    [223, 63, 8],
    [208, 47, 5],
    [190, 33, 2],
    [169, 22, 1],
    [146, 11, 1],
    [122, 4, 3]], dtype=np.float64)


def turbo(x):
    """x in [0, 1] -> (N, 3) uint8 Turbo colours"""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0) * (len(_TURBO) - 1)
    i = np.minimum(x.astype(int), len(_TURBO) - 2)
    t = (x - i)[..., None]
    return np.round(_TURBO[i] * (1 - t) + _TURBO[i + 1] * t).astype(np.uint8)


def height_colors(z, zmin, zmax):
    return turbo((np.asarray(z) - zmin) / max(zmax - zmin, 1e-6))


def write_png(path, rgba):
    """(H, W, 3|4) uint8 -> PNG (row 0 = top)"""
    rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
    h, w, c = rgba.shape
    raw = b''.join(b'\x00' + rgba[i].tobytes() for i in range(h))

    def chunk(tag, data):
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
    color_type = 6 if c == 4 else 2
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n')
        f.write(chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, color_type, 0, 0, 0)))
        f.write(chunk(b'IDAT', zlib.compress(raw, 6)))
        f.write(chunk(b'IEND', b''))


def write_pcd_rgb(path, xyz, rgb):
    """XYZRGB binary PCD (PCL packed rgb), viewable in pcl_viewer / RViz / CloudCompare"""
    n = len(xyz)
    a = np.zeros(n, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('rgb', 'u4')])
    a['x'], a['y'], a['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rgb = rgb.astype(np.uint32)
    a['rgb'] = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
    with open(path, 'wb') as f:
        f.write((f'# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\n'
                 f'TYPE F F F U\nCOUNT 1 1 1 1\nWIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n').encode())
        f.write(a.tobytes())


_CUBE_V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float) - 0.5
_CUBE_F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                    [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], np.int32)


def write_voxel_ply(path, centers, size, rgb):
    """coloured cube per voxel -> binary PLY mesh (MeshLab / CloudCompare / Blender)"""
    n = len(centers)
    v = (centers[:, None, :] + _CUBE_V[None] * size).reshape(-1, 3).astype(np.float32)
    col = np.repeat(rgb.astype(np.uint8), 8, axis=0)
    f = (_CUBE_F[None] + 8 * np.arange(n, dtype=np.int32)[:, None, None]).reshape(-1, 3)
    vert = np.zeros(len(v), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    vert['x'], vert['y'], vert['z'] = v[:, 0], v[:, 1], v[:, 2]
    vert['r'], vert['g'], vert['b'] = col[:, 0], col[:, 1], col[:, 2]
    face = np.zeros(len(f), dtype=[('n', 'u1'), ('i', '<i4', (3,))])
    face['n'], face['i'] = 3, f
    with open(path, 'wb') as fh:
        fh.write((f'ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\nproperty float x\nproperty float y\n'
                  f'property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n'
                  f'element face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n').encode())
        fh.write(vert.tobytes())
        fh.write(face.tobytes())


def voxelize(points, size, min_points=2):
    """-> voxel centers (M, 3) with at least min_points points"""
    k = np.floor(points / size).astype(np.int64)
    u, cnt = np.unique(k, axis=0, return_counts=True)
    return (u[cnt >= min_points] + 0.5) * size
