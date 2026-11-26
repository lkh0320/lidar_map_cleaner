"""Binary PCD read/write and the keyframe map-directory format.

Map directory layout (the format FAST_LIO / FASTLIO2_ROS2 PGO `save_maps` writes; easy to produce from any SLAM):
    <map_dir>/poses.txt     one line per keyframe:  <patch file name> x y z qw qx qy qz   (body pose in the map frame)
    <map_dir>/patches/*.pcd keyframe point clouds in the body frame
    <map_dir>/map.pcd       (optional) the merged map
"""
import os

import numpy as np
from scipy.spatial.transform import Rotation as Rot

_TYPES = {'F': 'f', 'U': 'u', 'I': 'i'}


def read_pcd(path):
    """-> (xyz float64 (N,3), intensity float32 (N,)). Binary PCD only."""
    with open(path, 'rb') as f:
        h = {}
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f'{path}: truncated header')
            line = line.decode('ascii', 'ignore').strip()
            if not line or line.startswith('#'):
                continue
            k = line.split()[0]
            h[k] = line.split()[1:]
            if k == 'DATA':
                break
        if h['DATA'][0] != 'binary':
            raise ValueError(f'{path}: only binary PCD is supported (got {h["DATA"][0]})')
        fields, sizes, types = h['FIELDS'], [int(s) for s in h['SIZE']], h['TYPE']
        counts = [int(c) for c in h.get('COUNT', ['1'] * len(fields))]
        n = int(h['POINTS'][0])
        dt = []
        for i, (name, s, t, c) in enumerate(zip(fields, sizes, types, counts)):
            nm = name if name != '_' else f'_pad{i}'
            dt.append((nm, f'{_TYPES[t]}{s}') if c == 1 else (nm, f'{_TYPES[t]}{s}', (c,)))
        a = np.frombuffer(f.read(n * sum(s * c for s, c in zip(sizes, counts))), dtype=np.dtype(dt), count=n)
    xyz = np.stack([a['x'], a['y'], a['z']], 1).astype(np.float64)
    inten = a['intensity'].astype(np.float32) if 'intensity' in a.dtype.names else np.zeros(n, np.float32)
    return xyz, inten


def write_pcd(path, xyz, intensity=None):
    n = len(xyz)
    a = np.zeros(n, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4')])
    a['x'], a['y'], a['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if intensity is not None:
        a['intensity'] = intensity
    with open(path, 'wb') as f:
        f.write((f'# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z intensity\n'
                 f'SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH {n}\nHEIGHT 1\n'
                 f'VIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n').encode())
        f.write(a.tobytes())


class Keyframes:
    """Keyframes of a map directory: names, body poses (R_wb, t_wb) and body-frame clouds (lazy)."""

    def __init__(self, map_dir):
        self.dir = map_dir
        self.names, t, q = [], [], []
        path = os.path.join(map_dir, 'poses.txt')
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} not found (see README: map directory format)')
        with open(path) as f:
            for line in f:
                s = line.split()
                if len(s) == 8:
                    self.names.append(s[0])
                    t.append([float(v) for v in s[1:4]])
                    q.append([float(s[5]), float(s[6]), float(s[7]), float(s[4])])  # file: w x y z
        if not self.names:
            raise ValueError(f'{path} has no keyframes (save with patches enabled)')
        self.t = np.array(t)
        self.R = Rot.from_quat(np.array(q))

    def __len__(self):
        return len(self.names)

    def cloud(self, k):
        return read_pcd(os.path.join(self.dir, 'patches', self.names[k]))
