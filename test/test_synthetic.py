"""Synthetic tests: a ray-cast box room, a VLP-16-like sensor and a person-sized box.

Run without ROS:  python3 -m pytest test/ -q
"""
import os
import tempfile
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from lidar_map_cleaner.dynamic_core import DynamicCore
from lidar_map_cleaner.pcd_io import Keyframes, read_pcd, write_pcd
from lidar_map_cleaner.remove_dynamic import classify
from lidar_map_cleaner.sensor_model import PRESETS, SensorModel

SENSOR_XYZ = np.array([0.0, 0.0, 0.4])
ROOM = (6.0, 4.0, 3.0)                  # half-size x, half-size y, ceiling height


def _beams(az_step=0.6):
    el = np.radians(np.arange(-15, 16, 2.0))
    az = np.radians(np.arange(-180, 180, az_step))
    E, A = np.meshgrid(el, az, indexing='ij')
    return np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)


def _box_hit(o, d, lo, hi):
    with np.errstate(divide='ignore', invalid='ignore'):
        t1, t2 = (lo - o) / d, (hi - o) / d
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    return np.where((tmax >= tmin) & (tmax > 0), np.maximum(tmin, 0), np.inf)


def scan(robot_xy, boxes, yaw=0.0):
    """ray-cast a scan from the robot at robot_xy -> body-frame points"""
    R = Rot.from_euler('z', yaw)
    o = np.array([robot_xy[0], robot_xy[1], 0.0]) + SENSOR_XYZ
    d = R.apply(_beams())
    hx, hy, hz = ROOM
    with np.errstate(divide='ignore'):
        ts = [np.where(d[:, 0] > 0, (hx - o[0]) / d[:, 0], (-hx - o[0]) / d[:, 0]),
              np.where(d[:, 1] > 0, (hy - o[1]) / d[:, 1], (-hy - o[1]) / d[:, 1]),
              np.where(d[:, 2] > 0, (hz - o[2]) / d[:, 2], (0 - o[2]) / d[:, 2])]
    t = np.min(np.stack(ts, 1), 1)
    for lo, hi in boxes:
        t = np.minimum(t, _box_hit(o, d, np.array(lo), np.array(hi)))
    pw = o + d * t[:, None]
    return R.inv().apply(pw - np.array([robot_xy[0], robot_xy[1], 0.0]))


PERSON = ((2.0, -0.25, 0.0), (2.5, 0.25, 1.7))


def person_at(y):
    return ((2.0, y - 0.25, 0.0), (2.5, y + 0.25, 1.7))
CABINET = ((-3.0, 2.0, 0.0), (-2.0, 3.0, 1.2))


def test_online_filter_removes_walking_person_keeps_static():
    sensor = SensorModel(**PRESETS['vlp16'], xyz=SENSOR_XYZ.tolist())
    core = DynamicCore(sensor)
    removed_person, removed_static, total_person = 0, 0, 0
    for k in range(80):                                  # 8 s at 10 Hz, robot drives slowly along x
        st = k * 0.1
        x = -2.0 + 0.02 * k
        py = -3.0 + 0.8 * (st - 3.0)                     # person walks across the room (0.8 m/s) from t = 3 s
        person = person_at(py)
        boxes = [CABINET] + ([person] if 3.0 <= st and py < 3.5 else [])
        pb = scan((x, 0.0), boxes)
        dyn = core.process(st, Rot.identity(), np.array([x, 0.0, 0.0]), pb)
        pw = pb + np.array([x, 0.0, 0.0])
        on_person = np.all((pw >= np.array(person[0]) - 0.02) & (pw <= np.array(person[1]) + 0.02), 1) \
            if len(boxes) > 1 else np.zeros(len(pw), bool)
        judged = on_person & (pb[:, 2] > 0.2)            # the filter only judges the body height band
        if st >= 3.5:
            total_person += judged.sum()
            removed_person += (dyn & judged).sum()
        removed_static += (dyn & ~on_person).sum()
    assert total_person > 0
    assert removed_person / total_person > 0.8, removed_person / total_person
    assert removed_static == 0, removed_static


def _write_map(d, frames):
    os.makedirs(os.path.join(d, 'patches'))
    with open(os.path.join(d, 'poses.txt'), 'w') as f:
        for i, (xy, boxes) in enumerate(frames):
            pb = scan(xy, boxes)
            write_pcd(os.path.join(d, 'patches', f'{i}.pcd'), pb)
            f.write(f'{i}.pcd {xy[0]} {xy[1]} 0 1 0 0 0\n')


def test_offline_removes_person_trace():
    # robot drives a loop; the person stands at PERSON only during the first 5 keyframes
    path = [(-4 + 0.2 * i, -2.5) for i in range(40)] + [(4 - 0.2 * i, 2.5) for i in range(40)]
    frames = [(xy, [CABINET] + ([PERSON] if i < 8 else [])) for i, xy in enumerate(path)]
    a = SimpleNamespace(max_range=15.0, floor_margin=0.15, max_height=2.2, min_see=3, ratio=0.6, voxel=0.1)
    with tempfile.TemporaryDirectory() as d:
        _write_map(d, frames)
        P, _, _, dyn = classify(Keyframes(d), SensorModel(**PRESETS['vlp16'], xyz=SENSOR_XYZ.tolist()), a, log=lambda *_: None)
    on_person = np.all((P >= np.array(PERSON[0]) - 0.05) & (P <= np.array(PERSON[1]) + 0.05), 1)
    on_cab = np.all((P >= np.array(CABINET[0]) - 0.05) & (P <= np.array(CABINET[1]) + 0.05), 1)
    on_person &= P[:, 2] > 0.2                           # only the judged height band (floor under the feet is kept)
    assert on_person.sum() > 50
    assert dyn[on_person].mean() > 0.8, dyn[on_person].mean()
    assert dyn[on_cab].mean() < 0.05, dyn[on_cab].mean()
    walls = ~on_person & ~on_cab & (P[:, 2] > 0.2)
    assert dyn[walls].mean() < 0.01, dyn[walls].mean()


def test_pcd_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = np.random.default_rng(0).normal(size=(100, 3))
        write_pcd(os.path.join(d, 'a.pcd'), p, np.arange(100, dtype=np.float32))
        q, i = read_pcd(os.path.join(d, 'a.pcd'))
        assert np.allclose(p, q, atol=1e-6) and i[5] == 5
