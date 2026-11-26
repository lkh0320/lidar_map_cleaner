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


def _write_session(d, path, boxes, T=None):
    """write a map dir (poses.txt, patches/, map.pcd) of `path`, expressed in a frame where world = T(session)"""
    os.makedirs(os.path.join(d, 'patches'))
    R_s, t_s = (Rot.identity(), np.zeros(3)) if T is None else T      # session frame -> world
    Ri, ti = R_s.inv(), -R_s.inv().apply(t_s)                          # world -> session frame
    allp = []
    with open(os.path.join(d, 'poses.txt'), 'w') as f:
        for i, xy in enumerate(path):
            pb = scan(xy, boxes)
            write_pcd(os.path.join(d, 'patches', f'{i}.pcd'), pb)
            pos = Ri.apply([xy[0], xy[1], 0.0]) + ti
            q = Ri.as_quat()
            f.write(f'{i}.pcd {pos[0]} {pos[1]} {pos[2]} {q[3]} {q[0]} {q[1]} {q[2]}\n')
            allp.append(Ri.apply(pb + np.array([xy[0], xy[1], 0.0])) + ti)
    write_pcd(os.path.join(d, 'map.pcd'), np.concatenate(allp))


def test_merge_recovers_known_transform_and_levels():
    from lidar_map_cleaner.merge_maps import main as merge_main
    shelf = ((3.0, -4.0, 0.0), (5.0, -3.4, 1.8))                       # breaks the room's symmetry
    boxes = [CABINET, shelf]
    path_a = [(-4 + 0.5 * i, -1.0) for i in range(17)]
    path_b = [(3.0, -3 + 0.5 * i) for i in range(13)] + [(3 - 0.5 * i, 2.5) for i in range(13)]
    tilt = Rot.from_euler('y', 1.5, degrees=True)                       # session A started with a wrong gravity estimate
    T_a = (tilt, np.zeros(3))
    T_b = (Rot.from_euler('z', 123.0, degrees=True), np.array([1.5, -2.0, 0.0]))
    with tempfile.TemporaryDirectory() as d:
        A, B, out = (os.path.join(d, n) for n in ('A', 'B', 'out'))
        _write_session(A, path_a, boxes, T_a)
        _write_session(B, path_b, boxes, T_b)
        assert merge_main([A, B, out]) == 0
        kf = Keyframes(out)
        # every keyframe must end up where it was in the (level) world, up to one global yaw/translation in the plane
        true = np.array([[x, y, 0.0] for x, y in path_a + path_b])
        est = kf.t
        assert np.abs(est[:, 2]).max() < 0.03, np.abs(est[:, 2]).max()      # floor levelled to z = 0
        ca, ce = true[:, :2].mean(0), est[:, :2].mean(0)
        H = (est[:, :2] - ce).T @ (true[:, :2] - ca)
        U, _, Vt = np.linalg.svd(H)
        R2 = Vt.T @ U.T
        err = np.linalg.norm((est[:, :2] - ce) @ R2.T + ca - true[:, :2], axis=1)
        assert err.max() < 0.05, err.max()


def test_prefer_reference_session():
    """A = tidy reference session, B = older session with a chair A saw as empty, and a box A never observed"""
    chair = ((-3.0, -1.5, 0.0), (-2.5, -1.0, 0.9))       # only in B, inside A's view
    far_box = ((3.5, 1.0, 0.0), (4.3, 1.8, 1.0))         # only in B, beyond A's range -> fills A's gap
    path_a = [(-5 + 0.2 * i, -2.5) for i in range(12)]   # A stays on the left side
    path_b = [(-3.5 + 0.25 * i, 0.5) for i in range(30)]
    a = SimpleNamespace(max_range=5.0, floor_margin=0.15, max_height=2.2, min_see=3, ratio=0.6, voxel=0.1,
                        prefer='a_', prefer_min_see=2, prefer_ratio=0.5)
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, 'patches'))
        with open(os.path.join(d, 'poses.txt'), 'w') as f:
            for tag, path, boxes in (('a', path_a, [CABINET]), ('b', path_b, [CABINET, chair, far_box])):
                for i, xy in enumerate(path):
                    write_pcd(os.path.join(d, 'patches', f'{tag}_{i}.pcd'), scan(xy, boxes))
                    f.write(f'{tag}_{i}.pcd {xy[0]} {xy[1]} 0 1 0 0 0\n')
        P, _, _, dyn = classify(Keyframes(d), SensorModel(**PRESETS['vlp16'], xyz=SENSOR_XYZ.tolist()), a, log=lambda *_: None)
    inside = lambda box: np.all((P >= np.array(box[0]) - 0.05) & (P <= np.array(box[1]) + 0.05), 1) & (P[:, 2] > 0.2)
    assert inside(chair).sum() > 30 and inside(far_box).sum() > 30
    assert dyn[inside(chair)].mean() > 0.8, dyn[inside(chair)].mean()      # tidied away in the reference -> removed
    assert dyn[inside(far_box)].mean() < 0.1, dyn[inside(far_box)].mean()  # reference never saw it -> kept
    walls = ~inside(chair) & ~inside(far_box) & ~inside(CABINET) & (P[:, 2] > 0.2)
    assert dyn[walls].mean() < 0.01, dyn[walls].mean()
