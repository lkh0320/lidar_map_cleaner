"""Online moving-object detector (ROS-free core, used by the node and the tests).

Principle ("appeared in previously free space"):
  keep range images of the last few seconds of scans. Project each point of the new scan into them;
  if earlier beams travelled clearly *beyond* the point (the space was empty then), the point belongs to
  something that moved in. Enough such votes -> dynamic.
Robustness:
  - eroded range images (min over azimuth neighbours) -> no false hits at object edges
  - cluster check: a voxel needs >= min_cluster detected points
  - track: detected voxels stay dynamic for track_time s (a person who briefly stops is still removed);
    the track is refreshed only by real detections, so it cannot creep onto neighbouring walls
  - static protection: voxels that have been observed in >= persist_scans scans over >= persist_time s are
    never removed. Sparse clutter (chair legs, shelves) lets some beams through from some viewpoints and would
    otherwise be flagged; on the development data this raised precision from ~61 % to ~92 %.
Only points whose body-frame height is within [min_height, max_height] are judged (floor / ceiling pass through).
"""
from collections import deque

import numpy as np

from .sensor_model import SensorModel

DEFAULTS = dict(
    window=2.5, min_age=0.3, n_past=8,
    min_votes=2, min_ratio=0.7,
    min_height=0.15, max_height=2.2, min_range=0.5, max_range=15.0,
    erode_cols=1, margin=0.15,
    voxel=0.3, min_cluster=3, track_time=1.0,
    persist_scans=30, persist_time=3.0,
)


def vkey(pw, size):
    """world points -> int64 voxel keys (21 bits per axis)"""
    k = np.floor(pw / size).astype(np.int64) + (1 << 20)
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


class DynamicCore:
    def __init__(self, sensor: SensorModel = None, **kw):
        self.s = sensor or SensorModel()
        unknown = set(kw) - set(DEFAULTS)
        if unknown:
            raise ValueError(f'unknown parameters: {sorted(unknown)}')
        for k, v in dict(DEFAULTS, **kw).items():
            setattr(self, k, v)
        self.past = deque()   # (stamp, R_ws, o_ws, raw_img, eroded_img)
        self.track = {}       # voxel key -> last detection time
        self.occ = {}         # voxel key -> [n scans observed, first time]

    def process(self, stamp, R_wb, t_wb, pb):
        """stamp (s), body pose in world (scipy Rotation, (3,)), body-frame points (N,3) -> dynamic mask (N,)"""
        ps_self = self.s.body_to_sensor(pb)
        rng = np.linalg.norm(ps_self, axis=1)
        cand = (pb[:, 2] > self.min_height) & (pb[:, 2] < self.max_height) & \
               (rng > self.min_range) & (rng < self.max_range)
        pw = R_wb.apply(pb) + t_wb
        dyn = np.zeros(len(pb), bool)
        ci = np.nonzero(cand)[0]
        keys = vkey(pw[ci], self.voxel)

        # static protection
        protected = np.zeros(len(ci), bool)
        if self.persist_scans > 0 and len(ci):
            prot = []
            for k in np.unique(keys).tolist():
                e = self.occ.get(k)
                if e is None:
                    self.occ[k] = [1, stamp]
                else:
                    e[0] += 1
                    if e[0] >= self.persist_scans and stamp - e[1] >= self.persist_time:
                        prot.append(k)
            if prot:
                protected = np.isin(keys, np.array(prot, np.int64))

        # compare with past scans
        while self.past and stamp - self.past[0][0] > self.window:
            self.past.popleft()
        usable = [x for x in self.past if stamp - x[0] >= self.min_age]
        if len(usable) > self.n_past:
            usable = [usable[i] for i in np.linspace(0, len(usable) - 1, self.n_past).round().astype(int)]
        if usable and len(ci):
            see = np.zeros(len(ci), np.int32)
            hit = np.zeros(len(ci), np.int32)
            for _, R_ws, o_ws, img, er in usable:
                r, row, col, ok = self.s.project(R_ws.inv().apply(pw[ci] - o_ws))
                m = self.margin + 0.02 * r
                rs = np.where(ok, img[row, col], np.inf)
                re = np.where(ok, er[row, col], np.inf)
                see += ok & np.isfinite(re) & (re > r + m)
                hit += ok & np.isfinite(rs) & (np.abs(rs - r) <= m)
            det = (see >= self.min_votes) & (see >= self.min_ratio * (see + hit)) & ~protected
            if det.any():
                uk, cnt = np.unique(keys[det], return_counts=True)
                for k in uk[cnt >= self.min_cluster].tolist():
                    self.track[k] = stamp

        # remove everything (judged height band) inside currently tracked voxels
        if self.track and len(ci):
            self.track = {k: v for k, v in self.track.items() if stamp - v <= self.track_time}
            if self.track:
                dyn[ci[np.isin(keys, np.fromiter(self.track.keys(), np.int64)) & ~protected]] = True

        R_ws, o_ws = self.s.sensor_pose_in_world(R_wb, t_wb)
        self.past.append((stamp, R_ws, o_ws, *self.s.range_images(ps_self, self.erode_cols)))
        return dyn
