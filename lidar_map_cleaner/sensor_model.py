"""Spinning-LiDAR range-image model shared by the online filter and the offline tools.

A scan is projected into a (rows x cols) image: rows = elevation bins, cols = azimuth bins.
Each pixel keeps the *closest* return. Two images are kept per scan:
  - raw:     closest range in the pixel ("did the beam hit something at range r?")
  - eroded:  min over +-erode_cols azimuth neighbours ("did *every* nearby beam pass beyond r?")
The eroded image is what "seen-through" tests use: a point is only considered empty space if the beam
and its horizontal neighbours all went further, which suppresses false positives at object edges.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as Rot


@dataclass
class SensorModel:
    v_min: float = -15.0          # lowest beam elevation (deg)
    v_max: float = 15.0           # highest beam elevation (deg)
    v_res: float = 2.0            # elevation spacing between beams (deg)
    az_res: float = 1.0           # azimuth bin size (deg)
    el_tol: float = 0.3           # a point must lie within this of a beam's elevation to be tested (deg)
    # sensor pose in the robot body frame (the frame of the input clouds / keyframes)
    xyz: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rpy: list = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def __post_init__(self):
        self.rows = int(round((self.v_max - self.v_min) / self.v_res)) + 1
        self.cols = int(round(360.0 / self.az_res))
        self.t_bs = np.asarray(self.xyz, dtype=np.float64)
        self.R_bs = Rot.from_euler('xyz', self.rpy)

    # body <-> sensor
    def body_to_sensor(self, pb):
        return self.R_bs.inv().apply(pb - self.t_bs)

    def sensor_pose_in_world(self, R_wb, t_wb):
        """(R_ws, o_ws) of the sensor given the body pose"""
        return R_wb * self.R_bs, t_wb + R_wb.apply(self.t_bs)

    def project(self, ps, el_tol=None):
        """sensor-frame points -> (range, row, col, valid)"""
        el_tol = self.el_tol if el_tol is None else el_tol
        r = np.linalg.norm(ps, axis=1)
        el = np.degrees(np.arcsin(np.clip(ps[:, 2] / np.maximum(r, 1e-9), -1.0, 1.0)))
        row_f = (el - self.v_min) / self.v_res
        row = np.round(row_f).astype(np.int64)
        ok = (row >= 0) & (row < self.rows) & (np.abs(row_f - row) * self.v_res < el_tol)
        col = np.floor((np.degrees(np.arctan2(ps[:, 1], ps[:, 0])) + 180.0) / self.az_res).astype(np.int64) % self.cols
        return r, np.clip(row, 0, self.rows - 1), col, ok

    def range_images(self, ps, erode_cols=1):
        """sensor-frame scan -> (raw, eroded) closest-range images, inf where empty"""
        r, row, col, ok = self.project(ps, el_tol=max(self.el_tol, 0.4 * self.v_res))
        img = np.full((self.rows, self.cols), np.inf)
        np.minimum.at(img, (row[ok], col[ok]), r[ok])
        er = img.copy()
        for d in range(1, erode_cols + 1):
            er = np.minimum(er, np.minimum(np.roll(img, d, 1), np.roll(img, -d, 1)))
        return img, er


PRESETS = {
    # tested on the robot this package was developed on
    'vlp16': dict(v_min=-15.0, v_max=15.0, v_res=2.0),
    # beam layouts from datasheets, not field-tested with this package
    'vlp32c_uniform_approx': dict(v_min=-25.0, v_max=15.0, v_res=1.33),
    'os1_64': dict(v_min=-22.5, v_max=22.5, v_res=0.7),
    'os1_128': dict(v_min=-22.5, v_max=22.5, v_res=0.35),
    'hesai_xt32': dict(v_min=-16.0, v_max=15.0, v_res=1.0),
}
