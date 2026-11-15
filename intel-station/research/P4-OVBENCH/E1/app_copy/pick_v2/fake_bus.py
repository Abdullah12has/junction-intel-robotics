"""Fake bus for the calibration dry run: a driver stand-in and a synthetic top camera.

No serial port and no camera are opened. FakeFollower implements the driver calls that
so101_motion.MotionPlayer makes (get_observation, send_action, set_torque, disconnect and the
two raw-tick helpers used by hold). FakeTopCamera renders a red disc where the true jaw tip
projects, using a 'true' table-to-pixel homography and 'true' per-joint zero offsets that the
fit must recover without seeing them.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from .kinematics import Chain, FollowerCalibration, tcp_xyz
from .table_calib import apply_homography

H_TRUE = np.array([[0.0, -900.0, 320.0],
                   [900.0, 0.0, 40.0],
                   [0.2, -0.1, 1.0]])  # table XY (m) -> pixels (640 x 480 top view, base near the top centre)
B_TRUE = np.array([1.8, -1.2, 0.9, 1.5, -0.8])  # true zero offsets in normalized units (pan, lift, elbow, wflex, roll)
DRY_PLATES = ((0.10, 0.14, 0.06), (0.10, -0.14, 0.06))  # fake plate discs (x, y, radius) in base_link metres
IMAGE_SHAPE = (480, 640)


class FakeFollower:
    """Tracks every goal exactly. read_bias is added to the read-back only (to test the arrival check)."""

    max_speed = 279.0  # the driver's own estimate for the calibrated follower

    def __init__(self, start, *, read_bias=None) -> None:
        self._q = np.asarray(start, dtype=float).copy()
        self.read_bias = np.zeros(6) if read_bias is None else np.asarray(read_bias, dtype=float)
        self.torque = False
        self.connected = True
        self.sent = 0

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def set_torque(self, enabled: bool) -> None:
        self.torque = bool(enabled)

    def true_positions(self) -> np.ndarray:
        return self._q.copy()

    def get_observation(self) -> SimpleNamespace:
        return SimpleNamespace(joint_positions=self._q + self.read_bias)

    def send_action(self, action) -> None:
        self._q = np.asarray(action, dtype=float).copy()
        self.sent += 1

    def _read_joint_positions(self) -> np.ndarray:
        return np.round(self._q * 10.0).astype(np.int32)

    def _write_joint_positions(self, raw) -> None:
        self._q = np.asarray(raw, dtype=float) / 10.0


class FakeTopCamera:
    """Callable returning an RGB frame with a red disc at the true tip pixel (plus pixel noise)."""

    def __init__(self, follower: FakeFollower, chain: Chain, cal: FollowerCalibration,
                 H: np.ndarray, b_true, *, noise_px: float = 0.5, seed: int = 0) -> None:
        self.follower = follower
        self.chain, self.cal = chain, cal
        self.H = H
        self.b = np.asarray(b_true, dtype=float)
        self.noise_px = float(noise_px)
        self.rng = np.random.default_rng(seed)

    def tip_pixel(self) -> np.ndarray:
        n = self.follower.true_positions()[:5]
        xy = tcp_xyz(self.chain, self.cal, n + self.b)[:2]
        return apply_homography(self.H, xy[None])[0] + self.rng.normal(0.0, self.noise_px, 2)

    def __call__(self) -> np.ndarray:
        u, v = self.tip_pixel()
        h, w = IMAGE_SHAPE
        img = np.full((h, w, 3), 70, dtype=np.uint8)
        yy, xx = np.mgrid[0:h, 0:w]
        img[(xx - u) ** 2 + (yy - v) ** 2 <= 36.0] = (230, 25, 25)
        return img
