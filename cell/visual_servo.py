"""Offline-testable wrist alignment. No device access and no model weight edits.

The measured Jacobian maps normalized joint changes to target-pixel changes in
this wrist camera. Only local, planar alignment is supported; not depth control.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')
LOW = np.array([-99.9] * 5 + [0.1])
HIGH = np.full(6, 99.9)


def vector(value, size):
    value = np.asarray(value, float)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f'Expected finite vector of length {size}')
    return value


def validate_chunk(actions):
    actions = np.asarray(actions, float)
    if actions.ndim != 2 or actions.shape[1] != 6 or len(actions) == 0 or not np.isfinite(actions).all():
        raise ValueError('Invalid action chunk')
    return actions


def bounded_target(raw, previous, speed, dt):
    raw, previous = vector(raw, 6), vector(previous, 6)
    if not np.isfinite(speed) or speed <= 0 or not np.isfinite(dt) or not 0 < dt <= .1:
        raise ValueError('Invalid speed or control interval')
    if np.any(previous < LOW) or np.any(previous > HIGH):
        raise ValueError('Previous target outside calibrated range')
    return previous + np.clip(np.clip(raw, LOW, HIGH) - previous, -speed * dt, speed * dt)


def fit_jacobian(joint_deltas, pixel_deltas, joints):
    """Fit from operator-measured isolated perturbations; NEVER move hardware here.

Each row is relative to the same reference pose, with a stationary target.
Reserve independent samples for validation before enabling a calibration.
"""
    x, y = np.asarray(joint_deltas, float), np.asarray(pixel_deltas, float)
    if len(set(joints)) != len(joints) or not 2 <= len(joints) <= 5 or any(j not in JOINTS[:5] for j in joints):
        raise ValueError('Choose 2–5 distinct arm joints, excluding the gripper')
    if x.ndim != 2 or x.shape[1] != len(joints) or y.shape != (len(x), 2) or len(x) < 2 * len(joints):
        raise ValueError('Insufficient calibration samples')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.linalg.matrix_rank(x) < len(joints):
        raise ValueError('Nonfinite or rank-deficient samples')
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    jac = coef.T
    if np.linalg.matrix_rank(jac) != 2 or np.linalg.cond(jac) > 30:
        raise ValueError('Pixel control is ill-conditioned')
    return jac, float(np.sqrt(np.mean((x @ coef - y) ** 2)))


@dataclass
class Calibration:
    camera_id: str
    image_size: tuple
    reference_state: np.ndarray
    reference_uv: np.ndarray
    jacobian: np.ndarray
    joint_indices: list
    pose_radius: float
    pixel_radius: float
    tolerance: float
    mapping_sha256: str

    @classmethod
    def load(cls, path, camera_id, image_size, mapping_sha256):
        c = json.loads(Path(path).read_text())
        if c.get('validated') is not True or c.get('schema') != 1:
            raise ValueError('Measured, independently validated calibration required')
        if (c['camera_id'], tuple(c['image_size']), c['mapping_sha256']) != (camera_id, tuple(image_size), mapping_sha256):
            raise ValueError('Camera, resolution or joint mapping differs from calibration')
        joints = c['joints']
        if len(set(joints)) != len(joints) or any(j not in JOINTS[:5] for j in joints):
            raise ValueError('Invalid correction joints')
        j = np.asarray(c['jacobian'], float)
        if j.shape != (2, len(joints)) or not np.isfinite(j).all() or np.linalg.matrix_rank(j) != 2 or np.linalg.cond(j) > 30:
            raise ValueError('Invalid Jacobian')
        for key in ('pose_radius', 'pixel_radius', 'tolerance', 'validation_rmse_px'):
            if not np.isfinite(c[key]) or c[key] <= 0:
                raise ValueError('Invalid calibration domain or validation error')
        if c['validation_rmse_px'] > c['tolerance'] / 2 or c['tolerance'] >= c['pixel_radius']:
            raise ValueError('Calibration precision insufficient for requested tolerance')
        return cls(camera_id, tuple(image_size), vector(c['reference_state'], 6), vector(c['reference_uv'], 2), j,
                   [JOINTS.index(n) for n in joints], c['pose_radius'], c['pixel_radius'], c['tolerance'], mapping_sha256)


class AlignmentController:
    """Bounded local alignment, stopping on lost target or calibration domain exit.

Caller supplies one fresh frame per update, correct target identity and timestamps.
Aligned is a pixel result ONLY: it never means grasped, lifted or placed.
"""
    def __init__(self, calibration, speed=3., max_seconds=5., stable_frames=3):
        self.c = calibration
        if not np.isfinite(speed) or not 0 < speed <= 5 or not 0 < max_seconds <= 10 or stable_frames < 2:
            raise ValueError('Invalid alignment bounds')
        self.speed, self.max_seconds, self.required = speed, max_seconds, stable_frames
        self.started = None
        self.stable = 0
        self.last_stamp = -float('inf')
        self.last_command = None

    def update(self, state, target_uv, frame_stamp, now, dt=1/30):
        state = vector(state, 6)
        if np.any(state < LOW) or np.any(state > HIGH):
            raise ValueError('Measured pose outside command limits')
        if not np.isfinite([frame_stamp, now]).all() or not 0 <= now - frame_stamp <= .25 or frame_stamp <= self.last_stamp:
            raise ValueError('Missing, stale or repeated visual feedback')
        self.last_stamp = frame_stamp
        if self.started is None:
            self.started = now
        if now - self.started > self.max_seconds:
            raise ValueError('Alignment timed out')
        if self.last_command is None:
            self.last_command = state.copy()
        if np.max(np.abs(self.last_command[self.c.joint_indices]-state[self.c.joint_indices])) > 1.5:
            raise ValueError('Actuator tracking gap exceeds local correction limit')
        target = vector(target_uv, 2)
        if np.max(np.abs(state[:5] - self.c.reference_state[:5])) > self.c.pose_radius:
            raise ValueError('Outside measured pose region')
        error = target - self.c.reference_uv
        if np.linalg.norm(error) > self.c.pixel_radius:
            raise ValueError('Target outside calibrated capture region')
        if np.linalg.norm(error) <= self.c.tolerance:
            self.stable += 1
            return self.last_command.copy(), 'aligned' if self.stable >= self.required else 'settling'
        self.stable = 0
        # Damped inverse: target pixels move toward the measured grasp-center pixel.
        j = self.c.jacobian
        delta = -0.5 * j.T @ np.linalg.solve(j @ j.T + .01 * np.eye(2), error)
        # Accumulate from the prior command, not measured state. Otherwise each
        # sub-deadband increment resets and the physical joint never responds.
        desired = self.last_command.copy()
        desired[self.c.joint_indices] += delta
        command = bounded_target(desired, self.last_command, self.speed, dt)
        if np.max(np.abs(command[self.c.joint_indices]-state[self.c.joint_indices])) > 1.5:
            raise ValueError('Proposed correction exceeds actuator tracking gap')
        if np.max(np.abs(command[:5] - self.c.reference_state[:5])) > self.c.pose_radius:
            raise ValueError('Correction would leave measured pose region')
        self.last_command = command.copy()
        return command, 'aligning'


def detect_target(rgb, hsv_low, hsv_high, roi, min_area=30, max_area=6000):
    """Return a single HSV component in an explicit ROI; refuse ambiguous scenes.

Thresholds/ROI are station-specific. Red may need a wrapped hue interval.
"""
    import cv2
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('Expected uint8 RGB image')
    x, y, w, h = map(int, roi)
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > rgb.shape[1] or y+h > rgb.shape[0]:
        raise ValueError('ROI outside image')
    low, high = vector(hsv_low, 3), vector(hsv_high, 3)
    if np.any(low < 0) or np.any(high < 0) or np.any(low > [179,255,255]) or np.any(high > [179,255,255]) or np.any(low[1:] > high[1:]):
        raise ValueError('Invalid HSV thresholds')
    hsv = cv2.cvtColor(rgb[y:y+h, x:x+w], cv2.COLOR_RGB2HSV)
    if low[0] <= high[0]:
        mask = cv2.inRange(hsv, low, high)
    else:
        mask = cv2.inRange(hsv, np.array([0,low[1],low[2]]), high) | cv2.inRange(hsv, low, np.array([179,high[1],high[2]]))
    n, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    candidates = [i for i in range(1,n) if min_area <= stats[i,cv2.CC_STAT_AREA] <= max_area]
    if len(candidates) != 1:
        raise ValueError(f'Expected one target, found {len(candidates)}')
    i = candidates[0]
    left, top, bw, bh, area = stats[i]
    if left == 0 or top == 0 or left+bw >= w or top+bh >= h:
        raise ValueError('Target touches ROI boundary')
    return centers[i] + [x,y], (int(left+x),int(top+y),int(bw),int(bh)), int(area)

class GraspVerifier:
    """Gate transport on measured 3-D co-motion in a fixed, gravity-aligned frame.

Inputs must be independently observed block and gripper positions in metres,
with stable target identity. Never use commanded poses as measured positions.
Unknown depth, occlusion or stale tracking revokes verification.
"""
    def __init__(self, min_lift=.015, relative_tolerance=.008, stable_frames=5):
        if not 0 < min_lift <= .05 or not 0 < relative_tolerance < min_lift or stable_frames < 3:
            raise ValueError('Invalid grasp verification limits')
        self.min_lift, self.tolerance, self.required = min_lift, relative_tolerance, stable_frames
        self.reference = None
        self.count = 0
        self.last_stamp = -float('inf')

    def begin_lift(self, block_xyz, gripper_xyz, target_id):
        self.reference = (vector(block_xyz,3), vector(gripper_xyz,3), target_id)
        self.count = 0
        self.last_stamp = -float('inf')

    def update(self, block_xyz, gripper_xyz, target_id, stamp, now):
        try:
            if self.reference is None or target_id != self.reference[2]:
                raise ValueError('No grasp reference or target identity changed')
            if not np.isfinite([stamp,now]).all() or not 0 <= now-stamp <= .25 or stamp <= self.last_stamp:
                raise ValueError('Stale grasp evidence')
            self.last_stamp = stamp
            block, gripper = vector(block_xyz,3), vector(gripper_xyz,3)
            b0,g0,_ = self.reference
            lifted = block[2]-b0[2] >= self.min_lift and gripper[2]-g0[2] >= self.min_lift
            together = np.linalg.norm((block-gripper)-(b0-g0)) <= self.tolerance
            self.count = self.count+1 if lifted and together else 0
            return self.count >= self.required
        except ValueError:
            self.count = 0
            return False
