"""SO-101 twin unit map (P3-TWIN-ROBOT, E1): Studio normalized joint values <-> MJCF joint radians.

Each constant is checked against its source by R/sim/twin/tests/test_kinematics.py:
  CALIBRATION  <- app/motion/project_robots.json, robot "SO101 Follower" (type SO101_Follower), payload.calibration
  URDF_LIMITS  <- repos/.../platforms/so101/urdf/so101.urdf, joint limit lower/upper
  K            <- app/tools/kin.py through app/pick_v2/kinematics.py (FollowerCalibration.rad_from_norm);
                  reference values in tests/data/kin_calibration_reference.json

Body joints (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll), normalized n in [-100, 100]:
    q = K[j] * n,    K[j] = (range_max - range_min) / 200 * 2*pi / 4095
This is the law kin.fk uses (sign +1, zero offset 0). n = 0 is the middle of the calibrated range.

Gripper, normalized n in [0, 100]. kin.fk does not use it, so it is outside G1. Three candidate laws; the choice is
open (README.md, gripper_check.json):
    urdf_zero                 q = k_g * n
    closed_at_0               q = -0.174533 + k_g * n            (norm 0 at the closed limit)
    closed_at_empty_reading   q = -0.174533 + k_g * (n - 0.52)   (logged empty-close reading at the closed limit)
    k_g = (range_max - range_min) / 100 * 2*pi / 4095 = (3577 - 2046) / 100 * 2*pi / 4095
The closed_at_0 form is the anchoring in the R/twin jointmap draft, read before R/twin was deleted (19:48Z). Nothing was
copied from R/twin for this law.

Pure Python (math only), so this file runs unchanged in sim-env and data-env.
"""
import math

TICKS_PER_REVOLUTION = 4095.0
BODY_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER_JOINT = "gripper"
JOINTS = BODY_JOINTS + (GRIPPER_JOINT,)
NORM_SPAN = {j: 200.0 for j in BODY_JOINTS}
NORM_SPAN[GRIPPER_JOINT] = 100.0

# (range_min, range_max) in ticks, app/motion/project_robots.json, robot "SO101 Follower"
CALIBRATION = {
    "shoulder_pan": (891, 3212),
    "shoulder_lift": (822, 3090),
    "elbow_flex": (898, 3102),
    "wrist_flex": (812, 3171),
    "wrist_roll": (74, 3899),
    "gripper": (2046, 3577),
}

# joint limits (rad) from the URDF <limit lower upper>
URDF_LIMITS = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
    "gripper": (-0.174533, 1.74533),
}

GRIPPER_CLOSED_STOP_RAD = -0.174533     # URDF lower limit of joint 'gripper'
GRIPPER_EMPTY_CLOSE_NORM = 0.52         # median of logged pos5 with goal5 <= 1 and pos5 <= 1 (336 rows, gripper_check.json); the earlier log summary said 0.59, not reproduced
GRIPPER_LAWS = {                        # law -> (offset_rad, n0): q = offset + k_g * (n - n0)
    "urdf_zero": (0.0, 0.0),
    "closed_at_0": (GRIPPER_CLOSED_STOP_RAD, 0.0),
    "closed_at_empty_reading": (GRIPPER_CLOSED_STOP_RAD, GRIPPER_EMPTY_CLOSE_NORM),
}


def rad_per_unit(joint):
    lo, hi = CALIBRATION[joint]
    return (hi - lo) / NORM_SPAN[joint] * 2.0 * math.pi / TICKS_PER_REVOLUTION


K = {j: rad_per_unit(j) for j in JOINTS}


def norm_to_rad(values):
    """Body joints in normalized units (first five values, BODY_JOINTS order) -> radians."""
    return [K[j] * float(v) for j, v in zip(BODY_JOINTS, values[:5])]


def rad_to_norm(values):
    """Body joint radians (BODY_JOINTS order) -> normalized units; inverse of norm_to_rad."""
    return [float(q) / K[j] for j, q in zip(BODY_JOINTS, values[:5])]


def _law_params(law):
    if law not in GRIPPER_LAWS:
        raise ValueError(f"unknown gripper law {law!r}; expected one of {tuple(GRIPPER_LAWS)}")
    return GRIPPER_LAWS[law]


def gripper_norm_to_rad(n, law):
    offset, n0 = _law_params(law)
    return offset + K[GRIPPER_JOINT] * (float(n) - n0)


def gripper_rad_to_norm(q, law):
    offset, n0 = _law_params(law)
    return n0 + (float(q) - offset) / K[GRIPPER_JOINT]


def gripper_tick_from_norm(n):
    """Raw tick for a gripper normalized value: 0..100 maps linearly onto range_min..range_max."""
    lo, hi = CALIBRATION[GRIPPER_JOINT]
    return lo + float(n) / 100.0 * (hi - lo)


def closed_at_0_rad_from_tick(tick):
    """Tick form of the closed_at_0 law: zero tick at the range midpoint, offset chosen so norm 0 sits at the closed
    stop. Equal to gripper_norm_to_rad(n, "closed_at_0") when tick = gripper_tick_from_norm(n)."""
    lo, hi = CALIBRATION[GRIPPER_JOINT]
    mid = (lo + hi) / 2.0
    rad_per_tick = 2.0 * math.pi / TICKS_PER_REVOLUTION
    offset = GRIPPER_CLOSED_STOP_RAD - (lo - mid) * rad_per_tick
    return (float(tick) - mid) * rad_per_tick + offset


def within_urdf_limits(joint, q):
    lo, hi = URDF_LIMITS[joint]
    return lo <= float(q) <= hi
