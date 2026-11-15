"""Constants for the pick stage. Metres and radians unless a name says otherwise.

Joint commands are in the driver's 'normalized' unit: body joints in [-100, 100], gripper in [0, 100]
(candidates/code/motion/so101_motion.py BODY_LIMIT, GRIPPER_LOW/GRIPPER_HIGH).
Values marked [unclear] are not in the repo or the docs and must be measured on the laptop (runbook.md).
"""

from __future__ import annotations

# Table plane in the URDF base_link frame. [unclear: base_link origin is assumed to sit on the table;
# measure by slow touch-down, see runbook.md step 3.]
TABLE_Z_BASE_M = 0.0

# DUPLO brick. brickfact.com/blog/guidebooks/lego-brick-dimensions: 2x4 about 64 x 32 x 19.2 mm (stated).
# 2x2 footprint about 32 x 32 mm and the same height are derived there (stud pitch 16 mm, derived).
BRICK_HEIGHT_M = 0.0192
BRICK_SIDE_2X2_M = 0.032
BRICK_LONG_2X4_M = 0.064

# TCP heights relative to the table.
# Hover TCP is 3 cm above the brick top. The task asked for 8 cm: top-down IK at 8 cm needs wrist_flex beyond
# its URDF limit (1.658 rad, normalized 91.6) at x = 0.18 m, measured in sim/run_sim.py (research.md section 3).
HOVER_CLEARANCE_M = 0.03
GRASP_TCP_ABOVE_TABLE_M = 0.010  # TCP at about mid-height of the brick (BRICK_HEIGHT_M / 2 = 9.6 mm)
# Lift TCP height. Top-down IK above about 5 cm is infeasible at the mat (wrist_flex limit, research.md section 3),
# so the transport from lift to a plate is not planned here: the SORT executor owns it (runbook.md section 6).
LIFT_CLEARANCE_M = 0.05

# Gripper commands in normalized units. [unclear: open/closed direction and open value vary by operator,
# FACTS-LIVE.md 13:08Z: closed on brick ~2.2-2.8, open 10-36.]
GRIPPER_OPEN_N = 30.0
GRIPPER_CLOSE_N = 0.0

# Reachability and collision thresholds.
IK_POS_TOL_M = 0.002
IK_ROT_TOL_RAD = 0.05  # about 3 degrees
JOINT_MARGIN_UNITS = 5.0  # keep this many normalized units away from both ends of the calibrated range
BASE_KEEPOUT_M = 0.09  # TCP must stay farther than this from the base vertical axis
PLATE_RADIUS_M = 0.045  # [unclear: plate footprint radius, measure on the table]
PLATE_MARGIN_M = 0.01  # jaw and wrist margin added to the plate radius for the path check
PLATE_CLEAR_Z_M = 0.12  # above this TCP height the plates are not an obstacle

# Home pose from OD-8a (DECISIONS.md, FACTS-LIVE 13:08Z): shoulder_lift ~-98.5, elbow ~96, wrist_flex ~62.
# pan and roll at home are not recorded [unclear], taken as 0.
HOME_NORM = (0.0, -98.5, 96.0, 62.0, 0.0)

# Speed caps from candidates/code/motion/so101_motion.py (tuning values, confirm on hardware).
DEFAULT_MAX_SPEED_UNITS_S = 25.0
HARD_MAX_SPEED_UNITS_S = 50.0
