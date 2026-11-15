# SO-101 twin robot model (P3-TWIN-ROBOT, E1)

Status 2026-11-14T19:55Z: G1 passes for so101_twin.xml. The TCP position from MuJoCo differs from kin.fk by at most 1.4e-8 mm over 21693 recorded rows (limit 0.5 mm). The gripper law and a jaw-gap measurement are open. Neither is part of G1.

## Files

| Path | Content |
|---|---|
| so101_twin.xml | MJCF written by the URDF-to-MJCF converter (E1/tools/urdf_tools.py, make_twin.py) from the URDF that app/tools/kin.py loads. 8 bodies (world + 7 links), 6 hinge joints (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper), 1 site (gripper_frame_link), 17 visual mesh geoms (contype 0, conaffinity 0), no collision geoms, no actuators. |
| meshes/ | The 13 STL files used by the visual geoms (see Mesh provenance). |
| units.py | Studio normalized units <-> radians, and the three candidate gripper laws. |
| tcp.py | Loads an MJCF, sets the five body joints, reads the TCP site, returns position and orientation errors. |
| ../tests/test_kinematics.py | 7 tests: G1 on the 200-sample fixed set, kin unit checks, calibration against project_robots.json, URDF limits, gripper law identities, mesh files. |
| ../tests/data/g1_fixed_sample.json | 200 samples (joints_norm, kin_tcp_mm, kin_R) written by E1/tools/kin_reference.py from kin.fk. |
| ../tests/data/kin_calibration_reference.json | kin's own k values and norm-to-rad points, written by E1/tools/kin_reference.py from kin.py's FollowerCalibration. |

Outside this folder, in E1 (R/work/P3-TWIN-ROBOT/E1): tools/g1_check.py (G1 gate; writes g1_report.json and metrics.json), tools/gripper_check.py (writes gripper_check.json), kin_reference.npz (kin.fk over all rows), official_urdf_twin.xml (same conversion without meshes; baseline), mesh_identity.json, logs/.

## Sources and hashes (sha256)

| File | sha256 |
|---|---|
| repos/edge-developer-kit-reference-scripts/.../platforms/so101/urdf/so101.urdf (read by kin.py) | 3a65d2d35e68a8d2f0c2cc176d19b884506543c93ba72980145b80abe276022c |
| app/motion/project_robots.json | 7fdfdc2d4532bb05d87f8e608c32091fc5471e0000dde2d80205882c6df26667 |
| app/tools/kin.py | a2753d7d7c1d92d0813a7fe036ffe08e422b50fdee68102c7b5e03956ad58043 |
| E1/kin_reference.npz | a4f4bbf6c8c6bc21c9220e20bef5ce20680b9330a5649b9ab760e208ca95a721 |
| tests/data/g1_fixed_sample.json | a633de1545808a9cd5f784c8e58c759696df870358686578ce1986e68bc03ed8 |
| tests/data/kin_calibration_reference.json | 435969675e8739a42c4b50a90022549a9ae882215cccf9579ef394f8132ce67f |
| robot/so101_twin.xml | b6bbd07d76d23f08bb979476bc68988fee744d733bc1912337503e2aa6489e47 |
| E1/official_urdf_twin.xml | e4c4700d02dc5ebc4677474917d9af7cccd869b182004b660c880c1639fb0bdb |
| robot/units.py | 72479efcc774044846d0e705b7ca201bf9085c5f9150417b1d9ef525504fd48c |
| robot/tcp.py | 2828c7021b6e1102abf7cb1b0b6f3caef819250869fe26cf0b18c9c912c6e213 |
| tests/test_kinematics.py | 4c4959e9af471b119550eb96a3f309cfa37248e7b1467ed6e1bab507e410384c |
| E1/g1_report.json | 68d814aba682feb1701535806eadae6e48d51d989424b7a7c20100e6733719ad |
| E1/metrics.json | a69c6c4563904ba58f2c4b35856004982bd96ce11a1ec378c6052198e73b65ac |
| E1/gripper_check.json | 6809e19f8297889e02866dfa61706d5d6502659486d87cb0c3bf3b4ff92c4825 |

Hashes of the report files are as computed at 19:54:59Z, after the last script run.

## Units

- Body joints: q = K n, K = (range_max - range_min) / 200 * 2*pi / 4095, in rad per unit: shoulder_pan 0.017806194, shoulder_lift 0.017399590, elbow_flex 0.016908596, wrist_flex 0.018097722, wrist_roll 0.029344547. These equal kin's values to 1e-12 (test_k_matches_kin_reference). n = 0 is the middle of the calibrated range. This is kin's law: sign +1, zero offset 0.
- Calibration ticks (app/motion/project_robots.json, robot "SO101 Follower"): shoulder_pan 891-3212, shoulder_lift 822-3090, elbow_flex 898-3102, wrist_flex 812-3171, wrist_roll 74-3899, gripper 2046-3577.
- Gripper: k_g = 15.31 * 2*pi/4095 = 0.0234909810 rad per unit.

## TCP

Site gripper_frame_link is the frame of the URDF joint gripper_frame_joint (fixed, parent gripper_link, origin xyz -0.0079 -0.000218121 -0.0981274, rpy 0 3.14159 0). kin.fk returns the same frame: its position in the base frame in mm (T[:3,3] * 1000) and its rotation T[:3,:3].

## G1 results (all 21693 rows from 17 episodes; fixed set of 200)

| Candidate | Max position error (mm) | Median (mm) | p99 (mm) | Max orientation error (deg) | Rows at or above 0.5 mm | Fixed-200 max (mm) |
|---|---|---|---|---|---|---|
| twin_ours | 1.3652e-08 | 1.2949e-08 | 1.3576e-08 | 3.5715e-09 | 0 | 1.3599e-08 |
| official_urdf (baseline) | 1.3652e-08 | 1.2949e-08 | 1.3576e-08 | 3.5715e-09 | 0 | 1.3599e-08 |

twin_ours PASS against the 0.5 mm limit. Orientation error is reported; the spec gives no threshold for it. The two candidates give identical TCPs (0.0 mm and 0.0 deg difference).

## Why this model

twin_ours is the converter output for the URDF that kin.py loads, with the visual meshes, so it is the deliverable. official_urdf_twin.xml is the same conversion without meshes; it is the baseline in metrics.json. R/twin was deleted at 19:48Z (D-102), so its Menagerie model is not used.

## Gripper (not part of G1)

Logged values (gripper_check.json, 17 episodes, 21693 rows):
- Commanded closed, reading 0.52 to 0.98: median 0.52 (n 336). The earlier log summary said 0.59. This run does not reproduce that value, so the check against 0.59 fails.
- Commanded closed, reading 1.57 to 2.42: median 1.57 (n 3716). Inferred from the logs only, not checked against images.
- Open: median 24.62 (n 15833). Commanded 25 on 15819 rows.

Candidate laws in units.py (q in rad):

| Law | Normalized value at URDF upper limit | Empty-close stall above the closed stop |
|---|---|---|
| urdf_zero: q = k_g n | 74.30 | 10.7 deg |
| closed_at_0: q = -0.174533 + k_g n | 81.73 | 0.7 deg |
| closed_at_empty_reading: q = -0.174533 + k_g (n - 0.52) | 82.25 | 0.0 deg (on the stop by construction) |

On the logs alone, the two anchored laws put the empty-close stall next to the URDF closed stop, and urdf_zero does not. That is a plausibility argument, not a measurement. Choosing a law needs a jaw-gap or closed-stop measurement, which was not made.

## Mesh provenance

The 13 STL files were copied from R/twin/vendor/mujoco_menagerie/robotstudio_so101/assets before R/twin was deleted. Each file is byte-identical to its plugin Git LFS object (sha256 and size; E1/mesh_identity.json, all_identical true). G1 does not use the meshes. so101_twin.xml loads all 13 in MuJoCo 4.0.1. No other file was copied from R/twin.

## Gaps

- No Menagerie prefetch (mujoco_menagerie missing; the plugin install was --no-deps) and no plugin test run.
- Visual geoms only: no collision geometry, actuators, or dynamics.
- Jaw gap not measured.
- 5 of 22 episode folders have no joints.csv (any_1731/ep_0003, any_1731/ep_0004, any_1740/ep_0005, any_1740/ep_0008, any_1740/ep_0018). They are excluded.
- Orientation threshold not specified; reported only.

## Run

```
. /home/intel/intel-junction-2026/work/P3-TWIN-ROBOT/E1/tools/env.sh
nice -n 19 $SIM_PY -I -B $R/sim/twin/tests/test_kinematics.py   # 7 tests; exit 0 when all pass
nice -n 19 $SIM_PY -I -B $E1/tools/g1_check.py                   # G1 over all rows; writes g1_report.json, metrics.json
nice -n 19 $DATA_PY -I -B $E1/tools/gripper_check.py             # writes gripper_check.json
```


## v2 (E2, P3-TWIN-ROBOT): graspable twin, PARTIAL

Files: so101_twin_v2.xml (new; v1 unchanged), generator work/P3-TWIN-ROBOT/E2/tools/make_v2.py, check work/P3-TWIN-ROBOT/E2/tools/check_v2.py, tests/test_grasp.py. Results and STATUS: work/P3-TWIN-ROBOT/E2/.

What changed from v1:
- 17 collision geoms (group 3, convex hulls of the v1 meshes). Arm links contype 2 conaffinity 1; gripper_link and moving jaw contype 1 conaffinity 1. Visual geoms unchanged.
- Excludes: world-base_link, wrist_link-gripper_link, gripper_link-moving jaw.
- 6 position actuators with kp, kv and forcerange (+/-10) from sim/twin/dynamics/twin_dynamics_v1.json; armature and frictionloss from the same file; damping 0.
- ctrl is in radians. Arm ctrl = k_j * goal (the urdf_zero law the DYN fit used). Gripper ctrl = -0.174533 + k_g*(goal - 0.52) (units.py closed_at_empty_reading). The DYN gripper gains were fit under the other law, so the gripper mixes laws.

Verified: v2 compiles; G1 on v2 passes (max 1.36e-8 mm, 200 samples); closing on nothing reads inside 0.50-0.60.

Not working yet: the brick grasp (0/10 holds; TCP settles 4-18 mm off the sort_any heights) and the jaw-gap function (returns the same value at every reading, so no implied openings yet).

Open: gripper law (URDF span 0.0191986 rad/unit vs tick scale 0.0234910 rad/unit; no single linear law fits 0.52, 1.57-2.29 and 24.6 for a 31.7 mm brick); approach axis assumed to be TCP site z; mat extent is a placeholder; brick mass is a solid upper bound and friction is assumed.

## v3 (E3, P3-TWIN-ROBOT): jaw pads, NOT yet passing the grasp

File: so101_twin_v3.xml (v1 and v2 unchanged). Built by work/P3-TWIN-ROBOT/E3/tools/build_v3.py from v2.
- Jaw hulls (gripper_link and moving jaw, group 3) replaced by two box pads (pad_fixed, pad_moving) fitted to jaw vertices within 15 mm of the TCP, in each body frame. Pads contype 1 conaffinity 1, condim 4, friction 0.35.
- Pad gap (mj_geomDistance) is monotone over the gripper range: 1.75 mm at reading 0.52, 3.7 mm at 1.57, 5.0 mm at 2.29, 36.7 mm at 24.6. The v2 hull overlap is gone.
- act_gripper kp 5.0, kv scaled from the DYN ratio, forcerange +/-10 (stall behaviour not yet matched to the logs).
- G1 on v3: max 1.36e-8 mm over 200 samples (PASS).

Not working: the brick is pushed away during the 26 mm descent (pad_fixed touches the brick top; the TCP sits over the fixed jaw, not between the jaws). test_grasp.py centres the jaw midpoint over the brick, but the brick still moves. Held count 0/10.

Gripper law evidence from real logs (E3 outputs/gripper_logs.json): the servo stops on the brick with goal 0 at readings 1.50 to 2.42, and empty stops at 0.52 and 0.65. Held minus empty is about 1 unit (0.024 rad), which is not consistent with a 31.7 mm brick at the 82.8 mm tip radius. Unresolved.
