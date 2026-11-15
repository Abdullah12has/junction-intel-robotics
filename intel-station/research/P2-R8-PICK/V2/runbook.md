# Runbook: tip-visible calibration routine (P2-R8-PICK, verifier V2)

Status: PARTIAL. The code and the fake-bus dry run are verified. The live routine has NOT been run on the robot. Only Opus may run it, under the conditions in section 3.

## 1. What the routine does

- Plans a pose grid at one height above the mat: z = 0.060 m above TABLE_Z_BASE_M, 40 poses (range 20 to 99), away from the base keep-out (0.09 m) and from plate discs (radius plus the 0.032 m budget).
- Moves to each pose through R/app/motion (MotionPlayer, speed 15 units/s, tolerance 6 units, 0.6 s settle). At each pose it finds the red jaw tip in the top view and records the read-back joints. A pose that does not arrive is dropped.
- Fits a planar homography (table to image) and five per-joint zero offsets by Levenberg-Marquardt. Gate: leave-one-pose-out RMS of 10 mm or less.

## 2. Scope and limits (read before any run)

1. One height only. The gate does NOT validate the per-joint offsets. In the fake-bus dry run the gate passes with wrong offsets (tip_only): shoulder_lift -19.03 against truth -1.2, wrist_flex 8.21 against 1.5, wrist_roll 36.27 against -0.8. Source: /home/intel/intel-junction-2026/work/P2-R8-PICK/V2/calib_dry_final/tip_calib_dry_run_20261114T155513Z.json (final code, 40 poses; the earlier sandbox run in V2/dry_run_final gives the same offsets to two decimals). The identifiability flag does not protect against this: in the same run lift, elbow, wrist_flex and roll are reported identifiable (their Jacobian columns are not exactly inside the homography span), yet their fitted values are wrong. Do not read the flag as validation.
2. shoulder_pan is not identifiable without anchors. A constant pan offset is a rigid rotation that a free homography absorbs. With plate anchors in the dry run, pan is 1.55 against truth 1.8, but lift (4.68), elbow (0.31), wrist_flex (-0.10) and wrist_roll (-1.26) remain wrong, and the gate still passes at 0.88 mm.
3. Do not drive grasp-height IK with the fitted offsets until a second height (the W2 plan uses 0.08 m and 0.02 m) or an independent check exists. Two heights need a 3x4 camera model; the current homography is planar.
4. Priors: the sim2 plate centres in geometry_sim2_V1.json (MJCF base frame) differ from the real release means in calib.json (release_plate_check_convention_A) by 20.1 mm for blue and 19.3 mm for green, mostly tangential. The priors are used only as keep-out discs and as a prior check, never as anchors. Unresolved.
5. The laptop motion environment (envs/motion-env) has no pyrealsense2 and no cv2. The live command stops at camera start, before any port is opened, until a camera-capable environment is provided. One test is skipped because cv2 is absent.
6. Not included: pick validation (offline_eval.json is NOT RUN), grasp-height measurement, the DET2 adapter, single-frame cluster locate, the W1 elongated-brick yaw rule, the I-52 hook, and any automatic hands-in-scene check. The hands check is manual (I-43 rule 3).

## 3. Preconditions (I-43, LAPTOP.md rule 8; Opus only)

1. Hassan's authorisation for today (I-43). Only Opus, or an agent whose task names this operation, may run the live command.
2. The follower port is /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D040939-if00. The routine resolves it through so101_motion.resolve_arm(follower=True) and checks the serial. Never use a ttyACM number. The port must be held by no process, and no Studio runtime session may claim the follower. Check /home/claude/kb/ledger/PAUSE.txt before camera or serial access.
3. Immediately before the run, a fresh camera check (wrist, and RealSense if free) shows no hands or people in the workspace. The routine asks for a typed YES; it does not look at the image.
4. Motion goes only through R/app/motion: speed 15 to 20 units/s, tolerance 6 units, torque off at the end.
5. Do not touch the leader arm, calibration files or Studio settings. The routine reads project_robots.json and writes only into the --out folder.

## 4. Exact commands

Run from the app folder that contains the pick_v2 package. Setup:

    cd /home/intel/intel-junction-2026/app
    PY=/home/intel/intel-junction-2026/envs/motion-env/bin/python
    URDF=/home/intel/intel-junction-2026/repos/edge-developer-kit-reference-scripts/samples/ai/edge-ai-demo-studio/workers/services/robotics-ai/platforms/so101/urdf/so101.urdf
    ROB=/home/intel/intel-junction-2026/app/motion/project_robots.json
    W=/home/intel/intel-junction-2026/work/P2-R8-PICK/V2

4.1 Tests (no hardware). Expected on the laptop: 13 passed, 1 skipped (cv2 not installed).

    SO101_URDF=$URDF SO101_ROBOTS=$ROB SO101_MOTION_DIR=/home/intel/intel-junction-2026/app/motion PYTHONDONTWRITEBYTECODE=1 $PY -m pytest pick_v2/tests -v -rs -p no:cacheprovider --basetemp=$W/pytest_tmp

4.2 Fake-bus dry run (no port, no camera). Expected: tip_only PASS and plate_anchors PASS. This checks the code path and the gate arithmetic, not the offsets.

    PYTHONDONTWRITEBYTECODE=1 $PY -m pick_v2.tip_calib dry-run --urdf $URDF --robots $ROB --motion-dir /home/intel/intel-junction-2026/app/motion --out $W/calib_dry --n-poses 40

4.3 Refit saved samples (no hardware). Use the samples file written by a dry run or a live run.

    PYTHONDONTWRITEBYTECODE=1 $PY -m pick_v2.tip_calib fit --urdf $URDF --robots $ROB --samples <path to tip_calib_*.json> --out $W/calib_refit [--anchors <anchors.json>]

4.4 Live (Opus only; preconditions in section 3; interactive terminal required).

    PYTHONDONTWRITEBYTECODE=1 $PY -m pick_v2.tip_calib live --urdf $URDF --robots $ROB --motion-dir /home/intel/intel-junction-2026/app/motion --out $W/calib_live --geometry $W/priors/geometry_sim2_V1.json --confirm-motion --speed 15 --n-poses 40 [--anchors <anchors.json>]

Sequence of 4.4: resolve the by-id port and check its serial; start the RealSense (needs pyrealsense2); prompt for a typed YES; open_follower_held; move through the poses; move home; prompt "Support the arm by hand, then press Enter"; torque off; fit and gate (about 1 to 3 minutes of computation, no motion); write one JSON; close the port.

After the Enter prompt the arm is unpowered. Keep it supported, or rest it on the table, until the result is printed. Ctrl-C at any point leaves torque on and the arm holding; support it by hand before releasing. Watch for: a hand or a person in the workspace, a pose that does not arrive, and any drop after torque is off.

4.5 Anchors file (optional). A JSON list of measured points only, for example [{"xy_m": [0.10, 0.14], "uv_px": [400.0, 300.0]}].

## 5. Output

One JSON file per run in --out (name tip_calib_<mode>_<UTC>.json; existing files are never overwritten). It holds the config, the plates, the poses, the samples (read-back joints, tip pixel, status), and per anchor set: status, homography, offsets in normalized units, zero shift in radians, in-sample pixel RMS, leave-one-out RMS, median and max, per-sample leave-one-out errors, identifiability per offset, the scope warning, and the prior check when priors are given.

## 6. Faults found and fixed in this revision

- live used the export's follower port, which is the stale /dev/ttyACM1 in project_robots.json. It now calls resolve_arm(follower=True), which returns the by-id path (so101_motion.py, FOLLOWER_PORT_BY_ID).
- After torque was switched off, disconnect() would re-enable torque while torque_on_disconnect is True (physicalai_so101_plugin so101.py:457-470), so the arm would be powered again after release. The routine now clears the flag after torque is off.
- live now requires a typed YES before the port is opened.
- The error message about torque is printed only when torque is still on.
- Found in W3, not fixed here: /home/claude/kb/work/P2-R8-PICK/W3/runbook.md line 61 passes square=True unconditionally. The W3 sim call site (W3/sim/run_sim.py line 199) passes square=det.square.

## 7. Still missing before any pick run

- A pixel-to-table homography at grasp height. Not established: the brick labels are unverified and the data-only fit gave a leave-one-out RMS of 46.15 mm (calib.json).
- An offline pick validation. offline_eval.json is NOT RUN.
- A measured grasp height in the pick chain (calib.json: grasp TCP z mean -0.0144 m, std 4.2 mm, n=12, convention A).
- The merge items in consolidated.md section 5, including the DET2 adapter, single-frame cluster locate, the W1 elongated-brick rule, and a live pick wrapper with a dry-run default.
