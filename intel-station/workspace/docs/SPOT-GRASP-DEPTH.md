# Spot-cell grasp with depth (I-59)

Verified live 2026-11-14, 16:48-16:54Z. The taped blue 2x2 DUPLO at the spot was grasped (gripper 2.42 closed on the brick, 1.44 while carrying), carried to the GREEN plate, released, and the arm returned home. Step logs: `logs/depth/servo/steps.tsv`, frames and depth per step in `logs/depth/servo/<HHMMSS>_<step>/`. Poses: `app/motion/gm/spot_grasp_v1.json`.

## Sensors (measured)

| Camera | Colour | Depth | Left IR | Link | Depth mode | Valid depth | Camera to mat | Tilt | Plane RMS |
|---|---|---|---|---|---|---|---|---|---|
| top D415 | /dev/video8 | /dev/video4 | /dev/video6 | USB2 (3-2.3) | 848x480@15 | 83% | 590 mm | 20° | 1.4 mm |
| close D415 sn 344223022914 | /dev/video14 | /dev/video10 | /dev/video12 | USB3 (2-4.4.4) | 424x240@15 | 61% | 147-149 mm | 50° | 0.9 mm |

- Depth unit is 1 mm (D400 default). Depth is read through plain V4L2 gray16le with ffmpeg (`app/tools/rs_grab.py`). librealsense is not used, so the monitor's colour streams keep running.
- Close camera at 848x480 gives only 26% valid depth because the scene is inside the D415 minimum range. Use 424x240.
- Depth check against a known object: the DUPLO 2x2 top face measured 19.0-19.5 mm (true 19.2 mm) in every grab.
- Intrinsics are nominal (HFOV 65°). The brick-height check bounds the resulting height error below 1 mm at the spot.

## Arm precision (root cause of "too high" readings)

- `run_sequence.py` connects with goal = present and disconnects with goal = present (physicalai_so101_plugin so101.py `disconnect` → `_hold_position`). Under gravity, shoulder_lift and elbow sag by the servo steady-state error at each connect and at each disconnect. Measured sag: 4.2 units (shoulder_lift) and 2.8 units (elbow). Effect: jaw tips 11-25 mm lower than commanded, and up to 12 mm off sideways.
- Fix: `app/tools/motion_server.py` holds one connection. It ramps from the last goal, removes the steady-state error with a clamped goal offset (±8 units) and closes the port without re-holding. Residual error measured ≤0.5 units on every joint.
  - Clients: `mcmd.py` sends one command, `mseq.py` runs a named sequence, `servo_step.py` moves then measures depth.
- Offsets the server needed at the spot: shoulder_lift -4.2 to -4.7 units, elbow -0.1 to -3.4 units.

## Jacobian at the spot hover

Measured at [-12.62, 30, -19.33, 69.73, -45.65] with `app/tools/jprobe.py`. Units are mm per joint unit. Mat frame: x = image right, y = toward the robot base, z = up.

| Joint | dx | dy | dz | Note |
|---|---|---|---|---|
| shoulder_pan | -4.3 | -2.3 | +0.7 | sideways, 4.9 mm/unit |
| shoulder_lift | -0.67 | +1.32 | -3.27 | |
| elbow_flex | -0.93 | +2.15 | -2.72 | |
| wrist_flex | -0.95 | +2.20 | -0.12 | toward the base, 2.4 mm/unit |
| wrist_roll | | | | rotates the jaws; wrist-cam brick yaw +1.37°/unit |

- Vertical descent at constant gripper pitch: change wrist_flex by -0.975 × the elbow change. Measured 1.70-1.77 mm of descent per elbow unit, which is about 83% of the linear prediction.

## Verified grasp procedure (spot cell)

1. Hover at [-12.62, 30.0, -19.33, 69.73, roll, 25]. Jaw tips are about 57 mm above the mat.
2. Align yaw: set roll += -(wrist-cam brick edge yaw)/1.37. For this brick that was -11.6, giving roll -57.25.
3. Centre toward the base: wrist_flex +4.7. The brick sat 11.3 mm toward the base from the line through the jaw tips.
4. Descend at constant pitch in three stages, keeping shoulder_lift at 30 and pan at -12.62. Values are (elbow, wrist_flex):
   - (-7.83, 63.23): tips at 32 mm
   - (-1.33, 56.93): tips at 19 mm
   - (5.47, 50.33): tips at 5-7 mm
5. Close the gripper to target 0.0. The reading was 2.42: brick held.
6. Lift to (elbow, wrist_flex) = (-3.03, 58.63) with the gripper target kept at 0.
7. Go to raised_closed [-12.62, -40, 80, 66, -45, 0], then home_closed [-6, -98, 96.6, 66, -45, 0], then the spot_fast sequences home_to_green_carry → release_green → green_to_home.

## Checks to automate (LOOP3 / autonomy)

- Brick present at the spot: close-depth top face at 19 ± 2 mm with at least 600 px in the spot box.
- Before the tips go below 24 mm:
  - The jaws are outside the brick footprint in depth.
  - The brick is centred between the jaws in the wrist camera.
  - The yaw is within ±3°.
- Grasp height: the lowest left-jaw depth in box x 150..192, y 30..175 (`rs_box.py`) is 4-8 mm.
- After closing: a gripper reading of 1.2-4.5 means the brick is held.
- After lifting: no 19 mm blob remains at the spot, and the wrist camera still shows the brick between the jaws.

## Safety events

- 16:50:22Z: a real hand reached over the blue plate during the lift. The top-view stop was correct; the arm held, then resumed after the area was clear.
- 16:52:52Z: false stop from the wrist view. A white brick looks skin-toned in the wrist camera (8247 px). Fix: the wrist view now needs at least 20000 px on 2 consecutive frames. The top and close views are unchanged.
- 16:20Z: three stale watchdog copies with old thresholds stopped the arm. Restart the watchdog only by its pid file, never by a name pattern (L-043).
