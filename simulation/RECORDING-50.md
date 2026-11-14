# Recording 50 brick inspection demonstrations

**Superseded for new recordings by [SORTING.md](SORTING.md) and
sorting-recording-plan.csv.** Keep the camera, operator and video-review guidance
below; use the new per-episode color/destination instructions and distribution.

Use the existing Studio/dashboard session and teleoperation controls. Do not
start another session on the same arm. Recording is an operator activity; none
of the scripts here connects to or moves a robot.

## Before the first episode

- Confirm both recording views: overhead sees the entire pickup/target area;
  arm camera sees the gripper and object during grasping. Camera roles in the
  handoff were inferred; verify them visually.
- Keep mounts, exposure, table and target fixed for this first dataset.
- Use the existing 640×480, 30 FPS recording configuration for both RealSense
  cameras. The old 424×240/6 FPS setting was for preview only. Dashboard preview
  rate is not recording rate.
- Verify local camera/session stability and playback of one completed episode
  before recording the batch. Recording metadata alone cannot prove video quality.
- Confirm the leader is available and calibrated before teleoperation. Do not
  recalibrate just to satisfy the simulator; never guess motor offsets.
- Use the same task text: `move the brick to the inspection spot`.

## Collection plan

Mark five reachable pickup regions on the table, clear of the base and target.
Choose their positions physically; no guessed Cartesian coordinates are sent to
the robot. For each region collect 10 successful episodes: five with a 2x2 brick and five
with a 2x4 brick (25 of each overall), with small changes
in position and modest brick rotation within the region. Keep the target fixed.

Each episode: start from a consistent clear pose; approach; grasp; lift; carry;
place entirely within the inspection spot; release; retreat; pause briefly so
the final placement is visible. Reset the scene outside recording.

`recording-plan.csv` reserves eight train and two validation slots in each region:
40 train and 10 validation overall, with 20 train and 5 validation per brick size. Put held-out episodes at distinct precise
positions/rotations; don't just repeat the same trajectory. This is a small
in-distribution validation set. Final robustness testing also needs new physical
placements and conditions not used in either split.

The CSV leaves `episode_index` and `accepted` blank. After video review, enter
the actual ID and `yes` only for successful, uninterrupted episodes. Failed
attempts can be kept for later recovery analysis, but should not be silently
labeled successful for behavior-cloning training. Record replacements for
rejected slots. Existing seven episodes may count only if they pass this review
and fit the camera/task setup; otherwise collect 50 fresh episodes.

For each accepted episode check both videos, grasp/lift/release, final target
placement, no missing/stale frames, and no operator assistance during the skill.
Finalize/flush the dataset through Studio before copying or auditing it.

Nominal full-height LEGO dimensions are configured in bricks.json. Confirm the
pieces are full-height bricks, not plates or stacks. Record target dimensions, arm-base pose
relative to the table, and camera placement. These are needed for simulation
matching; 50 demonstrations alone don't establish the real-to-sim joint map.

This single-task dataset trains the manipulation skill. It does not establish
language generalization; testing different instructions requires distinct tasks
and corresponding demonstrations later.
