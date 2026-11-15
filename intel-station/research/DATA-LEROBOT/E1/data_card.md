# Data card: ours-lerobot v1 (DATA-LEROBOT E1)

Dataset: `/home/intel/intel-junction-2026/data/ours-lerobot/v1` (LeRobot v3.0, Studio layout, built 2026-11-14T22:02:52Z by build_lerobot.py e1-2 2026-11-14).

## What is in it
- 3 episodes, 7750 frames at 30 fps (258.3 s), 3 tasks, three AV1 cameras per episode (top, close, wrist).
- Included only when a visual check confirms the brick was placed on the plate the loop chose and the OD-10 decision is correct (VISUAL table in the script, from P3-LABEL-OURS V1). Loop-only successes are not included (2 of them so far).
- Task text: "move the orange brick to the blue plate"; "move the red brick to the blue plate"; "move the yellow brick to the green plate".
- Colour and plate: orange on blue (OD-10 GOOD), red on blue (OD-10 GOOD), yellow on green (OD-10 DEFECT).
- Camera files used: close_20261114T170351Z.mkv, top_20261114T170351Z.mkv, w_20261114T174051Z.mkv, w_20261114T174151Z.mkv, w_20261114T174251Z.mkv, w_20261114T181351Z.mkv, w_20261114T181451Z.mkv, w_20261114T181951Z.mkv, w_20261114T182051Z.mkv.

## Per-episode labels (meta/episodes columns ours_*)
| episode | ep no | task | tier | needs_review | decision confidence | frames | note |
|---|---|---|---|---|---|---|---|
| any_1740/ep_0001 | 0 | move the orange brick to the blue plate | V | no | high | 3612 | orange 2x2 without tape (pre_close, stage1_wrist, closed_wrist); released_wrist: in the blue plate; loop colour yellow is wrong (scan hue 14); loop colour yellow, visual orange |
| any_1740/ep_0015 | 1 | move the red brick to the blue plate | V | yes | med | 2024 | released_close: red 2x2 in the blue plate; tape and orientation not resolvable at 640x480 |
| any_1740/ep_0017 | 2 | move the yellow brick to the green plate | V | yes | med | 2114 | released_top, released_close: cream/yellow 2x2 on the green plate; dark tape not confirmable at 640x480, so the green decision holds only if the tape is present |

Columns: ours_ep_no, ours_run, ours_episode, ours_colour, ours_plate, ours_od10_label, ours_tier (V visual, L loop-only), ours_verified_by, ours_decision_correct, ours_decision_confidence (high, med, low), ours_needs_review, ours_loop_task, ours_note.
- needs_review is true when the visual confidence is below high or when a tape or orientation item could not be confirmed at 640x480.

## Clock and alignment
- top (overhead, serial 243622061421): lag applied +21.5 ms (P3-VLA-CLOCK median; reliable in 11 of 15 episodes). The camera frame for grid time t is the frame nearest t + lag.
- close (close view, serial 243722064316): lag applied 0 ms applied: not determined (P3-VLA-CLOCK reliable in 3 of 15 episodes). The camera frame for grid time t is the frame nearest t + lag.
- wrist (wrist camera (MJPEG segments, 60 s)): lag applied +7.5 ms (P3-VLA-CLOCK median; reliable in 15 of 15 episodes). The camera frame for grid time t is the frame nearest t + lag.
- Joints: linear interpolation of pos (observation.state) and goal (action) onto a 30 fps grid from the rows between t_start and t_end of episode.json.
- Largest grid-row to camera-frame distance per camera (camera clock): {"top": 0.0334, "close": 0.0278, "wrist": 0.018}.

## Excluded candidates
- 38 of 41 candidates excluded; by category: joint or label data 1, loop not a success 14, loop success not visually verified 1, not finished 3, visual check 18, visual placed, data check failed 1. Full list with reasons: excluded.tsv.
- Visually placed but excluded for a data reason: any_1731/ep_0003.
- Visual entries without an episode folder: none.

## Known issues
1. Close camera clock is not determined; 0 ms is applied (flagged in every row's camera metadata).
2. Wrist offset conflict: P2-D-VLA1 open issue R4 (wrist alignment unresolved) reports wrist offsets of -103 and -161 ms; P3-VLA-CLOCK measured +7.5 ms (median, 15 of 15 episodes reliable). This build uses +7.5 ms. The conflict is open and needs a decision before training.
3. Top offset +21.5 ms is the P3-VLA-CLOCK median (11 of 15 reliable; any_1731 ep_0001 and ep_0002 and any_1740 ep_0001 and ep_0004 are not reliable there).
4. 2 of 3 included episodes have needs_review set (see the table above).
5. Loop colour names disagree with the visual check for: any_1740/ep_0001. The task text uses the visual colour name.
6. Monitor files recorded at 15 fps (relative clock, before 17:03:51Z) and wrist segments on a relative clock are not used: the coverage check excludes any episode that touches them.
7. The live monitor files (from 20:53:47Z) are still recording; the coverage check excludes an episode whose window the file does not reach.
8. Not imported into Physical AI Studio (not requested). The LeRobot loader check is in validation.json.

## Re-run
`nice -n 19 env OMP_NUM_THREADS=4 <R>/envs/lerobot-env/bin/python -I <R>/work/DATA-LEROBOT/E1/build_lerobot.py`. Episodes whose episode.json, joints.csv and VISUAL entry are unchanged reuse their videos; new visual verdicts are encoded and appended with the next stable episode number. Dataset tables are rebuilt each run.

## Visual check (5 frames, E1 worker)

The five frames in visual/ (sample_01.png to sample_05.png; choices in visual/samples.json) were read at panel size, about 320 x 290 px per camera. All five come from the three included episodes.
- sample_01, any_1740/ep_0001, 77.7 s: the top view shows the orange 2x2 in the blue plate; the wrist view shows an orange 2x2 still between the red fingers beside the blue plate.
- sample_02, any_1740/ep_0001, 34.8 s: the close and wrist views show the orange 2x2 at the gripper fingers; the top view also shows an orange brick already in the blue plate.
- sample_03, any_1740/ep_0001, 67.6 s: the wrist view shows an orange 2x2 in the gripper and a second brick in the blue plate.
- sample_04, any_1740/ep_0015, 32.2 s: the wrist view shows a red 2x2 in the gripper with the green plate in view; the top view shows white and yellow bricks already in the blue plate.
- sample_05, any_1740/ep_0017, 51.8 s: the wrist view shows a cream 2x2 with a dark strip on its top face in the gripper; the top view shows several bricks in the blue plate.

Reading: the panels are too small to confirm brick identity, tape or orientation. The blue plate already holds bricks before the placement frames in ep_0001 and ep_0015, so the top view alone does not show which brick was placed; check their release frames at full resolution before training. The dark strip on the ep_0017 brick fits the tape that the DEFECT label needs, but one frame does not confirm it.
