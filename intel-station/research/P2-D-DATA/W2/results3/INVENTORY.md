# INVENTORY (P2-D-DATA W2), generated 2026-11-14T13:17:10Z, input signature 7e8e73655c5c
Input: read-only snapshot of R/data/studio-datasets (LeRobot v3.0). Times in seconds from episode start. Frames decoded: yes. Verdict: OK / WARN / BAD.
## 1 Datasets
| dataset id | Studio name | default_task | environment_id | episodes | frames | fps | robot_type | camera serials |
|---|---|---|---|---|---|---|---|---|
| 6d6bdce3-b382-478d-9697-fb9e573b97af | Smoke test - inspection skill - 5 episodes | move the brick to the inspection spot | 72206569-df16-4a8b-a98f-247bac4de63f | 11 | 6693 | 30.0 | SO101_Follower | 243722064316, 243622061421 |

Caveats:
- Studio name 'Smoke test - inspection skill - 5 episodes' says 5 episodes; disk has 11 (name stale)
- task text is ['move the brick to the inspection spot', 'move the brick to the left plate'] for every episode; skill split is by gripper and motion events only

## 2 Per-episode segment cut points
| dataset | ep | frames | grasp closed frame | lift begins | S1 cut (+1 s) | S1 part frames | release frame | return to rest | S2/S3 part frames | pan at release | pan cluster | type | labels | verdict | brick at S1 cut (review) | plate (review) | release clear (review) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6d6bdce3 | 0 | 673 | 242 (8.07s) | 280 (9.33s) | 310 (10.33s) | 0-310 |  |  |  |  |  | grasp and lift, brick held to episode end (no release) | S1 | OK | yes | unassigned | na |
| 6d6bdce3 | 1 | 580 | 84 (2.80s) | 100 (3.33s) | 130 (4.33s) | 0-130 |  |  |  |  |  | grasp and lift, brick held to episode end (no release) | S1 | OK | yes | unassigned | na |
| 6d6bdce3 | 2 | 544 | 191 (6.37s) | 201 (6.70s) | 231 (7.70s) | 0-231 |  |  |  |  |  | grasp and lift, brick held to episode end (no release) | S1 | OK | yes | unassigned | na |
| 6d6bdce3 | 3 | 704 | 249 (8.30s) | 262 (8.73s) | 292 (9.73s) | 0-292 | 610 (20.33s) | 644 (21.47s) | 292-644 | -5.39 | high | full pick-and-place (grasp, lift, release, return) | S1+S2/S3 | OK | yes | unclear | unclear |
| 6d6bdce3 | 4 | 706 | 277 (9.23s) | 289 (9.63s) | 319 (10.63s) | 0-319 | 634 (21.13s) |  | 319-705 | -13.66 | high | grasp, lift, release (no return to rest) | S1+S2/S3 | WARN | yes | unclear | unclear |
| 6d6bdce3 | 5 | 647 | 235 (7.83s) | 255 (8.50s) | 285 (9.50s) | 0-285 | 549 (18.30s) | 618 (20.60s) | 285-618 | -12.62 | high | full pick-and-place (grasp, lift, release, return) | S1+S2/S3 | OK | yes | unclear | unclear |
| 6d6bdce3 | 6 | 512 | 173 (5.77s) | 196 (6.53s) | 226 (7.53s) | 0-226 | 425 (14.17s) | 479 (15.97s) | 226-479 | -13.14 | high | full pick-and-place (grasp, lift, release, return) | S1+S2/S3 | OK | yes | unclear | unclear |
| 6d6bdce3 | 7 | 1306 | 355 (11.83s) | 383 (12.77s) | 413 (13.77s) | 0-413 | 1212 (40.40s) |  | 413-1305 | 32.1 | high | grasp, lift, release (no return to rest) | S1+S2/S3 | WARN | yes | blue | yes |
| 6d6bdce3 | 8 | 342 |  |  |  |  | 152 (5.07s) | 315 (10.50s) | 0-315 | -54.67 | low | starts with brick held (no grasp in episode): release and return | S2/S3 | WARN | na | green | yes |
| 6d6bdce3 | 9 | 340 | 224 (7.47s) | 230 (7.67s) | 260 (8.67s) | 0-260 |  |  |  |  |  | grasp and lift, brick held to episode end (no release) | S1 | WARN | no | unassigned | na |
| 6d6bdce3 | 10 | 339 |  |  |  |  | 142 (4.73s) | 301 (10.03s) | 0-301 | -51.83 | low | starts with brick held (no grasp in episode): release and return | S2/S3 | WARN | na | green | yes |

Gripper events per episode (kind, frame, time):
- 6d6bdce3 ep0: close 242 (8.067s)
- 6d6bdce3 ep1: close 84 (2.8s)
- 6d6bdce3 ep2: close 191 (6.367s)
- 6d6bdce3 ep3: close 249 (8.3s), open 610 (20.333s)
- 6d6bdce3 ep4: close 277 (9.233s), open 634 (21.133s)
- 6d6bdce3 ep5: close 235 (7.833s), open 549 (18.3s)
- 6d6bdce3 ep6: close 173 (5.767s), open 425 (14.167s)
- 6d6bdce3 ep7: close 355 (11.833s), open 1212 (40.4s)
- 6d6bdce3 ep8: open 152 (5.067s)
- 6d6bdce3 ep9: open 190 (6.333s), close 224 (7.467s)
- 6d6bdce3 ep10: open 142 (4.733s)

## 3 Per-episode QA reasons
| dataset | ep | verdict | reasons | flags |
|---|---|---|---|---|
| 6d6bdce3 | 0 | OK |  | grasp and lift, brick held to episode end (no release) |
| 6d6bdce3 | 1 | OK |  | grasp and lift, brick held to episode end (no release) |
| 6d6bdce3 | 2 | OK |  | grasp and lift, brick held to episode end (no release) |
| 6d6bdce3 | 3 | OK |  | full pick-and-place (grasp, lift, release, return) |
| 6d6bdce3 | 4 | WARN | no return to rest | grasp, lift, release (no return to rest); no return to rest after release (part ends at episode end) |
| 6d6bdce3 | 5 | OK |  | full pick-and-place (grasp, lift, release, return) |
| 6d6bdce3 | 6 | OK |  | full pick-and-place (grasp, lift, release, return) |
| 6d6bdce3 | 7 | WARN | no return to rest | grasp, lift, release (no return to rest); no return to rest after release (part ends at episode end) |
| 6d6bdce3 | 8 | WARN | start pose not at home (lift/elbow) | starts with brick held (no grasp in episode): release and return |
| 6d6bdce3 | 9 | WARN | start pose not at home (lift/elbow) | grasp and lift, brick held to episode end (no release) |
| 6d6bdce3 | 10 | WARN | start pose not at home (lift/elbow) | starts with brick held (no grasp in episode): release and return |

## 4 Video files
| dataset | file | codec | size | ffprobe frames | expected | match |
|---|---|---|---|---|---|---|
| 6d6bdce3 | videos/observation.images.realsense 243722064316/chunk-000/file-000.mp4 | av1 | 640x480 | 1253 | 1253 | yes |
| 6d6bdce3 | videos/observation.images.realsense 243622061421/chunk-000/file-000.mp4 | av1 | 640x480 | 1253 | 1253 | yes |
| 6d6bdce3 | videos/observation.images.realsense 243722064316/chunk-000/file-001.mp4 | av1 | 640x480 | 3113 | 3113 | yes |
| 6d6bdce3 | videos/observation.images.realsense 243622061421/chunk-000/file-001.mp4 | av1 | 640x480 | 3113 | 3113 | yes |
| 6d6bdce3 | videos/observation.images.realsense 243722064316/chunk-000/file-002.mp4 | av1 | 640x480 | 2327 | 2327 | yes |
| 6d6bdce3 | videos/observation.images.realsense 243622061421/chunk-000/file-002.mp4 | av1 | 640x480 | 2327 | 2327 | yes |

## 5 Gaps per need (sub-episodes counted)
| need | target | minimum | usable now | short of target | short of minimum | usable ids |
|---|---|---|---|---|---|---|
| S1 | 40 | 30 | 8 | 32 | 22 | 6d6bdce3-b382-478d-9697-fb9e573b97af:0, 6d6bdce3-b382-478d-9697-fb9e573b97af:1, 6d6bdce3-b382-478d-9697-fb9e573b97af:2, 6d6bdce3-b382-478d-9697-fb9e573b97af:3, 6d6bdce3-b382-478d-9697-fb9e573b97af:4, 6d6bdce3-b382-478d-9697-fb9e573b97af:5, 6d6bdce3-b382-478d-9697-fb9e573b97af:6, 6d6bdce3-b382-478d-9697-fb9e573b97af:7 |
| S2 | 30 | 25 | 1 | 29 | 24 | 6d6bdce3-b382-478d-9697-fb9e573b97af:7 |
| S3 | 30 | 25 | 2 | 28 | 23 | 6d6bdce3-b382-478d-9697-fb9e573b97af:8, 6d6bdce3-b382-478d-9697-fb9e573b97af:10 |

Full pick-and-place episodes (count for S1 and S2/S3): 3. Unusable: []. Pending review: {'S1': [], 'S2/S3': ['6d6bdce3-b382-478d-9697-fb9e573b97af:3', '6d6bdce3-b382-478d-9697-fb9e573b97af:4', '6d6bdce3-b382-478d-9697-fb9e573b97af:5', '6d6bdce3-b382-478d-9697-fb9e573b97af:6']}

Requests (for Opus to relay to Hassan):
- S1: Record 32 more S1 parts to reach target 40 (minimum 30: 22 still missing). Full pick-and-place takes also count for S2/S3 (sub-episodes counted in the same way).
- S1: S1 part = start to 1.0 s after the grasp closes and the lift begins; the brick must be in the jaws at that cut.
- S1: Pending visual review (brick in jaws at S1 cut): [].
- S1: Grasp poses so far (pan at closure): -47.8 to -8.4 deg. Vary brick start positions outside this range.
- S1: Takes that start with the brick already held have no S1 part (2 episodes here).
- S1: Scene: Top camera 243622061421 at episode start shows several bricks (mixed colours) clustered on the mat between the blue and green plates.
- S2: Usable blue-plate S2/S3 parts now: 1 (target 30, minimum 25). Record 29 more; 24 needed for the minimum.
- S2: Release pan angles (deg, cluster): [('6d6bdce3-b382-478d-9697-fb9e573b97af:3', -5.4, 'high'), ('6d6bdce3-b382-478d-9697-fb9e573b97af:4', -13.7, 'high'), ('6d6bdce3-b382-478d-9697-fb9e573b97af:5', -12.6, 'high'), ('6d6bdce3-b382-478d-9697-fb9e573b97af:6', -13.1, 'high'), ('6d6bdce3-b382-478d-9697-fb9e573b97af:7', 32.1, 'high'), ('6d6bdce3-b382-478d-9697-fb9e573b97af:8', -54.7, 'low'), ('6d6bdce3-b382-478d-9697-fb9e573b97af:10', -51.8, 'low')]. Plate for each assigned from the top camera at the release frame; unassigned parts do not count.
- S3: Usable green-plate S2/S3 parts now: 2 (target 30, minimum 25). Record 28 more; 23 needed for the minimum.
- S3: Release pan angles (deg, cluster): [('6d6bdce3-b382-478d-9697-fb9e573b97af:3', -5.4, 'high'), ('6d6bdce3-b382-478d-9697-fb9e573b97af:4', -13.7, 'high'), ('6d6bdce3-b382-478d-9697-fb9e573b97af:5', -12.6, 'high'), ('6d6bdce3-b382-478d-9697-fb9e573b97af:6', -13.1, 'high'), ('6d6bdce3-b382-478d-9697-fb9e573b97af:7', 32.1, 'high'), ('6d6bdce3-b382-478d-9697-fb9e573b97af:8', -54.7, 'low'), ('6d6bdce3-b382-478d-9697-fb9e573b97af:10', -51.8, 'low')]. Plate for each assigned from the top camera at the release frame; unassigned parts do not count.

## 6 Calibration (summary; full values in calib.json)
- suggested max_s: {'S1': 18.8, 'S2': 37.0, 'S3': 37.0, 'rule': '1.5 x p95 of observed part durations (n small)'}
- stationary threshold 4.05 deg/s; gripper closed-on-brick median {'n': 9, 'p50': 2.547, 'p95': 2.678, 'min': 0.98, 'max': 2.678, 'values': [2.482, 2.286, 2.547, 2.678, 2.678, 2.547, 2.547, 2.482, 0.98]}

## 7 Contact sheets (top-camera release frames and S1 cut frames)
- /home/intel/intel-junction-2026/work/P2-D-DATA/W2/results3/contact-sheets/6d6bdce3_243622061421.png
- /home/intel/intel-junction-2026/work/P2-D-DATA/W2/results3/contact-sheets/6d6bdce3_243722064316.png

## 8 Self-check
| dataset | check | result | detail |
|---|---|---|---|
| 6d6bdce3-b382-478d-9697-fb9e573b97af | info.total_episodes == episodes meta rows | PASS | 11 vs 11 |
| 6d6bdce3-b382-478d-9697-fb9e573b97af | info.total_frames == data rows | PASS | 6693 vs 6693 |
| 6d6bdce3-b382-478d-9697-fb9e573b97af | info.total_frames == sum(meta length) | PASS | 6693 |
| 6d6bdce3-b382-478d-9697-fb9e573b97af | episode data rows == meta length (all) | PASS |  |
| 6d6bdce3-b382-478d-9697-fb9e573b97af | frame_index contiguous 0..L-1 (all) | PASS |  |
| 6d6bdce3-b382-478d-9697-fb9e573b97af | timestamp on frame grid, max err <= 1e-3 s | PASS | 2e-06 |
| 6d6bdce3-b382-478d-9697-fb9e573b97af | dataset index ranges match lengths (all) | PASS |  |
| 6d6bdce3-b382-478d-9697-fb9e573b97af | robot_type == SO101_Follower | PASS | SO101_Follower |
| 6d6bdce3-b382-478d-9697-fb9e573b97af | fps == 30 | PASS | 30 |
| 6d6bdce3-b382-478d-9697-fb9e573b97af | stored joint names == contract joint names | PASS | ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos'] |
| 6d6bdce3-b382-478d-9697-fb9e573b97af | camera serials == NEEDS pair | PASS | ['243622061421', '243722064316'] |
| 6d6bdce3-b382-478d-9697-fb9e573b97af | all video files present and probed | PASS |  |
| 6d6bdce3-b382-478d-9697-fb9e573b97af | video frame counts match meta (+-2, all files) | PASS | {'videos/observation.images.realsense 243722064316/chunk-000/file-000.mp4': 1253, 'videos/observation.images.realsense 243622061421/chunk-000/file-000.mp4': 1253, 'videos/observation.images.realsense 243722064316/chunk-000/file-001.mp4': 3113, 'videos/observation.images.realsense 243622061421/chunk-000/file-001.mp4': 3113, 'videos/observation.images.realsense 243722064316/chunk-000/file-002.mp4': 2327, 'videos/observation.images.realsense 243622061421/chunk-000/file-002.mp4': 2327} |
| 6d6bdce3-b382-478d-9697-fb9e573b97af | first and last frame decode (all episodes, both cams) | PASS |  |
| 6d6bdce3-b382-478d-9697-fb9e573b97af | closed-on-brick median within 1.5..3.5 (all grasps) | FAIL | min 0.98 max 2.678 |
| 6d6bdce3-b382-478d-9697-fb9e573b97af | open peak before grasp within 10..36 (all grasps) | FAIL | min 16.264 max 36.381 |
| 6d6bdce3-b382-478d-9697-fb9e573b97af | every grasp is a decrease (open peak - closed > 8) | PASS |  |
| 6d6bdce3-b382-478d-9697-fb9e573b97af | Studio name episode count agrees with disk | FAIL | Studio name says 5 episodes, disk has 11 |

## 9 Coverage checklist
- [x] 6d6bdce3-b382-478d-9697-fb9e573b97af: info, tasks, episode meta, data parquet read
- [x] 6d6bdce3-b382-478d-9697-fb9e573b97af: Studio identity GET
- [x] 6d6bdce3-b382-478d-9697-fb9e573b97af: video videos/observation.images.realsense 243722064316/chunk-000/file-000.mp4 probed
- [x] 6d6bdce3-b382-478d-9697-fb9e573b97af: video videos/observation.images.realsense 243622061421/chunk-000/file-000.mp4 probed
- [x] 6d6bdce3-b382-478d-9697-fb9e573b97af: video videos/observation.images.realsense 243722064316/chunk-000/file-001.mp4 probed
- [x] 6d6bdce3-b382-478d-9697-fb9e573b97af: video videos/observation.images.realsense 243622061421/chunk-000/file-001.mp4 probed
- [x] 6d6bdce3-b382-478d-9697-fb9e573b97af: video videos/observation.images.realsense 243722064316/chunk-000/file-002.mp4 probed
- [x] 6d6bdce3-b382-478d-9697-fb9e573b97af: video videos/observation.images.realsense 243622061421/chunk-000/file-002.mp4 probed
- [x] 6d6bdce3 episode 0: segmented and QA row written
- [x] 6d6bdce3 episode 1: segmented and QA row written
- [x] 6d6bdce3 episode 2: segmented and QA row written
- [x] 6d6bdce3 episode 3: segmented and QA row written
- [x] 6d6bdce3 episode 4: segmented and QA row written
- [x] 6d6bdce3 episode 5: segmented and QA row written
- [x] 6d6bdce3 episode 6: segmented and QA row written
- [x] 6d6bdce3 episode 7: segmented and QA row written
- [x] 6d6bdce3 episode 8: segmented and QA row written
- [x] 6d6bdce3 episode 9: segmented and QA row written
- [x] 6d6bdce3 episode 10: segmented and QA row written
