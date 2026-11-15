# P3-CODE-REVIEW / E1: review of the pick-anywhere code (read-only)

Revision reviewed: R/app HEAD bbb3d75 (baseline 800e3c9 at 19:55Z, then 2dd4a23, 871e4b8, 8afa94e, bbb3d75). The 167 in-scope files were re-hashed at 20:35Z against raw/03-sha256-scope.txt: 167 OK, 0 changed (raw/32-sha-recheck.txt). Working tree: one untracked file, simgate/watch_sim_v2.sh (outside scope; F-87). Nothing under R/app or R/docs was edited. Findings: findings.tsv (93 rows, F-01..F-93). Every row has file, line, severity, category, finding and suggested fix.

Severity count (findings.tsv, 93 rows): critical 3, high 27, medium 44, low 19.

## 1. Correctness and safety risks

Ranked by how directly they can move the arm wrongly, skip the hand check, misplace a brick, lose a recording, or crash the loop.

1. **Descent from the size estimate, no hard floor (critical, F-47).** sort_any.py:340, 343, 358, 363. use = tip if |tip - fk| <= 15. Final TCP height = TIP_FINAL + (fk - use). A size estimate 15 mm above FK puts the tips about 7 mm below the mat. The retry path (tip_final 6) reaches about 0 mm. No measured height limits the last descent. Scenario: a tilted or upside-down brick gives a large size error, the arm presses the jaws into the mat, and the fixed jaw loosens (PICK-ANYWHERE already says contact loosens it).
2. **Stale motion target after e-stop (critical, F-01).** motion_server.py:154-156, 221, 224. move() sets self.target = q even when paced() stopped on an e-stop. The next rel or grip builds on that unreached target, so the arm moves from present to the unreached pose.
3. **Clear check reads stale frames and removes the stop (critical, F-37).** sort_loop.py:54-92 reads run/live_*.jpg with no age check and removes run/estop when no hand is seen. A stopped recorder means the last frame is reused indefinitely, so the loop moves with no live sensing. monitor_cams.sh has no restart (F-72, script ends in wait, line 19).
4. **Fail-open clearance and blind final descent (high, F-09, F-10, F-41).** grasp_auto.py:305-309 lets stage 2 run with no clearance measured (None passes the guard). Lines 319-320 use a fixed extra when the jaw height is missing. sort_loop.py:259 and :267 repeat the pattern (6.8 fallback). Breaks lesson L-049.
5. **Gripper opened over a held brick (high, F-43).** sort_loop.py:386, 450, 462 open the gripper (25) on any exception or stop with no held check; the stop path places only if the plate was already decided (444-446).
6. **Right jaw not checked (high, F-61).** jaw_check.py:85 and :88 test d[0] only. A right-jaw shift passes the start-up check, the periodic check (sort_any.py:456 runs it with --no-move) and the rewrite of the grasp target.
7. **Periodic jaw check fails open (high, F-45, F-46).** sort_any.py:461 logs "jaw check skipped" and the loop continues. A shift of 25 px or more logs "jaw check ok" (473) and keeps the old target.
8. **Hand watchdog silent death (high, F-06, F-20).** hand_watch.py:60-92 has no heartbeat and no exception guard. Nothing stops a move in flight, because the server acts only on run/estop.
9. **Blind region of the close view (high, F-07).** handdet.py:52-54 and 76-81: while the gripper holds, the close view removes every skin component touching the zone, so a hand at the jaws is invisible to it. The wrist view is off while holding (hand_watch.py:67).
10. **Neighbour switch in the servo (high, F-57).** pick_any.py:122-148 falls back to compatible colours and servo() passes no size reference (150-177). This is the 17:56Z switch that lesson L-054 forbids.
11. **Lost episode record (high, F-44, F-51).** sort_loop.py:466-473 and sort_any.py:563-581 index os.listdir results with [-1]. With no top_ or close_ file the IndexError hides the original error and episode.json is not written.
12. **Wrong brick at station re-pick (high, F-50).** sort_any.py:410-413 takes candidates(...)[0], which can be a different brick from the one put down.
13. **Commands dropped by a partial read (medium, F-05).** motion_server.py:189-193 reads a line still being written, consumes it, and answers id -1. The client then waits for its ack (grasp_auto cmd timeout 120 s).
14. **Motion stalls without a hold (high, F-03).** so101_motion.py:457-516: MotionStalledError is raised (line 516) and nothing calls hold(), though the docstring (500) says the arm is held where it stopped.
15. **Studio sessions stopped by a repo tool (high, F-69).** studio_sessions.py:27-30: --stop-all stops every runtime session on 127.0.0.1:7860 with no confirmation.
16. **Sync scripts read folders outside the project (high, F-70).** sync_sim*.sh and sync_datasets*.sh (SRC, line 3) read /home/intel/junction-cell/ and /home/intel/.local/share/physicalai/. LAPTOP.md names other folders under /home/intel as off limits. Reviewed by reading only; not run. Needs owner confirmation before the push.
17. **Runtime tool rewrites a tracked file (high, F-62).** servo_step.py:50-54 writes app/motion/gm/servo.yaml on every run.

## 2. Robustness gaps

- Unhandled exceptions end the process with no result note: grasp_auto main catches only Stop (F-32). Recovery gives up after 40 attempts with no needs_operator file (sort_any.py:604-623, F-52; P2-C item 2 covers this).
- Stale or missing frames accepted: fresh() returns an image after 3 s with no age check (grasp_auto.py:84-95, F-11); snap() can return None (F-11).
- Stale pid and stale motion_active: grasp_auto.py:233 tests only that the pid file exists (F-16); resume.sh never clears motion_active or estop (F-12).
- Non-atomic writes: motion_server.py:166 and so101 write_pose (804, F-89) and mcmd append (F-90).
- Unlogged degradation: kin not importable makes the wrist view silently unused (hand_watch.py:28-31, F-25); TARGET_PX and JAW_MASK fall back silently (grasp_auto.py:29-33 and 44-45, F-18).
- Calibration files without checks: kin.load_calib returns None (kin.py:54-58) and m_base raises TypeError mid-motion (F-59); jaw_calib.py rewrites the target with no reference check (F-67); calib_M and calib_wrist overwrite the same wrist_calib.json with no residual gate (F-65).
- Scripts that move the arm and then crash leave it displaced: scan_calib.py (F-68), jprobe.py (F-64), calib_wrist.py (F-65).
- Teaching writes: write_pose is not atomic (F-89).
- Model loaded at import: wristcam.py raises TypeError when the model file is missing (F-80); collect.py skips silently (F-81).
- Race between the loop, the server and the watchdog: the loop's clear check, the e-stop write and the server's ESTOP path act on the same run/estop flag with no heartbeat (F-06, F-37, F-20).
- State that survives a restart wrongly: the periodic jaw-target change is in memory only (F-46); the start-up check runs only with gripper target at least 20 (sort_any.py:680, F-53).

## 3. Maintainability

- Hard-coded roots and sys.path in about 15 places; kin.py loads the URDF from repos/ outside R/app, so the push is not self-contained (F-23). P3-SIM-LOOP's path patch must change all of them.
- Gripper "held/closed" predicate has five thresholds (hand_watch 20, server 5, grasp_auto 1.2 and 1.0, sort_loop 0.6 and 1.0-6.0, PICK-ANYWHERE 1.1) (F-21, F-42).
- Final-height and descent constants unnamed and inconsistent (F-22). Duplicate constants in two units (kinematics pan axis in m, kin.py in mm; F-76).
- Duplicated logic: skin and red-jaw masks copied from handdet.py into sort_loop (F-39); three red-jaw detectors with different thresholds (jaw_check.py, jaw_calib.py, wrist_brick.py; F-84); two wrist-brick detectors (grasp_auto.wrist_brick and wrist_brick.py; F-84); camera projection twice (calib3.py and wristcam.py; F-78); motion server twice (motion_server.py and motion_server_v1.py; F-73); sync scripts in pairs (F-71).
- Dead code: motion_server_v1.py (F-73), solve_ik_multi (F-74), calib_M.gray (F-66), rs_jaws unused flags (F-82), GRASP_DZ (F-60), handdet n0 (F-29), handdet AREA_EDGE dead branch (F-30).
- Functions over 60 lines (ast, raw/39-fn-lengths-structure.txt): grasp_auto.main 148 lines, sort_loop.pick 90, sort_any.cycle 85, motion_server.main 84, rs_height.main 79, sort_loop.main 71, servo_step.main 69, motion_server_v1.main 69. 8 of 261 functions over 60 lines across 40 Python files.
- Comments and docstrings narrate incidents: "Opus I-59", "17:47Z drop", "13:42Z coordinator", "Validated 17:13Z" (kin.py:2), in several places (F-26). tools/README.md asks authors to write the incident in every comment, which conflicts with the review criterion. Incident history belongs in docs/CHANGELOG.md or commit messages.
- Magic numbers: 65.0 field of view and 24 jaw threshold repeated in five scripts (F-83); 3 px and y 268 (F-67).
- Import-time work: sort_any reads spot_fast.yaml at import (F-24); sort_loop and grasp_auto read masks at import (F-24).
- Missing type hints and docstrings: the loop's tool scripts (sort_loop, sort_any, pick_any cycle, rs_*, calib_*) have module docstrings but no typed interfaces; kinematics.py and so101_motion.py are typed.
- Stale docs: tests README reports 2 FAIL; HEAD gives 11 PASS (F-86). Docs say recovery about 15 min, code about 18 min (F-86).

## 4. Test gaps (with the test that would protect each function)

Current suite (raw/30-tests-run.txt): 11 PASS, 0 FAIL, 0 ERROR, 0 SKIP, 2.2 s, exit 0. Covers pick_any.blobs/find (hover, servo, split), sort_any.candidates (plate exclusion), sort_any.close_pre_verdict, handdet.is_hand (smoke and labelled top/close), kin.fk heights, kin.ik_rel grid, sort_any.grasp_reachable.

| function or module | gap | test that would protect it |
|---|---|---|
| motion_server.Server.move, paced, rel, grip | no test (F-33) | scripted fake player: cut paced mid-move, assert target == present, rel/grip re-anchor (F-01) |
| hand_watch.main | no test (F-34) | one synthetic frame plus fake state; estop only when active and persistent; crash path (F-06) |
| grasp_auto.jaws_clear_of_brick, stage guards | no test (F-35) | synthetic height arrays: clearance None backs off; missing jaw height aborts (F-09, F-10) |
| sort_any.grasp descent | no test of the 15 mm size error (F-47) | size estimate 15 mm above FK must abort; floor on commanded z |
| sort_loop.wait_clear, hands_seen | no age check tested (F-37) | stale frame and missing heartbeat are not clear |
| sort_loop finally / save_ep | no test (F-44, F-51) | no top_/close_ files: episode.json still written |
| jaw_check gate | right jaw untested (F-61) | synthetic right-jaw shift must fail |
| handdet.gripper_zone, holding branch | no test of hand at jaws (F-07) | labelled hand-at-jaws frames with and without brick |
| handdet features | fragmented hand untested (F-08) | fragmented-hand fixture summed over bounding box |
| pick_any.servo, find COMPAT | no test of same-size compatible neighbour (F-57) | neighbour of compatible colour nearer than target must not be chosen |
| so101_motion.plan_ramp, validate_target | no test (F-36) | peak rate <= max_speed, peak accel <= max_accel, last setpoint == target |
| so101_motion.MotionPlayer.run stall | no test (F-03) | fake driver that never reaches goal; assert hold called |
| so101 write_pose | no test (F-89) | interrupted write leaves old file intact |
| poses_from_calib (validate, refuse_taught) | no test | placeholder and taught-file refusal; speed range |
| run_sequence dry run and exit codes | no test | exit 2 on stall, exit 4 on bad speed |
| kin.load_calib None path | no test (F-59) | missing wrist_calib.json fails at start |
| kin.ik / ik_rel roll limit | roll limit 97 vs 99 untested (F-58) | roll clip is the single config value |
| pick_v2 rot_log near pi | no test (F-75) | rotation at pi returns the right sign |
| wristcam (model missing) | no test (F-80) | missing model: explicit error, not TypeError |
| jaw_calib / jaw_check masks | no test | fixed frame: both tips and target |
| rs_* depth tools, servo_step | no test; rs_box usage ok | rs_jaws on a recorded depth map (none in fixtures) |
| mcmd id and reply matching | no test (F-90) | id uniqueness across restarts |

The suite is fast and offline. run_tests.py removes its temp folder; no other write check was done.

## 5. Performance hot spots (measured, offline)

Measured with raw/31-time_fixtures.py on the fixture frames (640x480), 3 repetitions per call (raw/31-timing.txt). These are this laptop's numbers, not the robot PC's.

| call | median | max | where it is used |
|---|---|---|---|
| sort_any.candidates on scan_now.jpg | 26.4 ms | 128.3 ms | wait_for_bricks polls every 0.15 s (F-55) |
| pick_any.blobs, 33 frames | 16.7 ms | 28.0 ms | every servo iteration and every candidates call (F-56) |
| handdet.is_hand, close (holding=False) | 6.9 ms | 9.3 ms | hand watchdog, up to 3 views every 0.25 s (F-28) |
| handdet.is_hand, close (holding=True) | 7.9 ms | 9.7 ms | |
| handdet.is_hand, wrist | 4.6 ms | 4.8 ms | |
| handdet.is_hand, top | 2.1 ms | 2.5 ms | |
| pick_any.find | 0.002 ms | 0.002 ms | negligible |

Consequences: the three hand views take about 17 ms per watchdog tick in the worst case (9.3 ms close, 4.8 ms wrist, 2.5 ms top, all at their maxima), about 7% of its 0.25 s period. candidates() alone takes about 17% of each 0.15 s poll interval at the median and up to 85% at its maximum, before the fresh() wait. Caching detections per frame (F-55) and computing HSV once per frame (F-28) are the first two changes to measure after the refactor.

Not measured: rs_height plane fitting (600 RANSAC iterations on 40000 points), solve_top_down (8 seeds, up to 80 iterations, each with 6 FK calls), and the 65-70 s typical cycle claim in PICK-ANYWHERE. These are estimated statically only.

## 6. AI, Claude, Anthropic, Opus, Haiku, "agent", "assistant" mentions under R/app and R/docs

Method: grep -rnIi over R/app and R/docs (excluding .git and __pycache__), case-insensitive, pattern claude|anthropic|opus|haiku|agent|assistant|standalone ai. Raw output: raw/24-ai-mentions-full.txt; re-check at 20:35Z with the same count (raw/25-ai-mentions-recheck.txt, 200 lines; diff only in path prefix, raw/34-ai-diff.txt). File-name check (find -iname for claude, anthropic, opus, haiku, agent, assistant): no matching file names.

Total: 200 lines in 74 files. By folder: anomaly 129, tools 38, loop_v2 6, pick_v2 7, loop_v3 4, eyeinhand 3, docs 3, loop 2, motion 2, telemetry 2, dashboard_v5 1, ovrun 1, simgate 1, vla 1.

Caution: the pattern also matches "ai" as a word (for example the path segment "edge-ai-demo-studio" in kin.py:19, a third-party repo path) and words like "agent" in unrelated text. Each line below is a mention to review, not necessarily a branding mention.

In-scope list (app/tools, app/motion, app/eyeinhand, docs/PICK-ANYWHERE.md; this is the list that must change before the push):

- app/tools/scan_calib.py: 2
- app/tools/pick_any.py: 2
- app/tools/motion_server_v1.py: 2
- app/tools/README.md: 1
- app/tools/rs_height.py: 2
- app/tools/jaw_calib.py: 2
- app/tools/grasp_auto.py: 2
- app/tools/sort_any.py: 2
- app/tools/rs_footprint.py: 2
- app/tools/review_ep.py: 2
- app/tools/peek.py: 1
- app/tools/rs_box.py: 2
- app/tools/wrist_rec.sh: 2
- app/tools/sort_loop.py: 2, 453
- app/tools/monitor_cams.sh: 2
- app/tools/calib_wrist.py: 2
- app/tools/jaw_check.py: 2
- app/tools/hand_watch.py: 1
- app/tools/handdet.py: 7 ("Opus visual labels" and work/handdet path)
- app/tools/wrist_brick.py: 2
- app/tools/tests/conftest.py: 106
- app/tools/tests/README.md: 39, 44
- app/tools/tests/fixtures/MANIFEST.tsv: 50 (not read in full; match only)
- app/tools/tests/test_perception.py: 138
- app/tools/tests/test_kin.py: 6
- app/tools/rs_grab.py: 2
- app/tools/calib_M.py: 2
- app/tools/resume.sh: 2
- app/tools/kin.py: 2, 19 (URDF path under repos/.../robotics-ai, a third-party path)
- app/tools/studio_sessions.py: 1
- app/tools/motion_server.py: 2
- app/tools/servo_step.py: 2
- app/tools/rs_jaws.py: 2
- app/tools/mseq.py: 2
- app/tools/jprobe.py: 2
- app/eyeinhand/collect.py: 1
- app/eyeinhand/calib3.py: 1, 208
- app/motion/gm/spot_grasp_v1.json: 2 ("_note")
- app/motion/tools/poses_from_calib.py: 149
- docs/PICK-ANYWHERE.md: 3, 90, 181

Outside scope (also under R/app and R/docs, must go before the push or be excluded from it):

- app/anomaly/det2_f2/calib_tape.py: 29 lines (1, 5, 7, 9, 74, 83, 84, 86, 95, 112, 114-119, 125, 135, 143-145, 147, 149, 155, 158, 160, 166, 172, 177)
- app/anomaly/det2_v1/calib_tape.py: same 29 lines
- app/anomaly/det2_e2/calib_tape.py: same 29 lines
- app/anomaly/det2_f2/zoom_disputed.py: 8 (1, 19-24, 58); same 8 in det2_v1 and det2_e2 (zoom_disputed.py)
- app/anomaly/det2_f2/models/tape_thresholds.json: 11; same in det2_v1 and det2_e2
- app/anomaly/det2_f2/tests/test_inspector.py: 252
- app/anomaly/det2_e3/inspector.py: 20; app/anomaly/det2_e3/sheets.py: 1
- app/anomaly/anom3/infer.py: 6; eval.py: 12; loeo.py: 15; train.py: 4, 191; RUN.md: 23, 34, 45, 47, 50, 66; extract_crops.py: 3
- app/loop_v2/app/loop/studio_runtime_client.py: 1, 7; policy_cycle.py: 60; poses.yaml: 4; policy_live.py: 148; loop_v2/tests/test_v2.py: 307
- app/loop/studio_runtime_client.py: 1, 7
- app/loop_v3/app/loop/run_spot.py: 248; spot_rules.yaml: 14, 25, 30
- app/pick_v2/tests/test_tip_calib.py: 20-23; app/pick_v2/tip_calib.py: 54, 55, 441
- app/vla/IMPORT.md: 3
- app/ovrun/runner.py: 205
- app/simgate/geometry_handoff.py: 82
- app/telemetry/final_pack.py: 82; telemetry/run_final.sh: 2
- app/dashboard_v5/notes.md: 10

Git history: the baseline commit 800e3c9 contains all of the above, so the mentions remain in history after a working-tree cleanup. The push needs an orphan history or a history rewrite, decided by the owner.

## 7. Sequenced refactor plan (starts only after both patches land)

Gate 0. Opus has applied P3-SIM-LOOP's R/sim/loop/app_paths.patch (JUNCTION_RUN_ROOT) and P2-C-LOOPFIX's loopfix.patch (heartbeat, needs_operator, tests) to R/app. Both patches are pending at the time of this review. Before step 1: run the suite (must be 11 PASS plus the new tests) and record the output. No step starts without that.

Each step is one small commit; the suite must stay green after each.

1. Remove dead code: motion_server_v1.py (F-73), solve_ik_multi (F-74), calib_M.gray (F-66), rs_jaws unused flags and their call-site arguments (F-82), GRASP_DZ (F-60), handdet n0 (F-29). Tests: suite unchanged.
2. Move the repo-root and sys.path logic into one config module that reads JUNCTION_RUN_ROOT (F-23, F-83, F-76). Tests: suite plus a test that the default root equals today's paths.
3. One gripper-state predicate with named constants (F-21, F-42, F-22). Tests: boundary tests at each threshold.
4. Fail-closed safety, one commit per finding: F-01 (re-anchor target), F-03 (hold before stall raise), F-09 and F-10 and F-41 (None means not clear; no fixed final extra), F-47 (descent from FK with floor; size estimate as cross-check with tolerance), F-61 (both jaws), F-37 (frame age and heartbeat; remove estop only under both), F-43 (keep holding on stop), F-44 and F-51 (guard listdir, write episode record first), F-45 and F-46 (failed jaw check pauses). Each commit adds the test named in section 4. These change behaviour; each needs a replay or sim run before the robot.
5. Motion-server adapter and writes: one close adapter for the private driver calls (F-14); atomic state, pid and command writes (F-15, F-89, F-90); rotating log (F-13); command reader keeps partial lines (F-05). Tests: server with a fake driver.
6. Handdet: zone exclusion limited to held-brick colour (F-07); sum fragments (F-08); remove dead edge branch (F-30); reuse one HSV conversion (F-28). Tests: hand-at-jaws labelled frames, fragment fixture; timing report.
7. Performance after tests: cache candidates per frame and reachability per brick track (F-55); blobs with bounding boxes and one HSV plane (F-56). Gate: the timing report must improve and the suite stays green.
8. Calibration writers: one calibration writer with residual gates (F-65, F-67, F-68); load calibration once at start (F-59).
9. Narration and docs last: remove incident comments (F-26) and stale doc figures (F-86); move history to docs/CHANGELOG.md.
10. Pre-push gate: remove studio_sessions.py from the repo or gate it (F-69); resolve sync scripts and their paths with the owner (F-70, F-71); commit or remove simgate/watch_sim_v2.sh (F-87); delete the AI mentions in section 6 (in and out of scope); decide the history rewrite; run the full suite and the timing report once more.

## 8. Coverage checklist

Source: R/app/tools (all .py, .sh, .md, .json, .jsonl), R/app/tools/tests (all .py, README.md, fixtures), R/app/eyeinhand (all), R/app/motion (all .py, .yaml, .json, tools/), R/app/pick_v2/kinematics.py, R/docs/PICK-ANYWHERE.md. Inventory: raw/01-inventory-scope.tsv (179 lines, incl. directories and binary fixtures). Hashes: raw/03 (167 scoped files, all OK at 20:35Z).

- [x] app/tools/motion_server.py, hand_watch.py, handdet.py, resume.sh, grasp_auto.py, sort_loop.py, sort_any.py, pick_any.py, kin.py (read in full, earlier session)
- [x] app/tools/motion_server_v1.py (read in full)
- [x] app/tools/mcmd.py, mseq.py, servo_step.py, jprobe.py, jaw_check.py, jaw_calib.py, calib_wrist.py, calib_M.py, scan_calib.py (read in full)
- [x] app/tools/rs_grab.py, rs_height.py, rs_footprint.py, rs_jaws.py, rs_box.py, wrist_brick.py, peek.py (read in full)
- [x] app/tools/review_ep.py, studio_sessions.py, arm_free_watch.sh, monitor_cams.sh, wrist_rec.sh, sync_datasets.sh, sync_sim.sh, sync_datasets_v2.sh, sync_sim_v2.sh (read in full; none run)
- [x] app/tools/README.md (read in full)
- [x] app/tools/wrist_calib.json, jaw_calib_brickbg.json, scan_calib.json (read in full)
- [x] app/tools/jaw_calib.json (read in an earlier session via the log entry, call 10)
- [x] app/tools/scan_pairs.jsonl (9 lines, shown in full in the saved batch output)
- [ ] app/tools/tests/fixtures/MANIFEST.tsv (85 lines): not read; provenance and checksums not reviewed
- [x] app/tools/tests/fixtures/reach_cases.tsv (10 lines, shown in full)
- [x] app/tools/tests/fixtures/hand_labels.csv (41 rows, shown in full)
- [ ] fixture images (hover, servo, split, hand, close, scan_now.jpg): used only through the test suite and timing; not inspected by eye
- [x] app/tools/tests/run_tests.py, conftest.py, test_kin.py, test_perception.py, README.md (read in full)
- [x] app/eyeinhand/calib3.py, collect.py, wristcam.py, README.md (read in full); wrist_cam_model.json (100 lines, read in the saved batch output; structure only)
- [x] app/motion/run_sequence.py, teach_poses.py, tools/poses_from_calib.py, poses.example.yaml (read in full)
- [x] app/motion/so101_motion.py (822 lines, read in full, earlier session)
- [x] app/motion/project_robots.json (structure only: shape with values masked; no values printed, raw/39)
- [ ] app/motion/gm/*.yaml and gm/spot_grasp_v1.json (8 files, 1404-1584 lines for the spot files): line counts only; top-level keys not parsed (raw/39). Values not reviewed.
- [x] app/pick_v2/kinematics.py (405 lines, read in full)
- [x] docs/PICK-ANYWHERE.md (192 lines, read in full); docs/SPOT-GRASP-DEPTH.md (not in review scope; hashed only)
- [x] AI-mention grep: raw/24 and raw/25 (200 lines), section 6 (in-scope and out-of-scope lists)
- [x] function-length scan: raw/39 (261 functions in 40 Python files)
- [x] offline suite: raw/30 (11 PASS); timing: raw/31 (candidates, blobs, handdet, find)
- [ ] git log --stat per commit: raw/12 (read in the earlier session, not reread)
- [x] R/app git status: raw/33 (one untracked file)

## 9. Self-check (R12): facts re-verified in the source during this session

1. jaw_check.py:85 and :88 test d[0] only (grep, raw/35).
2. servo_step.py:54 writes app/motion/gm/servo.yaml (grep, raw/35).
3. sort_any.py:456 and :680 run jaw_check with --no-move (grep, raw/35; the start-up check is skipped when the gripper target is below 20).
4. kin.py:40 defines ik(); calib_wrist.py:28 and :34 call it (grep, raw/35). Not a missing function.
5. kin.py:71 PAN_XY_MM = (38.8353, 0.0) mm; pick_v2/kinematics.py:351 PAN_AXIS_XY = (0.0388353, 0.0) m (grep, raw/36).
6. kinematics.py:359 ROLL_AT_PSI0 = -2.7438 (grep, raw/36).
7. so101_motion.py:804 write_pose (grep, raw/35).
8. collect.py:55 check=False (grep, raw/36).
9. monitor_cams.sh:19 is the wait line; no restart loop (file read).
10. mcmd.py:15 and :22 (id formula and reply match, grep, raw/36).
11. calib_M.py has no call to gray() (grep, raw/35 shows only the definition).
12. rs_box.py:14 reads sys.argv[4]; the usage line passes "--min-h 2.5", so the value is correct. Not a bug; recorded as a style issue only (no argparse).
13. Sync scripts SRC on line 3 (file read).
14. Measured: candidates median 26.4 ms; blobs 16.7 ms; handdet close 6.9 ms (raw/31).
15. Suite: 11 PASS, 0 FAIL at HEAD bbb3d75 (raw/30). The README's "last run: 2 FAIL" is stale (F-86).

Correction made during the check: none of the earlier findings needed reversal. Line citations for F-03, F-04, F-11 and F-36 were corrected (raw/37).

## 10. Notes for the coordinator

- Desktop Commander budget: about 27 Desktop Commander calls in this resumed session (earlier calls are in laptop-log.md); the hard limit for the rest of the task is 80.
- The memory file and the project docs were not read or changed by this agent.
- P2-C-LOOPFIX names wait_for_bricks in the loop's clear check. The function exists in sort_any.py:205 only; sort_loop.py has wait_for_brick (line 95). The patch should name the function it changes.
