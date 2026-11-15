# INTERFACE.md: what the pick-anywhere loop reads and writes (E1, P3-SIM-LOOP stage S1)

Scope: the loop = app/tools/sort_any.py and the modules it imports: grasp_auto (ga), pick_any (pa), sort_loop (sl), kin, handdet, inspector (app/anomaly/det2_f2), rs_height. R = /home/intel/intel-junction-2026.
Citations are file:line in R/app/tools unless a full path is given. Short names: sa = sort_any.py, sl = sort_loop.py, ga = grasp_auto.py, pa = pick_any.py, kin = kin.py, ms = motion_server.py, hw = hand_watch.py, mp = app/motion/so101_motion.py.
Verification: grep -n and read_file on 2026-11-14 ~20:10-20:50Z. Saved grep outputs: R/work/P3-SIM-LOOP/E1/s1_*.txt. Source identity: sha256 in work/P3-SIM-LOOP/E1/laptop-log.md (L-04).

## 0. Access notes
- Files outside the spec's list that were read (read only): R/work/P3-TWIN-DYN/E1/full_result.json (coordinator-directed), R/sim/twin/dynamics/twin_dynamics_v1.json, R/logs/sort/any_1740/loop.log and R/logs/sort_any_any_1740.out (comp_iters check), R/logs/sort/any_1740/ep_0017 listing.
- One metadata-only listing touched R/run: `ls -la run/mserver/server.log` (size and mtime; no content read). Disclosed in STATUS.md.

## 1. Roots, literals, environment
- R literal: ga:24, kin:18, hw:24, ms:41, resume.sh:8, monitor_cams.sh:6, wrist_rec.sh:5. sa:31, sl:27, pa:22 take R from ga.R.
- sys.path literals (code location): sa:23-24, sl:22-23, ga:21, pa:18, kin:15 (R/app), hw:25, ms:30 (R/app/motion), jaw_check.py:16. inspector.py:26 uses its own directory (HERE).
- Environment variables read by any loop module: none (grep environ|getenv over sa, sl, ga, pa, kin, handdet, hw, ms: no hits; s1_grep_io.txt).
- Interpreters (absolute): ga:26 PY = R/envs/data-env/bin/python; sa:456 and sa:680 run jaw_check.py with it, cwd R/app/tools. Motion server = R/envs/motion-env/bin/python (resume.sh:42).
- Time: all stamps UTC (time.gmtime / strftime). Episode folders ep_%04d.

## 2. Motion server protocol (real process: app/tools/motion_server.py)
Files under D = R/run/mserver (ms:42):
- reads D/cmd.jsonl: append-only JSON lines written by the client; the server keeps a byte offset (ms:173-175; read loop ms:189-191).
- writes D/state.json: written to D/state.tmp, then os.replace (ms:165-167), after every command (ms:239).
- writes D/server.log (log() ms:53-55; state JSON logged ms:168); D/pid created at start (ms:184), removed at exit (ms:252-253).
- R/run/estop (ms:43) read by Server.estop() (ms:81-82). R/run/motion_active (ms:44) created while a move runs (ms:131), removed afterwards (ms:152-153).
- Follower port constant (ms:45): /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D040939-if00. resolve_arm reads R/app/motion/project_robots.json (ms:178).
- Constants: PORT ms:45; GRIP = 5 ms:46; HARD_SPEED = 45.0 ms:47; HARD_ACCEL = 200.0 ms:48; OFFSET_CLAMP = 8.0 ms:49.

Ops (dispatch ms:204-238; reply after every op ms:239):

| op | request fields (server default) | server code | loop callers |
|---|---|---|---|
| move | q[6], speed, comp (default True, ms:205), accel (default 60) | Server.move ms:113-157 | ga:78 (ga.move); pa:113 (pa.move_rel); pa:192 (align_yaw, speed 25, accel 120); sa:421 (tmove, accel 120); sa:398, 403 (ga.move) |
| rel | dq[6] added to target; speed, comp (default True), accel (ms:220-222) | supported | no loop caller |
| grip | value, speed (default 20), accel 120, keep_offset; comp False (ms:223-226) | grip only, body goal kept | sa:369, 384, 400, 442, 522, 615, 644; legacy ga:328, 335, 351 |
| state | none (ms:227) | returns current state | ga:213-214 (ga.state) |
| rec_start | path; CSV header t,goal0..5,pos0..5, rows about 30 Hz (ms:206-213; rec_tick ms:68-80) | joints log | sa:493 with path = ep_dir/joints.csv |
| rec_stop | none (ms:214-219) | closes joints log | sa:502, 524, 538, 557, 608, 642 |
| exit, release | none (ms:229-231); release also torque off (ms:243) | | no loop caller |

Unknown op: msg "unknown op X", status unchanged (ms:235). Exceptions: status "error", msg "error: ..." (ms:236-238).

Reply fields (Server.state ms:159-168; dict built ms:161-164):
- last_id (request id), status (ready | started | ok | estop | error), msg, t (HH:MM:SSZ),
- target[6] (last commanded), goal[6] (last setpoint sent), present[6] (measured, from pl.current()), err[6] = target - present, offset[6] (compensation offset).
- status "ok" only after ramp and compensation finish (ms:155-156). An e-stop refuses a move (ms:114-116) or holds at present (ms:104).
- Units: normalized SO-101 units (body -100..100, gripper 0..100). Joint order (project_robots.json "joint_order"): shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper (index 5 = GRIP).

Client side (ga):
- ga.cmd(c) (ga:60-75): adds id = int(time.time()*1000) % 1e9, appends the line to SD/cmd.jsonl (ga:63), polls SD/state.json until last_id == id (ga:68). Timeout 120 s, then Stop.
- ga.move(q, speed, comp=True) (ga:77-82): status != "ok" raises Stop. pa.move_rel (pa:116-118), pa.align_yaw (pa:192-194) and sa.tmove (sa:421-423) check status the same way.
- Fields the loop reads: status and msg (only inside Stop text), last_id (ga:68), present[5] (gripper: sa:371, 378, 435-436, 509-511, 610-611, 643-644, 680, 685), present[:5] (body: sa:395, 435-436, 509), target[5] (holding test: sa:610-611, 643-644, 680, 685; sl:72; hw:45-47).
- err is read only by legacy code (ga main, grasp_open note; sl legacy cycle).
- Gripper meaning (docs/PICK-ANYWHERE.md, grasp step 5): empty close reads 0.59; held 2x2 or 2x4 reads 1.57-2.42; held = present[5] >= 1.1 (test at sa:383; release on a miss at sa:384).

## 2b. Compensation and goal ramp: the real code path that the S3 sim server must port
Coordinator note applied: the sim server reproduces the goal ramp and the compensation iterations with the same constants and reply fields.
Source: R/app/tools/motion_server.py (line numbers verified with grep -n; saved in s1_grep_comp.txt, sha256 prefix da0633a28cebb3d5).
- Constants: GRIP = 5 (ms:46), HARD_SPEED = 45.0 (ms:47), HARD_ACCEL = 200.0 (ms:48), OFFSET_CLAMP = 8.0 (ms:49). mp:48 HARD_MAX_SPEED = 50 is a different ceiling; the server uses its own 45.
- Pacing: send_for(goal, s) sends round(s / pl.dt) setpoints (ms:84-85). paced() sends one setpoint per pl.dt (ms:109 advances the clock) and stops on e-stop (ms:104).
- Entry: move(q, speed, comp, keep_grip=False, accel=60.0, keep_offset=False) (ms:113). E-stop: status "estop", msg "refused: estop present" (ms:114-116). validate_target(q) (ms:120).
- Transit rule: comp False and keep_offset False zero the offset (ms:122-123). The goal is g = q + offset on the five body joints (ms:125; g[GRIP] not offset, ms:143-144). validate_target(g) with offset reset on failure (ms:126-130).
- Goal ramp: ok = paced(plan_ramp(self.goal, g, speed, pl.dt, max_accel = min(accel, HARD_ACCEL))) (ms:133). plan_ramp = mp:158-198 (smoothstep ramp from the last goal to g).
- Settle: ok = ok and send_for(g, 0.5) (ms:134).
- Compensation loop (ms:135-150): its = 0 (ms:135); while ok and comp and its < 10 (ms:136). cur = pl.current(); err = q - cur; err[GRIP] = 0 (ms:137-139). Stop when max |err[:5]| <= 0.5 (ms:140-141). offset[:5] = clip(offset[:5] + 0.7 * err[:5], -8, 8) (ms:142). g = q + offset, validated (ms:143-148). ok = send_for(g, 0.35) (ms:149). its += 1 (ms:150).
- Finish: ACTIVE removed in finally (ms:151-153); self.target = q (ms:154); status "ok" if ok (ms:155-156); reply msg = "comp_iters=%d" % (its if comp else 0) (ms:157).
- Reply carriage: msg (state.json) carries comp_iters after every move. server.log carries the full state JSON (ms:168). The loop log carries comp_iters only inside STOP lines after an e-stop ended a move: R/logs/sort/any_1740/loop.log has 49 such lines, e.g. "17:46:11Z ep 4: STOP move_rel [0, 0, 45.0] -> estop comp_iters=0; recovering" (also R/logs/sort_any_any_1740.out).
- Defaults at the call sites: server dispatch comp default True for move and rel (ms:205, 222); grip comp False with keep_offset (ms:226). Loop: ga.move comp=True (ga:77); pa.move_rel comp=True (pa:109); sa.tmove comp=False (sa:420) but go_scan and station_inspect pass comp=True (sa:398, 403 and the go_scan call).
- Pose-dependent offset: the offset is kept across compensated moves (ms:122-123) and resets only on uncompensated transit moves.
- Ramp and loop timing constants that S3 must match: pl.dt from MotionPlayer (mp:410-444; rate default DEFAULT_RATE_HZ = 30.0 at mp:50; dt derivation to confirm in S3); mp:47 DEFAULT_MAX_SPEED = 25; mp:49 DEFAULT_MAX_ACCEL = 50; mp:51 DEFAULT_TOLERANCE = 3.0; mp:52 DEFAULT_SETTLE_S = 0.5 (the server passes tolerance=3.0 at ms:182).
- Driver methods the server calls (S3 must emulate): pl.current() (mp:445), pl.hold() (mp:449), pl.dt, pl.driver.send_action, drv.set_torque (ms:243), drv._connection.port_handler.closePort (ms:248).
- Import hazard: mp imports numpy, yaml, loguru and physicalai_so101_plugin.constants at module level (mp:33-36). The sim server copies the pure functions it needs (validate_target mp:134-157, plan_ramp mp:158-198) rather than importing mp, unless sim-env has those packages (check in S3).

## 3. Camera frames (how ga.fresh / ga.snap get them)
- Live frames: R/run/live_top.jpg, R/run/live_close.jpg, R/run/live_wrist.jpg. Written by ffmpeg with a 640x480 input (monitor_cams.sh:11, 15; wrist_rec.sh:12) at 4 fps via `-vf fps=4 -update 1 -y` (monitor_cams.sh:13, 17; wrist_rec.sh:13). ffmpeg writes the file in place (not atomic).
- ga.fresh(view, after) (ga:84-95): path R/run/live_{view}.jpg (ga:85). Polls up to 60 x 0.05 s for getmtime > after (ga:86-89), then cv2.imread (BGR). Falls back to a plain imread (ga:95).
- ga.snap(tag) (ga:98-104): t = now + 0.3 (ga:99); reads top, close and wrist via fresh; writes OUT/{tag}_{view}.jpg (ga:103). OUT = episode folder (sa sets ga.OUT in cycle, sa:476-).
- pa.frame() = ga.fresh("wrist", now + 0.25) (pa:101-102). Used by sa:217, 255, 267, 276, 351, 361 through pa.blobs / pa.find / pa.servo.
- sa:352: ga.fresh("close", now + 0.2) (pre-grasp close view). sa:362-377: ga.snap calls (stage1, grasp_open, closed, lifted). sa:405: ga.snap("station").
- sl:76 reads R/run/live_{view}.jpg with cv2.imread and no mtime test (hands only).
- Frame contract for a sim: a new JPEG with a new mtime about every 0.25 s, 640x480, decodable by cv2.imread.

## 4. Hands and the clear check
- sl.hands_seen(sec) (sl:54-80): reads the tail of logs/monitor/hand_watch.log (sl:57); reads R/run/mserver/state.json target[5] (sl:72); reads R/run/live_top.jpg and live_close.jpg (sl:76-77) and calls handdet.is_hand (sl:70, 77).
- sl.wait_clear(reason) (sl:82-93): stops on R/run/sort_stop (sl:85); removes R/run/estop after no hands for 3 s (sl:88-89).
- sa uses from sl only: hands_seen (sa:213), wait_clear (sa:609, 684), close_verdict (sa:406), decide (sa:545, 595), save_crop (sa:357, 408), mask_jaws (sa:308). sa never calls ga.depth (RealSense path) or sl.pick.
- hw (hand_watch.py, the separate watchdog; not started in sim): reads live frames (hw:69, 77) and state (hw:33, 37-41); motion_active (hw:64); writes R/run/estop (hw:89); log R/logs/monitor/hand_watch.log (hw:61); pkill -INT -f run_sequence.py (hw:90); estop frame R/logs/monitor/estop_*.jpg (hw:91).
- handdet.py (pure cv2 and numpy): ROI (handdet:27), AREA_ALONE and AREA_EDGE (handdet:29-30), PERSIST (handdet:31), CLOSE_VALUE_FLOOR (handdet:32), is_hand (handdet:98).

## 5. Calibration and static inputs (read only by the loop)
| path | read at | used for |
|---|---|---|
| R/app/tools/scan_calib.json | pa:28 (SCALF), pa:203-204 load_scan; sa:47-51 scan_cal() | q_scan, target_scan, brick size at scan |
| R/app/tools/scan_pairs.jsonl | sa:64 (SCAN_PAIRS); read sa:85-86 | learned scan map; APPENDED at sa:131 (section 8) |
| R/app/tools/wrist_calib.json | kin:27 (CALF); kin:54-57 load_calib | wrist Jacobian M_tool (kin.m_base, kin:60) |
| R/app/tools/jaw_calib.json | ga:31 | TARGET_PX (jaw centre in wrist view); fallback (392, 268) at ga:30-32 |
| R/app/tools/wrist_jaw_mask.png | ga:44-45 | JAW_MASK (static: the wrist camera rides on the gripper); applied inside pa.blobs (pa:46-91) |
| R/app/tools/close_jaw_mask_hover.png | sl:134 | CLOSE_MASK_HOVER (hover close view; legacy path, sa uses mask_jaws) |
| R/app/motion/project_robots.json | kin:22 (CAL = FollowerCalibration.from_studio_json); ms:178 | normalized <-> radian map (kin.fk kin:30-33 uses CAL.rad_from_norm) |
| R/repos/edge-developer-kit-reference-scripts/samples/ai/edge-ai-demo-studio/workers/services/robotics-ai/platforms/so101/urdf/so101.urdf | kin:19-20 | FK and IK chain (pick_v2.kinematics.Chain, imported kin:16) |
| R/app/motion/gm/spot_fast.yaml (JSON content) | sa:427 (WP); sl:31 and sl:188 (run_fast) | pose table for place() (sa:432-447) |
| R/app/anomaly/det2_f2/models/tape_thresholds.json | inspector:65 THR_PATH via inspector:26 HERE; load_thresholds inspector:457-461 | DET2 tape thresholds |
| R/app/pick_v2/kinematics.py (import kin:15-16); R/app/anomaly/det2_f2/inspector.py (import sa:29, sl:25) | at import | code (numpy, cv2, json, xml only; no hardware) |

## 6. Outputs written by the loop (real cell paths)
- R/run/sort_stop removed at start (sa:677-678; sl:426-428 legacy). R/run/sort_pause written by periodic_jaw_check on a failed check (sa:466).
- R/logs/sort/<RUN> with RUN = R/logs/sort/<run name> (sa:668): loop.log (sa:40-44, log()), ep_NNNN/ (cycle, sa:476-), ep_NNNN/crops/, episode.json (save_ep sa:563-577), cycles.jsonl (save_ep), joints.csv (motion server, from sa:493), snapshots OUT/{tag}_{view}.jpg (ga.snap ga:98-104), crops (sa:269, 351, 356, 407-408, sl legacy).
- episode.json field "cams": the newest top_*.mkv and close_*.mkv listed from R/logs/monitor (sa:567-570).
- Motion server: R/run/mserver/server.log (ms:53-55), R/run/mserver/stdout.log (resume.sh:42, not the server).
- jaw check: R/logs/depth/servo/jaw_check.jpg (jaw_check.py:82, absolute path).
- Scan pairs: APPENDED to R/app/tools/scan_pairs.jsonl (sa:131).

## 7. Hardware touch points (real cell only; the sim must never reach them)
- Follower serial port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D040939-if00: ms:45 (opened only by the motion server); resume.sh:9.
- Cameras: /dev/video8 (top), /dev/video14 (close), /dev/video16 (wrist): monitor_cams.sh:11, 15; wrist_rec.sh:12; resume.sh:23-25.
- RealSense: ga:147 runs rs_grab.py inside ga.depth() (ga:144-160). sa does not call ga.depth; sl.pick (legacy) does.
- Studio: resume.sh:19 (studio_sessions.py). Process kill: hw:90 (pkill -INT -f run_sequence.py).
- Process control: resume.sh and the pid files (run/*.pid). The sim does not use them.

## 8. Items the S2 patch must change (found in S1)
(a) sys.path literals (sa:23-24, sl:22-23, ga:21, pa:18, kin:15, jaw_check.py:16): point them at the code directory of the file, so the copy imports its own modules. Otherwise the copy silently imports R/app/tools.
(b) R/run paths -> RUN_ROOT. Single variable: JUNCTION_RUN_ROOT, default R/run (today's path). Sites: ga:27 (SD), ga:85 (live), sa:211, 213, 466, 677-678, 683, sl:72, 76, 85, 88-89, 99, 101, jaw_check.py: none. Real cell unchanged when the variable is unset.
(c) R/logs paths -> LOG_ROOT. Default R/logs. With the variable set: <JUNCTION_RUN_ROOT>/logs. Sites: sa:567-569 (monitor listing), sa:668 (RUN), sl:57 (hand_watch.log), jaw_check.py:82 (absolute path, writes a JPEG). Legacy sl main (sl:420, 469-471) and ga main (ga:230) are not executed by the loop but use the same literal; patch them too for safety.
(d) SCAN_PAIRS (sa:64) appends to R/app/tools/scan_pairs.jsonl (sa:131): a write into R/app. Point it at RUN_ROOT/scan_pairs.jsonl. The file starts empty in sim (the sim geometry differs from the real scan calibration). fit_scan_corr already tolerates a missing file (sa:85).
(e) jaw_check.py: T = R/app/tools (line 19) and lines 92-93 write wrist_jaw_mask.png and jaw_calib.json into T (the calibration branch). The loop calls it with --no-move (sa:456, 680) so that branch should not run, but the copy must not be able to write into R/app: point T at the copy directory.
(f) save_ep (sa:567-570): [-1] on an empty list raises IndexError when no MKVs exist (the sim writes none). save_ep is called outside the try in run_one (sa:625-655), so the loop would crash after cycle 1. Patch: tolerate empty lists.
(g) Motion server import (mp:33-36): only ms imports mp. The sim server must not import mp unless sim-env provides numpy, yaml, loguru and physicalai_so101_plugin (S3 check).

## 9. Sim contract (what S3 and S4 must provide)
- RUN_ROOT/mserver/cmd.jsonl (read), RUN_ROOT/mserver/state.json (atomic write), RUN_ROOT/mserver/pid, RUN_ROOT/mserver/server.log.
- Ops: move, grip, state, rec_start, rec_stop (used by the loop); rel, exit, release (for parity). Reply fields as in section 2. Compensation and ramp as in section 2b.
- RUN_ROOT/live_top.jpg, RUN_ROOT/live_close.jpg, RUN_ROOT/live_wrist.jpg: 640x480 JPEG, new mtime about every 0.25 s, written atomically (sim choice; the real ffmpeg writes in place).
- RUN_ROOT/estop (honoured), RUN_ROOT/motion_active (touched while a move runs), RUN_ROOT/sort_stop, RUN_ROOT/sort_pause.
- LOG_ROOT/sort/<run>/ (loop output); LOG_ROOT/monitor/ must exist (empty is allowed only after patch (f)).
- Units: normalized, same mapping as kin.CAL (project_robots.json).

## 10. Timing reference (real cell)
- docs/PICK-ANYWHERE.md: a typical clean cycle takes about 65-70 s (pick 30-48 s, place 20 s). Real reference episode: R/logs/sort/any_1740/ep_0017 (episode.json and joints.csv, 189362 bytes). Its timings are read in S5.

## 11. Dynamics inputs for S3 (P3-TWIN-DYN)
- Published parameters: R/sim/twin/dynamics/twin_dynamics_v1.json (stage twin-dyn-v1, utc 2026-11-14T19:50:19Z). Per joint: kp, kv, frictionloss, armature, forcerange = 10.0, delay (0.07-0.11 s). k_rad_per_norm_unit: 0.0169-0.0293 rad per normalized unit.
- Control law (that file): force = clip(kp*(ctrl - q) - kv*qdot, -forcerange, forcerange); ctrl = k * goal(t - delay). timestep 0.004 s; integrator implicitfast. Model file: so101_dyn_v1.xml in the same folder.
- Equality check: R/work/P3-TWIN-DYN/E1/full_result.json (utc 2026-11-14T19:50:19Z) params_6x6 rows equal the published per-joint kp, kv, frictionloss, armature, forcerange and delay (all six rows checked by eye). S3 uses the published file.
- In-sample RMSE in normalized units (both files): pan 0.271, lift 0.819, elbow 0.826, wrist_flex 0.323, wrist_roll 0.179, gripper 0.242.
- NOT VERIFIED: the no-dynamics steady-state errors of about 1.9 (shoulder_lift) and 2.2 (elbow) in the coordinator note are not in either file. The in-sample RMSE values above are a different quantity. The source of the 1.9 and 2.2 figures is requested from the coordinator. S3 measures the no-dynamics and dynamics steady-state errors in the sim itself and reports both.
