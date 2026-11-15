# Studio "Run model" live test (GD-22): procedure

Prepared for Opus. Not executed. E1 (VLA-PREP) read the Studio clone (/home/claude/kb/repos/physical-ai-studio, HEAD 285924d9, read only) and the laptop state at 21:41-21:51Z (R = /home/intel/intel-junction-2026). E1 did not move the robot, open cameras or serial ports, touch the motion server, the Studio UI (:3000) or API (:7860), R/run, or the Studio data folder.

Citations are repo-relative paths with line numbers, or laptop paths under R.

## 1. State at 21:50Z (what the test would meet)

- Studio UI listens on 127.0.0.1:3000 (pid 1629977). Nothing listens on 127.0.0.1:7860, the base URL the loop uses (shared/contracts/LOOP-CONTRACT.md:5). The team has to start the backend first.
- Sort loop run any_2127 is active (logs/sort_any_any_2127.out, last servo line 21:51:05Z). run/sort_any.pid names pid 399110, which was not alive at 21:48Z. Check the process list before stopping anything.
- Holders: motion server pid 3522921 on the follower port (now /dev/ttyACM0); hand_watch pid 3522883; monitor_cams pid 3521843 on /dev/video8 and /dev/video14; wrist_recorder pid 3521845 on /dev/video16.
- No PAUSE.txt in R; no run/sort_stop or run/sort_pause.
- Build: shared/studio-api/system_info.json reports studio_version 0.3.0, git_sha 17714488 (dirty). The clone is 285924d9 (clean). Check each route below against the running backend before use.

## 2. Blockers (any one stops the test)

1. The API on 7860 is down (section 1).
2. The follower entry points at the leader. shared/studio-api/project_robots.json (11:39Z snapshot) lists SO101 Follower with connection_string /dev/ttyACM1 and SO101 Leader with /dev/ttyACM2. At 21:41Z the follower's by-id device (5B3D040939) is /dev/ttyACM0, and the leader's (5B90148378) is /dev/ttyACM1. Studio would drive the leader arm. Stop unless the follower entry resolves to the follower device. Changing it is a Studio data change, so Opus decides.
3. The saved environment carries the leader. shared/studio-api/project_environments.json has tele_operator robot d361855b (SO101 Leader). The UI copies that into the load (application/ui/src/features/runtime-session/runtime-session-provider.tsx:67-70), and the backend connects a set tele-operator robot (application/backend/src/runtime_session/environment.py:106). Use option A in section 4 unless Opus approves option B.
4. Our holders must be released first (section 3).
5. Device. Pass GPU explicitly. Studio's default 'auto' resolves to CPU (ledger/DECISIONS.md:112, D-080).

## 3. Stop our stack, in this order

Sources: docs/PICK-ANYWHERE.md, section "Run, stop, pause"; app/tools/resume.sh.

1. `touch run/sort_stop`. The loop removes the file itself. Wait for the last log line and confirm no sort_any.py process remains.
2. Hand back: `mseq.py ../motion/gm/servo.yaml spot_up_home_closed`, then `mcmd.py '{"op":"release"}'`. The first command moves the arm home, so Hassan must be present.
3. Kill by pid file: hand_watch (run/hand_watch.pid), monitor_cams (run/monitor_cams.pid), wrist_recorder (run/wrist_recorder.pid).
4. Stop the motion server by run/mserver/pid. resume.sh step 5 starts it with that file. It holds the follower port.
5. Check that nothing holds the follower device or /dev/video8, /dev/video14, /dev/video16. This is the check resume.sh steps 1 and 2 make.

## 4. Studio side

Option A (leader-free, preferred for a short test): the loop's own draft. app/loop/orchestrator.py:153-157 builds the environment with the follower and tele_operator none. Do not use the client helper environment_draft (candidates/code/loop_v2/app/loop/studio_runtime_client.py:182-196) for this test, because it copies the saved tele_operator (line 191).

Option B (UI route, needs Opus approval because the saved environment carries the leader):

1. Run model button: application/ui/src/features/models/models-table/model-row.tsx:97.
2. Start dialog: start-inference-dialog.tsx:34 (device list from /api/system/devices/inference), :40 (default device), :49-55 (opens the inference page with the device in the URL). Choose GPU.
3. Inference page: application/ui/src/features/models/inference/inference-page.tsx:23-37 and :80-106.
4. Inference mode panel, application/ui/src/features/runtime-session/workspace/inference-mode.tsx: :29 (task defaults to the dataset's default_task), :37 (execution mode sync), :38 (request threshold 0.5), :76 (task is read-only while running), :88 (action rate), :111 (Stop), :115 (Play).

Commands and payloads (the client sends the same commands as the UI):

- Socket: ws://127.0.0.1:7860/api/projects/{project_id}/runtime/ws?hz=20. Router prefix application/backend/src/api/runtime_ws.py:51, route :86, observation rate :98 (le 60, default 20). The UI opens this path at runtime-session-provider.tsx:135.
- Load environment: {"command":"load_environment", ...draft...} (studio_runtime_client.py:317).
- Load model: {"command":"load_model","model_id":"<id>","backend":"openvino","device":"GPU"} (client :340).
- Start: {"command":"set_control","control":{"type":"model","task":"<task>","hz":30,"execution":{"mode":"sync","request_threshold":0.5},"grace_s":3.0}} (client :221-225 and :372; bounds in application/backend/src/schemas/runtime_session.py: hz le 500 at :60, grace_s le 30 at :62-65).
- Hold: {"command":"set_control","control":null} (client :379; inference-mode.tsx:55; session-workspace.tsx:83).

Run limits: one run, at most 30 s. The ledger cleared a 30 s test for f06a253e only (ledger/DECISIONS.md:116, D-083). It gives no clearance for 389da0fd. The offline dry run (work/VLA-PREP/E1/dryrun.json) found 389da0fd's actions not sane, so do not run it on the arm without Opus's decision.

Hands and stop: check the Studio stream for hands before Play. One person keeps Stop in reach. The hands-in-scene hook in the sandbox live runner is a stub (work/P2-B-OVRUN/E1/code/runner.py:95-97), so this check is manual.

Stop sequence: Stop (inference-mode.tsx:111) sends hold. Then POST /api/runtime/sessions/{session_id}/stop (application/backend/src/api/runtime_sessions.py:57; ids from GET at :20). Close the socket, which unloads the session (verified runtime map: /home/claude/kb/final/T19-PAS-BE-RT.md). The robot worker allows 40 s for stop (application/backend/src/workers/robot_worker.py:108). Wait for it before any hardware action.

Evidence: UTC times of Play and Stop; screenshots of the Studio stream and status; Studio log lines for the window; model id, task, device and settings. The monitor MKVs cannot cover the run, because Studio holds the cameras.

## 5. Restore

1. Confirm the Studio session is closed and nothing holds the follower port or the cameras.
2. Run app/tools/resume.sh with no flag. It refuses if a Studio session mentions the follower (step 1), checks the cameras, starts the recorders, hand_watch and the motion server (torque on at the present pose), and moves nothing (header lines 2-6; steps 1-5).
3. Only when Hassan says the robot is ours again, run `app/tools/resume.sh --go <run name>` (ledger/DECISIONS.md:129, D-096; docs/PICK-ANYWHERE.md).

## 6. Not verified

- Routes and bounds in the running build (the checks above used the clone only).
- Whether the UI start dialog defaults to CPU (D-080 says 'auto' resolves to CPU).
- Studio speed limits: none found in the files read. The sandbox runner caps each tick at 25/30 units (work/P2-B-OVRUN/E1/code/runner.py:35).
- The action-rate range shown in the UI.
