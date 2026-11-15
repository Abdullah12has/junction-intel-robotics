# Junction 2026 sorting cell: submission README

Status at 2026-11-14 15:11Z (laptop clock 17:11 EET, UTC+2), revised after the 15:25Z decision corrections (D-070, D-080): build in progress. No result in this pack is a take. Every take result is TBD and is listed in evidence.md with the file that will supply it. One benchmark run (run B) is measured and is shown as a benchmark, not as a take result.

Source keys (inline cites use these keys; the coverage checklist says which files were opened in this run and which in an earlier run):
- G: R/repos/physical-ai-studio/JUNCTION2026.md (laptop copy, 140 lines, sha256 prefix 566a5cd7, same as the KB raw copy).
- T08: /home/claude/kb/candidates/T08-L-JUNCTION.md (KB raw copy of the guide, grep line numbers).
- DEC: /home/claude/kb/ledger/DECISIONS.md (D-nnn or OD-nn).
- INS: /home/claude/kb/ledger/INSTRUCTIONS.md (I-nn).
- LC: /home/claude/kb/shared/contracts/LOOP-CONTRACT.md (section).
- FL: /home/claude/kb/shared/FACTS-LIVE.md (time stamp).
- PLAN: /home/claude/kb/candidates/P2-P-PLAN.md (section).
- PROBE: R/logs/probe_handover_opus.log.
- GMB / GMG: R/logs/gm/blue_round_trip_142907.log / R/logs/gm/green_round_trip_143045.log. Their own timestamps carry +02:00; this pack gives UTC.
- CUR: R/data/curated/episodes_v1.csv.
- BENCH: R/work/P2-B-OPT/E1/bench_runB_quiet.json and bench_summary_runB.txt (run B: started 14:57:53Z, finished 15:02:36Z). R/work/P2-B-OPT/E1/bench.json is being rewritten by a later run and is not cited.
- BSRC: R/app/telemetry/bench_ov.py (writes bench.json; 20 warm-up and 200 timed iterations).
- EV: R/app/loop_v2/app/loop/events.py (event names and control commands).
- RULES: R/app/loop_v2/app/loop/policy_rules.yaml (modified 15:03Z).
- PLIVE: R/app/loop_v2/app/loop/policy_live.py (live refusal checks).
- PRUN: R/app/loop_v2/app/loop/policy_run.py (command set).
- ARM: R/run/arm_state.
- TEL: R/logs/telemetry/20261114.jsonl and R/app/telemetry/telemetry.py.
- MODELS: /home/claude/kb/shared/curation/models.tsv.
- RAS: /home/claude/kb/repos/robotics-ai-suite/src/components/object-detection/workshop/.workshop/object-detection.yaml (RAS repo copy; benchmark recipe at lines 38-47). Both RAS repo copies were searched for InferenceModel: 0 matches.
- RPL: R/work/P2-B-OVRUN/E1/results/replay_CPU.json, replay_GPU.json and replay_summary.json (the D-080 replay; read 15:27Z to 15:31Z).
- R = /home/intel/intel-junction-2026 (laptop root).

## 1. What the cell does

One brick at a time is taken from a cluster of DUPLO-size bricks on a black mat and sorted by what it looks like [FL 13:45Z; FL 11:30Z]. The routing rule is final [DEC OD-10]:
- GOOD (clean 2x2) goes to the BLUE plate, which is on the left of the top camera view [FL 13:52Z].
- DEFECTIVE goes to the GREEN plate, on the right. A brick is defective if it has tape on it or is not a 2x2, for example a 2x4 [DEC OD-10].
- UNCERTAIN goes to GREEN, the safe bin [DEC OD-10; LC section 8].
- A brick seen as CONFIDENT_DEFECT goes straight to GREEN with no inspection step [INS I-52; LC section 8].

Each cycle runs SCAN, TRIAGE, INSPECT (only when triage asks for it), DECISION, ACT, VERIFY, then back to HOME [LC section 8]. architecture.md has the diagrams.

## 2. Pipeline mapping (PERCEIVE, DETECT, REASON, EXECUTE, OPTIMIZE)

| Stage | Component | Source | State at 15:11Z |
|---|---|---|---|
| PERCEIVE | Top RealSense (dataset key `realsense 243622061421`) and close low RealSense (`realsense 243722064316`), read from the Studio runtime stream | LC section 1; FL 13:23Z; RULES studio block | All four Studio cameras reported Online at 12:00Z [FL 12:00Z] (earlier check, not re-read) |
| PERCEIVE | Wrist camera USB2.0_CAM1 (/dev/video16, 640x480 at 30 fps), continuous recorder | app/tools/wrist_rec.sh; DEC D-062 | Recorder pid file present (R/run/wrist_recorder.pid, written 14:33Z). The loop does not check that pid (section 8, item 5) |
| DETECT | Per-brick scan: shape from the top view, tape from the close view | app/anomaly/det2_e2/scan_once.py (args --top, --close, --out); P2-B-DET2 spec | Two builds on disk (det2_e2, det2_e3); no real-data verdict yet (TBD) |
| DETECT | PatchCore (Anomalib 2.7.0), FP16 static IR, GPU first, NPU time-boxed | DEC D-033, D-045; app/anomaly/static_ir.py | Candidate. Run B measured the IR as built: native batch 2, FP16 weights [BENCH]. Batch 1 per brick is TBD. Real-data AUROC and threshold TBD |
| REASON | Triage (I-52) and fail-safe routing (OD-10) | LC section 8; app/loop_v2/app/loop/triage.py; RULES triage block | Module on disk; not run live. uncertain_margin 0.1 is marked [unclear] in RULES |
| EXECUTE | Learned pick-and-place through the Studio runtime, one task text per plate | DEC OD-12 (D-060); LC sections 1 and 6; RULES plates block | Handover PASS on the probe [PROBE; DEC D-057]. Policy only at smoke level [MODELS]. Both plate model ids are PLACEHOLDER in RULES. Device: GPU set explicitly by D-080; RULES plates still say auto (section 8, item 14) |
| EXECUTE | Scripted sort (Plan S) and pick-anywhere | app/motion (F1, DEC D-046); app/pick_v1, app/pick_v3 | Gate G-M PASS, two round trips [GMB; GMG; DEC D-063]; pick not validated live |
| OPTIMIZE | Per-device benchmark: 20 warm-up and 200 timed iterations, explicit devices | BSRC; BENCH | Run B complete: 12 rows (4 model variants on CPU, GPU and NPU), all status ok [BENCH]. Not precision-matched; not a take result |
| OPTIMIZE | 1 Hz telemetry sampler, no root | app/telemetry/telemetry.py (PERIOD_S = 1.0); telemetry_ctl.sh | Running; R/logs/telemetry/20261114.jsonl written at 15:10:54Z [TEL] |
| Operator | Dashboard: live preview, verdicts, counts, STOP | P2-B-DASH spec | Not on disk (app/dashboard absent, checked 15:08Z) |

## 3. How to run

Interpreter for the loop, dashboard and scan: /home/intel/intel-junction-2026/envs/anomalib-env/bin/python (env name from the LC header; interpreter path not run in this pack).

### 3.1 Preflight, in this order
1. Studio backend on 127.0.0.1:7860 and UI on 127.0.0.1:3000 are up (ports from [FL 11:40Z]; check before a run). None of the commands below changes Studio files.
2. Start the telemetry sampler: /home/intel/intel-junction-2026/app/telemetry/telemetry_ctl.sh start. Check it with telemetry_ctl.sh status [TEL].
3. Stop the wrist recorder before any loop run, because the loop's wrist camera owns /dev/video16: kill $(cat /home/intel/intel-junction-2026/run/wrist_recorder.pid) [wrist_rec.sh header; PLAN R12]. This stop is manual: the loop package does not check the pid (section 8, item 5).
4. Check the arm state: R/run/arm_state must say FREE, written by the arm watch every 15 s [FL 13:55Z; ARM]. At 15:08:09Z it read "BUSY 1436289". That pid is a Studio backend publisher worker (process name physicalai.capture.transport._publisher_worker), not part of this pack, and nothing here stops it. No motion and no live start is allowed while the state is not FREE.
5. The follower is reached only by its by-id path, /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D040939-if00, never a ttyACM number [FL 13:42Z; RULES paths block]. Both G-M logs record that the export connection string /dev/ttyACM1 was ignored [GMB line 1; GMG line 1].
6. Live runs only: the presence flag file R/run/HASSAN_PRESENT must exist. The live command refuses without it [PRUN; PLIVE; LC section 4; DEC D-052].

### 3.2 Commands (status at 15:11Z; none of these was run in this pack)
- The loop command set is defined in R/app/loop_v2/app/loop/policy_run.py [PRUN]: `scenarios` (list the dry-run scenarios), `dry SCENARIO LOGS_ROOT` (one dry-run policy cycle with fakes; no Studio, camera, port or robot) and `live RULES` (live policy run; refused unless the presence flag exists and GO is typed). The runner named in that file's docstring, tools/run_policy.py, is not on disk, and the file has no __main__ entry, so no command runs as written. The runner path is TBD.
- Live run: `live RULES` asks for GO and stops before any device opens unless every input exists [PRUN; PLIVE]. The rules file (modified 15:03Z) has PLACEHOLDER plate model ids, det2.module_dir null and still.exposure null, so a live start is refused now [RULES]. The same file sets device auto in the BLUE and GREEN plate blocks (lines 32 and 41), which D-080 rules out. Set both to GPU before any live start (section 8, item 14).
- Dashboard: spec only (P2-B-DASH). The spec's run folder, logs/loop, conflicts with the rules' logs_root (section 8, item 4). Not on disk.
- Stop a run: the operator control file control.json takes the commands start, pause, resume, next, stop and quit [EV]. The dashboard that writes it is not on disk.
- Scan one frame pair: /home/intel/intel-junction-2026/envs/anomalib-env/bin/python /home/intel/intel-junction-2026/app/anomaly/det2_e2/scan_once.py --top TOP_IMAGE --close CLOSE_IMAGE --out OUT_DIR. Arguments verified on disk in an earlier run; not run in this pack (no camera opened).
- Telemetry: telemetry_ctl.sh start, status, stop [verified on disk in an earlier run].
- Benchmark: app/telemetry/bench_ov.py takes no command-line arguments (the file has no argparse and no sys.argv, checked 15:08Z). It writes R/work/P2-B-OPT/E1/bench.json [BSRC]. Interpreter TBD.

### 3.3 Before the take
See recording-checklist.md for the scene, cameras, rehearsals and fallbacks.

## 4. Safety rules in force

- Live gating: `live` needs the typed GO and the presence flag [PRUN; PLIVE]. policy_live also refuses while a plate model_id is a placeholder, det2.module_dir is null, or still.exposure is null [PLIVE; RULES].
- The loop is specified to refuse while the wrist recorder pid is alive (P2-B-LOOP spec). The code does not check it (section 8, item 5).
- Motion is refused when a hand is seen in either view (P2-B-LOOP2 section 3, spec). A camera check for hands and people comes before every motion run [INS I-44].
- Speed: the G-M round trips ran at 15 units/s. The arrival tolerance was raised from 3 to 6 after that run [DEC D-063]. In both logged runs all 23 steps arrived [GMB; GMG].
- Torque is off at the end of a run [LAPTOP.md, robot motion rule 5]. If an error happens with a brick held, the torque is held, the run stops and reports; torque is never dropped mid-air [LC section 7].
- Any error or timeout goes to HOLD, then STOPPED [LC section 4].
- Fallback runner app/ovrun/runner.py, behind the Studio runtime path (D-080): its timeout must not drop torque. On timeout it holds, stops the policy and returns home by script [DEC D-080].
- Home gripper value is about 25, never 0 [P2-B-LOOP2 section 3].
- A physical cut-off within reach is required at every arm window [PLAN R11]. No hardware e-stop is documented in the inputs (section 8, item 12).
- Plan S refuses to open a port that Studio or another process holds [P2-R6-MOTION auditor rules; PLAN section 1]. The handover waits until nothing holds the follower [LC section 7]. At 15:08Z the arm state reads BUSY (section 3.1, step 4).

## 5. Data policy

- Only the team's recorded Studio datasets are used, and every episode is verified by us before use: task text, destination, brick held, scene [INS I-54].
- The team's models, code, environments, docs and plans are not used [INS I-54; DEC D-066]. The team model fa42ee76 was loaded once, as a probe test [PROBE 14:13:30Z; DEC D-066].
- Simulation data is not used for training, detection or calibration [DEC D-055; DEC D-069]. It is kept as a geometry cross-check only [DEC D-069].
- Episode sets, as recorded: v1 (CUR, 14:51Z) keeps 3 of 16 episodes of dataset 6d6bdce3: episodes 11 and 12 to BLUE, episode 15 to GREEN. The other 13 are dropped with reasons [CUR]. Convention v2 (DEC D-067, 14:50Z) also admits episode 7 (BLUE) and episode 14 (GREEN), and adds the S1 put-downs 3 to 6 to the MT-ALL set with "on the mat" wording. The v2 CSV is not on disk yet (section 8, item 2).
- Labels come from appearance only, never from where an episode dropped the brick or from its task text [DEC OD-10].

## 6. Models

| Model | Type and backend | Trained on | State |
|---|---|---|---|
| 389da0fd-e840-4224-b0a0-09aa5be7a8fc, "S1 ACT curated v0 (Opus)" | ACT, OpenVINO export, 2056 steps (8 epochs). Benchmarked as two IR weight variants: as exported (FP32 weights) and compressed to FP16 [BSRC] | S1 curated v0, dataset a4ba6ede-092c-450f-8c76-559f66991dbd, 4 episodes [MODELS; FL 14:08Z] | Smoke run only: TRAIN-OK says v0 is a 2k-step pipeline smoke run [TRAIN-OK] |
| Per-plate or text-conditioned models for the v2 sets | TBD | P2-D-CUR1 | Not in models.tsv (last update 14:01Z); RULES plate model ids are PLACEHOLDER |
| PatchCore detector (Anomalib 2.7.0), FP16 static IR | OpenVINO IR, GPU first; benchmarked as native batch 2 with FP16 weights [BSRC] | DEC D-033, D-045 | Real-data model and threshold TBD (P2-B-DET2 report, not read) |
| PaDiM detector IR, batch 1 | OpenVINO IR; benchmarked only [BENCH] | Not stated in this pack's sources | Role in the cell TBD |

Studio supports ACT, SmolVLA and Pi0.5 [T08, guide text, grep line 200]. The handover probe loaded a model with backend openvino [PROBE]; the trained ACT was exported to OpenVINO [FL 14:08Z].

## 7. Results

All take result values are TBD. See evidence.md for each rubric line, the exact metric, the source file and the gate. The run B benchmark numbers are measured, but they are benchmark values, not take results (evidence.md, Rubric 4).

Policy device rule (D-080, recorded 15:25Z): the policy runs on the iGPU, with device GPU set explicitly on every load_model. D-080 states that Studio's default device 'auto' resolves to CPU; the code lines it cites were not found on disk in this pass. Replay of ACT 389da0fd through the OpenVINO PhysicalAI InferenceModel, 52 query ticks per device (latency.update_at_query_ticks): GPU p50 9.29 ms and p95 11.23 ms [RPL replay_GPU.json]; CPU p50 121.00 ms and p95 146.88 ms [RPL replay_CPU.json]. These are replay values, not take results. The same replay summary gives executed-action error 4.978 against hold-state error 1.556 (CPU entry, in-sample), so no policy success is claimed [RPL replay_summary.json].

## 8. Open items found in this run

1. The driver log prints role=leader for the port that our motion code opens as the follower (GMB line 2; GMG line 2). FL 13:42Z maps serial 5B3D040939 to the follower. The cause is not verified. Check before any motion run.
2. The episode sets disagree: CUR (v1, 14:51Z) drops episodes 7 and 14, while DEC D-067 (v2, 14:50Z) admits them. The v2 CSV does not exist yet.
3. No runnable loop entry point. policy_run.py defines the command set, but tools/run_policy.py, named in its docstring, is not on disk, and policy_run.py has no __main__ entry (checked 15:08Z). No orchestrator with a dry-run flag exists. The exact run commands stay TBD.
4. Run folder path conflict. LOOP-CONTRACT section 5 names R/logs/loop/<run_id>/ and the symlink R/logs/loop/current [LC section 5]. policy_rules.yaml sets logs_root to R/logs/loop_v2 [RULES]. The dashboard spec reads logs/loop. Neither folder exists (checked 15:08Z). One path must be chosen before the dashboard command is fixed.
5. The wrist recorder stop is not enforced in code. Nothing in app/loop_v2/app/loop refers to the recorder pid (checked 15:08Z). The spec (P2-B-LOOP) says the loop refuses while the pid is alive.
6. The arm state read BUSY at 15:08:09Z (pid 1436289, a Studio backend publisher worker) [ARM; ps check]. Motion stays blocked until the state reads FREE.
7. Two copies of studio_runtime_client.py exist: R/app/loop/ (with probe_handover.py) and R/app/loop_v2/app/loop/. The header of policy_live.py names app/loop/studio_runtime_client.py. Which copy the loop imports is not verified.
8. The benchmark rows are not precision-matched. The bench does not set a precision hint. It reads OpenVINO's INFERENCE_PRECISION_HINT after compile, which reports float32 on CPU and float16 on GPU and NPU in all 12 rows [BENCH; BSRC]. The machine was not quiet by the measured load: loadavg_1min_before_timed ranged from 3.18 to 7.82 [BENCH], although the file is named runB_quiet. bench.json is being rewritten by a later run (run_final.log: harness start 15:04:42Z) and must not be cited until that run has an exit line.
9. Package power cannot be read without root. The telemetry line at 15:08:34.768Z has "pkg_w": null [TEL]. BENCH says RAPL energy_uj is root-only and battery power is not system power on AC [BENCH power block]. Power per inference and per cycle are TBD.
10. Tape appearance is recorded two ways: DEC D-059 says clear or brownish tape on the yellow 2x2; DEC D-068 says dark grey gaffer-type tape. The scene set-up uses D-068. Confirm on the bench.
11. Robotics AI Suite (D-070, corrected 15:25Z). Our OpenVINO runner is not a Robotics AI Suite integration. The RAS repo copies have 0 matches for InferenceModel, and the guide links Studio exports to OpenVINO PhysicalAI InferenceModel at JUNCTION2026.md line 122 [G; checked in the KB repo copy]. The citable RAS item is the workshop benchmark recipe, object-detection.yaml lines 38-47 (benchmark_app with -hint latency on CPU, GPU and NPU) [RAS]. Run B follows that recipe's explicit-device and latency-hint method with our own script, not benchmark_app: bench_ov.py compiles with PERFORMANCE_HINT LATENCY at line 77 [BSRC]. The P2-R9-RAS report is not in this task's inputs.
12. No hardware e-stop is documented in the inputs. Only a physical cut-off within reach is required [PLAN R11].
13. The spec gave the guide path /home/intel/Documents/physical-ai-studio/JUNCTION2026.md. That folder is outside the allowed folder [LAPTOP.md; INS I-19], so the same file was read from R/repos/physical-ai-studio (sha256 prefix matches the KB copy). The guide has no submission, deadline, video or prize text (earlier grep: one unrelated line, G line 98).
14. RULES plates set device auto (BLUE line 32 and GREEN line 41 of policy_rules.yaml, read 15:27Z). D-080 requires device GPU on every load_model. Both must change before any live policy start. This pack does not edit the rules file or any code.

## Coverage checklist (internal, for review; remove before publishing)

- [x] G lines 129-140 (rubric table) read on the laptop copy (earlier run); grep for submission, deadline, video, readme, prize: 1 hit, unrelated (earlier run).
- [ ] /home/intel/Documents/physical-ai-studio/JUNCTION2026.md: not opened (outside the allowed folder); prefix-matched only.
- [x] DEC: all entries read in the KB copy (earlier run, D-001 to D-069, OD-1 to OD-12); D-063 and D-067 re-read at 15:10Z.
- [x] INS: I-01 to I-54 read (earlier run).
- [x] LC: sections 1 to 8 read (earlier run); the section 5 path line re-checked at 15:08Z.
- [x] FL: all entries read (earlier run, to 14:43Z).
- [x] PLAN: sections 1 to 8 read (earlier run).
- [x] Tasks read (earlier run): P2-B-LOOP, P2-B-LOOP2, P2-B-DASH, P2-B-OPT, P2-B-OVRUN, P2-B-WRIST, P2-B-PLANS, P2-B-DET2, P2-B-ANOM-REAL, P2-R8-PICK, P2-R6-MOTION, P2-R7-RUNTIME, P2-D-DATA, P2-D-CUR1, P2-T-TRAIN, P2-B-GM, P2-R9-RAS, P2-B-PROBE, P2-D-SIM2.
- [ ] Tasks not read: P2-B-ANOM-LAPTOP, P2-B-ANOM-TC, P2-D-BUILD, P2-D-DESIGN, P2-D-SIM, P2-HW-*, P2-R0 to P2-R5, P2-SAVE-*, T07-IMG-P2 (not needed for the pack's facts).
- [ ] Other workers' outputs (DET2 report, PICK offline_eval, OVRUN report, R9 report): not opened (isolation, not inputs).
- [x] Laptop (earlier run): root, app, envs, logs, run, data/curated/episodes_v1.csv, telemetry tail, telemetry.py (PERIOD_S), telemetry_ctl.sh, wrist_rec.sh, bench_ov.py, bench.json (running at that time), GMB, GMG, PROBE, scan_once.py argparse.
- [x] Laptop (this run, 15:08-15:11Z): bench_runB_quiet.json (all 1215 lines), bench_summary_runB.txt, bench_run_B.log, run_final.log, bench_ov.py (grep), events.py, policy_rules.yaml, policy_run.py, policy_live.py (grep), arm_state, ps for pid 1436289, G-M logs (both, 28 lines each), telemetry tail, logs/ and app/ directory listings.
- [x] Sandbox (this run): LAPTOP.md, the task file, the five drafts, STATUS.md, LOOP-CONTRACT (grep), P2-P-PLAN (grep), DECISIONS (grep for D-041, D-063, D-067).
- [x] Correction pass (15:25Z to 15:31Z), sandbox: DEC D-069, D-070 (corrected) and D-080 read in full in ledger/DECISIONS.md; RAS repo copies: object-detection.yaml lines 38-47 read, InferenceModel grep (0 matches in both copies), physical_ai_studio.md grepped for Studio lines; JUNCTION2026.md line 122 read (KB repo copy); the physicalai code lines cited by D-080 were searched for in the KB repos and the laptop E1 folder and not found.
- [x] Correction pass (15:25Z to 15:31Z), laptop (read-only): policy_rules.yaml lines 24-44 and the device lines; bench_ov.py grep (line 77); E1 results listing; replay_CPU.json and replay_GPU.json latency keys (lines 1476-1494); replay_summary.json CPU keys.
