DONE 2026-11-14T15:36Z P2-D-SUBMIT W1: five submission documents and this STATUS, revised for the coordinator correction (D-070 corrected; D-080 device rule recorded; Robotics AI Suite cited as the benchmark recipe). Sandbox and laptop copies are checked by SHA-256 in laptop-log.md section 4; every take value is TBD with its source file; blockers in section 4.
STARTED 2026-11-14T14:52Z P2-D-SUBMIT W1: reading protocol, guide, ledger, tasks

## 1. Deliverables

| File | Lines | SHA-256 prefix | Sandbox path | Laptop path |
|---|---|---|---|---|
| README.md | 149 | 2ced2d8f | /home/claude/kb/work/P2-D-SUBMIT/W1/README.md | /home/intel/intel-junction-2026/submission/README.md |
| architecture.md | 122 | 266cee7e | /home/claude/kb/work/P2-D-SUBMIT/W1/architecture.md | /home/intel/intel-junction-2026/submission/architecture.md |
| evidence.md | 138 | 9f12be50 | /home/claude/kb/work/P2-D-SUBMIT/W1/evidence.md | /home/intel/intel-junction-2026/submission/evidence.md |
| storyboard.md | 73 | 16deb7ea | /home/claude/kb/work/P2-D-SUBMIT/W1/storyboard.md | /home/intel/intel-junction-2026/submission/storyboard.md |
| recording-checklist.md | 107 | 7988e216 | /home/claude/kb/work/P2-D-SUBMIT/W1/recording-checklist.md | /home/intel/intel-junction-2026/submission/recording-checklist.md |

Also in the sandbox folder: STATUS.md (this file) and laptop-log.md (every laptop call in this run, with output). STATUS.md is copied to /home/intel/intel-junction-2026/work/P2-D-SUBMIT/W1/STATUS.md.

Hashes were compared on both sides after the correction pass (laptop start_process and sandbox sha256sum; outputs in laptop-log.md section 4). All five documents match, and STATUS.md matches its work-folder copy. The drafts contain no em dash, no en dash and no non-ASCII character (grep, 15:36Z).

## 2. R12 self-check (rows 1-31: laptop and sandbox re-reads, 15:08Z to 15:17Z; rows 32-40: correction pass, 15:25Z to 15:31Z)

Forty checks are listed. Thirty-nine are PASS and one is WITHDRAWN.

| # | Fact checked | Source | Method | Result |
|---|---|---|---|---|
| 1 | Run B summary has 12 data rows: 4 model variants on CPU, GPU and NPU, all status ok | bench_summary_runB.txt | read_file | PASS |
| 2 | Run B started 14:57:53.823Z and finished 15:02:36.071Z; OpenVINO 2026.4.1; warm-up 20, timed 200 | bench_runB_quiet.json | read_file, 2 parts | PASS |
| 3 | ACT FP32-weight IR p50: CPU 83.279, GPU 3.1851, NPU 3.8307 ms | bench_runB_quiet.json | read_file | PASS |
| 4 | ACT FP16-weight IR p50: CPU 80.3801, GPU 3.2107, NPU 3.8594 ms | bench_runB_quiet.json | read_file | PASS |
| 5 | PatchCore IR input is [2,3,256,256], batch 2 | bench_runB_quiet.json | read_file | PASS; corrects the earlier "batch 1" |
| 6 | PaDiM batch 1: p50 CPU 7.378, GPU 1.4847, NPU 6.1837 ms; NPU busy 0.9369 over 1 sample | bench_runB_quiet.json | read_file | PASS |
| 7 | INFERENCE_PRECISION_HINT is read after compile (not set by the bench); float32 on CPU and float16 on GPU and NPU in all 12 rows | bench_runB_quiet.json; bench_ov.py grep | read_file, grep | PASS; corrects the earlier "bench sets the FP16 hint" wording |
| 8 | Power not measured: pkg_w and energy_per_inference_j null; "root-only" text in the power block | bench_runB_quiet.json | read_file | PASS |
| 9 | Idle 30 s baseline: package temperature mean 67.6, max 95.0 C, 30 samples | bench_runB_quiet.json | read_file | PASS |
| 10 | All four ACT GPU and NPU timed windows had 0 telemetry samples (0.661 to 0.798 s) | bench_runB_quiet.json | read_file | PASS |
| 11 | loadavg_1min_before_timed ranged 3.18 to 7.82 over the 12 rows | bench_runB_quiet.json; summary | read_file | PASS; the file name says "quiet" but the load was not |
| 12 | bench_ov.py: WARM 20, TIMED 200, writes bench.json, no argparse and no sys.argv | bench_ov.py | grep | PASS |
| 13 | bench.json last written 15:08:03Z; run_final.log starts 15:04:42Z with no exit line | ls -l; run_final.log | start_process, read_file | PASS |
| 14 | events.py: EVENT_TYPES at lines 26-29 (scripted and policy names); COMMANDS = start, pause, resume, next, stop, quit | events.py | read_file | PASS |
| 15 | policy_rules.yaml: logs_root logs/loop_v2 (line 9); presence flag run/HASSAN_PRESENT (line 10); both plate model ids PLACEHOLDER; det2.module_dir null; still.exposure null | policy_rules.yaml | read_file | PASS |
| 16 | policy_live.py refusal checks: presence flag, placeholder model id, det2.module_dir, still.exposure | policy_live.py | grep | PASS |
| 17 | policy_run.py command set: scenarios; dry SCENARIO LOGS_ROOT; live RULES, which asks for GO | policy_run.py | read_file | PASS |
| 18 | tools/run_policy.py is absent; policy_run.py has no __main__ entry | ls; grep | start_process | PASS; new finding |
| 19 | No wrist or recorder pid reference in app/loop_v2/app/loop/ (matches are joint names and binary .pyc files) | grep -rn | start_process | PASS; new finding; the README loop-refusal claim is spec only |
| 20 | logs/loop and logs/loop_v2 absent; app/dashboard absent; logs/ovrun present | ls -d | start_process | PASS |
| 21 | LOOP-CONTRACT line 23 names R/logs/loop/<run_id>/ and the current symlink | LOOP-CONTRACT.md | grep | PASS |
| 22 | arm_state reads "BUSY 1436289 (checked 2026-11-14T15:08:09Z)"; pid 1436289 is a Studio backend publisher worker (elapsed 53:43) | run/arm_state; ps | read_file, ps | PASS |
| 23 | Both G-M connect lines carry one role=leader token, on the by-id follower port 5B3D040939-if00 | GMB, GMG | grep -o | PASS |
| 24 | G-M: 23 of 23 steps arrived in both runs; max error 5.17 (blue, step 18) and 5.27 (green, steps 10 to 14); connect times 16:29:07.588 and 16:30:47.953 (+02:00) | GMB, GMG | read_file | PASS |
| 25 | D-063: G-M passed at 14:33Z at 15 units/s; tolerance 3 too tight, use 6; logs say 52 s and 60 s | DECISIONS.md | grep | PASS; D-063 rounds the max error to 5.3 |
| 26 | D-067 (14:50Z): verify_PP-BOTH checked for episodes 11, 12 and 15 | DECISIONS.md | grep | PASS (line cut at 330 characters) |
| 27 | P2-P-PLAN has no other run-folder path; the only logs/loop mention is in LOOP-CONTRACT | P2-P-PLAN.md | grep | PASS |
| 28 | Telemetry sampler is writing: last line 15:08:34.768Z, pkg_w null; file modified 15:10:54Z | 20261114.jsonl | tail; ls | PASS |
| 29 | Two copies of studio_runtime_client.py exist (app/loop and app/loop_v2/app/loop) | ls | start_process | PASS |
| 30 | app/loop_v2/tests holds test_v2.py (16582 bytes, written 15:06Z) beside __init__.py | ls | start_process | PASS; an earlier listing showed only __init__.py, before the file appeared |
| 31 | Earlier remark that PatchCore and PaDiM figures match "synthetic numbers in D-041": D-041 (line 54) is a candidate-selection record with no such figures | DECISIONS.md | grep | WITHDRAWN; not used in any document |
| 32 | InferenceModel: 0 matches in both RAS repo copies (confirms the D-070 correction) | robotics-ai-suite and robotics-ai-suite-staging repo copies | grep -rn | PASS |
| 33 | JUNCTION2026.md line 122 links Studio exports to OpenVINO PhysicalAI InferenceModel; line 116 is the Step 6 heading | physical-ai-studio/JUNCTION2026.md (KB repo copy) | sed -n 116,123p | PASS |
| 34 | RAS workshop benchmark recipe: object-detection.yaml lines 38-47; line 42 loops over CPU, GPU and NPU; line 45 runs benchmark_app -m $M -d $D -hint latency -t 10 | robotics-ai-suite object-detection.yaml (repo copy) | read lines 34-50 | PASS |
| 35 | bench_ov.py line 77 compiles with PERFORMANCE_HINT LATENCY (run B latency hint) | app/telemetry/bench_ov.py (laptop) | grep -n | PASS |
| 36 | D-080 device rule: GPU explicit on every load_model; auto resolves to CPU (per D-080); fallback timeout must not drop torque | ledger/DECISIONS.md line 112 | read lines 101-113 | PASS for the ledger text; the physicalai code lines it cites (model.py 159-160 and 675-682; openvino.py 171) were not found on disk, so that part is not verified |
| 37 | Replay latency, update_at_query_ticks (52 ticks each): CPU p50 121.00348 and p95 146.87733 ms; GPU p50 9.29050 and p95 11.22876 ms | P2-B-OVRUN E1 results replay_CPU.json and replay_GPU.json, lines 1492-1493 | sed -n; grep | PASS |
| 38 | Replay summary, CPU block: executed MAE 4.978 against hold-state MAE 1.556 (in-sample) | replay_summary.json, lines 31 and 56 | grep -n | PASS; supports the no-success rule |
| 39 | RULES plates BLUE (line 32) and GREEN (line 41) both set device auto | app/loop_v2/app/loop/policy_rules.yaml | sed -n 24,44p | PASS; conflicts with D-080; open item 14 |
| 40 | D-070 corrected text (15:25Z): the RAS docs do not name InferenceModel; citable RAS item is the workshop benchmark recipe | ledger/DECISIONS.md line 113 | read lines 101-113 | PASS |

Carried from earlier runs and not re-read in this pass: G line numbers and the rubric weights; T08 grep lines; FL entries (11:30Z to 14:43Z); the CUR CSV (16 rows, KEEP 11, 12, 15, frame counts); models.tsv and TRAIN-OK; the shared inspection-label row count; probe log values (14:13:27Z to 14:13:31Z); scan_once.py arguments; telemetry_ctl.sh usage; wrist_rec.sh header; P2-D-DATA targets; the unrelated G line 98 hit; DEC D-024, D-033, D-037, D-042, D-045, D-052, D-055, D-057, D-059, D-060, D-062, D-066, D-068, D-069; OD-5, OD-8a, OD-9, OD-10, OD-11, OD-12; I-19, I-44, I-52, I-54.

## 3. Corrections made in this pass

- Benchmark source: "BENCH" now means run B (bench_runB_quiet.json, bench_summary_runB.txt). The earlier provisional rows (CPU 83.279 ms, GPU 3.1851 ms, NPU pending) are run B values, and the NPU rows are now present.
- PatchCore: the bench measured batch 2 (earlier drafts said batch 1). Batch 1 per brick is TBD.
- Precision: the bench reads INFERENCE_PRECISION_HINT after compile. It does not set it. The ACT rows are two IR weight variants. The earlier wording was replaced.
- Loop code path: app/loop_v2/app/loop/ (events.py, policy_run.py, policy_live.py, policy_cycle.py, triage.py, policy_rules.yaml). The earlier references to app/loop_v2/triage.py and run_log.py are removed.
- Event names: now from events.py (EVENT_TYPES, lines 26-29). The old RUNLOG source is retired. The list now includes inspect_cameras, handover, align, sort_start and sort_end.
- Commands: the command set comes from policy_run.py. The dry-run and live commands are marked not runnable (runner missing). No command was run in this pack.
- Live gating: added the policy_rules and policy_live conditions (placeholder model ids, det2 module_dir null, still.exposure null).
- Wrist recorder: "the loop refuses while the pid is alive" is marked as spec only, because the code has no such check.
- Run folder: the logs/loop (contract) and logs/loop_v2 (rules) conflict is now an open item.
- Camera IDs: the RULES comments give a mapping (earlier drafts said unknown). It is marked as not checked against the live camera list.
- Line numbers: the bench_ov.py line numbers are removed from every document.
- Times: the G-M log times (+02:00) are converted to UTC. The G-A gate is now "PASS (probe 14:13:27Z to 14:13:31Z)", replacing "PASS at 14:14Z".
- Arm state: BUSY at 15:08:09Z is added to the preflight, the checklist and the open items.
- Storyboard: the benchmark source is run B; the TEL field names temp_c.pkg and npu.busy are confirmed.
- Withdrawn: the D-041 remark (item 31 above).
- Robotics AI Suite (coordinator correction, D-070 corrected 15:25Z): the runner is not described as a RAS integration. The RAS item cited is the workshop benchmark recipe (object-detection.yaml lines 38-47), whose explicit-device and latency-hint method run B follows. README item 11 and evidence Rubric 2 were changed; the earlier "no decision through D-069" wording is replaced.
- Policy device (D-080): recorded as GPU explicit, with auto resolving to CPU per D-080. The replay numbers are cited to their result files. The RULES plates still say auto (lines 32 and 41): README open item 14 and a go/no-go gate in the checklist.

## 4. Blockers and gaps (not resolved by this pack)

- No runnable loop entry point. tools/run_policy.py is absent and policy_run.py has no __main__ entry. Dry and live runs cannot be started as documented.
- Arm state BUSY (pid 1436289, a Studio backend publisher worker). This pack did not stop it; it is not a pack process. Motion stays blocked until the state reads FREE.
- Run folder path conflict: logs/loop (LOOP-CONTRACT section 5) against logs/loop_v2 (policy_rules.yaml). One path must be chosen.
- The wrist recorder stop is manual; no code check exists.
- Driver role=leader label on the by-id follower port: unresolved (FL 13:42Z maps the port to the follower).
- Episode sets: v1 drops episodes 7 and 14 that D-067 admits; the v2 CSV is not on disk.
- Power and energy: no non-root source; all power fields are null or "not measured".
- Benchmark: not precision-matched; load before the timed windows 3.18 to 7.82; bench.json is being rewritten by run_final, so only the run B files are cited.
- Two copies of studio_runtime_client.py: which copy the loop imports is not verified.
- The guide under /home/intel/Documents was not opened (outside the allowed folder, L-007). The repo copy was read instead.
- Laptop calls made before the context summary are not kept verbatim (see laptop-log.md section 0).
- The remote-devices MCP server reported a connection failure (503) during this run. It was not used.
- RULES plates set device auto (lines 32 and 41), which D-080 rules out. Not changed by this pack (no code or rules edits).
- The physicalai code lines cited by D-080 (model.py 159-160 and 675-682; openvino.py 171) were not found in the KB repos or the laptop E1 folder in this pass. The auto-resolves-to-CPU statement rests on D-080.

## 5. Memory and privacy

- No memory tool was called. No memory fact was used in any document.
- The documents hold the team's file names, dataset keys and Studio identifiers that the inputs already contain. No personal details were added.

## 6. Tool notes

- Desktop Commander calls were limited to read_file, write_file, start_process, get_tool_result and ToolSearch loads (read_process_output was loaded but not called). One edit_block call appended section 7 to this file on the laptop copy. No kill_process, force_terminate, move_file, set_config_value, shutdown or sudo was used.
- Laptop shell commands were read-only: ls, grep, sha256sum, tail, head, date, ps and echo. No motion command, no camera or serial access, no process started or stopped.
- The loop package and the bench were read, not run. No file outside /home/intel/intel-junction-2026 was listed or read, except the ps line that names one process path, recorded without reading that folder.
- A start_process `true` call, made before the summary, is recorded in laptop-log.md section 0.
- Correction pass: laptop calls were read-only start_process commands (sed, grep, ls, find, sha256sum) and edit_block calls that applied the same old and new text as the sandbox edits to the laptop submission copies (laptop-log.md section 4).

## 7. Coverage checklist (internal, for review; remove before publishing)

- [x] The five documents were checked for em dashes, en dashes and non-ASCII characters (grep, 15:17Z and 15:36Z); each ends with a Coverage checklist (15:36Z).
- [x] Laptop submission copies: SHA-256 identical to the sandbox drafts (15:18Z, and again after the correction pass; laptop-log.md section 4).
- [x] R12: 40 facts listed in section 2 (39 PASS, 1 WITHDRAWN); 10 were required.
- [x] Every laptop call in this run is listed in laptop-log.md with its output; calls before the summary are in its section 0 (not retained verbatim).
- [x] STATUS.md checked for non-ASCII characters (0 lines, 15:20Z).
- [ ] Guide under /home/intel/Documents: not opened (outside the allowed folder, L-007).
- [ ] Other workers' outputs (DET2 report, PICK offline_eval, OVRUN report, R9 report): not opened (isolation, not inputs).
