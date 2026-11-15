# Evidence: one table per rubric line

Status at 2026-11-14 15:11Z, revised after the 15:25Z decision corrections (D-070, D-080). Rubric weights and what judges look at: [G lines 133-140, rubric table; T08 grep line 228-232]. Points total 100 [G line 140].

How to read the tables:
- Value "TBD" means no verified measurement exists yet. The source column names the file that will supply the value.
- "Measured" means the value was read from the file named in the row, in this run or in an earlier run listed in the coverage checklist.
- "Run B" means BENCH: R/work/P2-B-OPT/E1/bench_runB_quiet.json (started 14:57:53Z, finished 15:02:36Z) and bench_summary_runB.txt. Run B is a benchmark, not a take result. R/work/P2-B-OPT/E1/bench.json is being rewritten by a later run and is not cited.
- Source keys as in README.md. R = /home/intel/intel-junction-2026. RUN_ID is the run folder name (yyyymmdd-HHMMSS, UTC); the run folder path itself is open (README section 8, item 4).

## Rubric 1: Task Success Rate (25 points)

Judges look at accuracy, cycle time, and robustness to shifted objects and clutter [G line 135].

| Metric | How measured | Source file | Value | Target or gate |
|---|---|---|---|---|
| Correct sorts in the take | A trial is correct when the brick is on the plate OD-10 requires (BLUE for GOOD, GREEN for DEFECTIVE or UNCERTAIN), checked by the top frame at release | Run folder events.jsonl, verify events (target_id, plate_intended, release_side, ok) [EV; LC section 8]. Folder path open (README section 8, item 4) | TBD | Take: at least 3 correct sorts [PLAN section 2, Take row; PLAN section 1, SC1] |
| Sorted trials in the 16-trial run | 16 trials; count of sorted bricks, with shifted and clutter scenes if the team sets them up | Run folder events.jsonl (verify events) | TBD | At least 11 of 16 [PLAN section 2, G-SC2 row]. If shifted and clutter scenes are not set up, report normal-only counts [PLAN R10, C-7] |
| Plate balance | Correct sorts counted per plate | Same events file | TBD | Each plate at least 4 of 5 [PLAN section 2, G4-SORT row] |
| Inspection check set | Defect recall and good accuracy on the labelled check set, counted once | shared/inspection-labels.tsv (Opus labels; 160 lines at 14:58Z) and the per-brick verdicts in the detector output (P2-B-DET2 report, not read) | TBD | Defect recall at least 0.80; good accuracy at least 0.90 [PLAN section 2, G-CHECK row] |
| Cycle time | Mean cycle time per run and per-skill duration p50 and p95 | Run folder events.jsonl, stats event (mean_cycle_s) and skill_end events (duration_s) [EV; LC section 5]; calib.json in the P2-D-DATA report folder [P2-D-DATA item 3] | TBD | None set in the inputs; suggested skill max_s = 1.5 x p95 [P2-D-DATA item 3] |
| Gate G-A, handover (measured, not a take result) | Environment connect 14:13:27Z, environment loaded by 14:13:29Z; model load 0.91 s; follower port free after unload 0.0 s; reload with the same draft at 14:13:31Z; model kept on reload: true | PROBE (earlier run) [DEC D-057] | PASS (probe 14:13:27Z to 14:13:31Z) | Gate G-A at 14:20Z [PLAN section 2] |
| Gate G-M, recorded paths (measured, not a take result) | Blue: 23 of 23 steps arrived; maximum error 5.17 units (step 18); connect 14:29:07.588Z, disconnect 14:29:59.893Z. Green: 23 of 23 steps arrived; maximum error 5.27 units (steps 10 to 14); connect 14:30:47.953Z, disconnect 14:31:47.976Z. Speed 15 units/s [DEC D-063] | GMB and GMG (re-read at 15:10Z; their timestamps are +02:00) | PASS | Gate G-M at 15:00Z [PLAN section 2] |

Note on G-M: DEC D-063 logs the maximum error as 5.3 units and the durations as 52 s and 60 s. The per-step values in the logs are cited here. The G-M numbers come from the recorded human paths replayed by our code, not from a brick. They show that the motion path reaches the plates within tolerance. They are not sort results.

## Rubric 2: Intel Tech Integration (25 points)

The rubric line names offloading to the NPU or iGPU, deployment with OpenVINO, and the Robotics AI Suite [G line 136]. This pack does not claim a Robotics AI Suite integration of our runner (D-070, corrected 15:25Z).

| Metric | How measured | Source file | Value | Target or gate |
|---|---|---|---|---|
| Components with an explicit device | Count of components with a named device and a measured latency | BENCH run B | 12 of 12 benchmark rows have status ok (4 model variants on CPU, GPU and NPU). The cell's own component list: TBD (architecture.md section 3) | None set in the inputs |
| Policy exported and run on OpenVINO | Policy IR exported with backend openvino, then loaded through Studio's runtime with backend openvino | MODELS (backends column = openvino for 389da0fd); PROBE load_model line (14:13:30Z); RULES plates block (backend openvino) | Export recorded (MODELS). Run on our own loop: TBD (live refused while the plate model ids are placeholders) | Studio supports ACT, SmolVLA and Pi0.5 [T08 grep line 200] |
| Detector on OpenVINO GPU | PatchCore FP16 IR, latency per brick | BENCH run B (native batch 2 IR): GPU p50 8.4712 ms per batch of 2 | Batch 2 measured; batch 1 per brick TBD | GPU first, NPU time-boxed, CPU baseline [DEC D-033] |
| NPU offload, measured or failed | Latency of each model on the NPU, plus NPU-busy evidence from telemetry | BENCH run B (status ok for all NPU rows): ACT FP32 IR p50 3.8307 ms; ACT FP16 IR 3.8594 ms; PatchCore (batch 2) 81.6142 ms; PaDiM 6.1837 ms. NPU busy in the telemetry window: PatchCore 0.9949 mean (16 samples); PaDiM 0.9369 (1 sample); the ACT NPU windows had 0 samples | Measured (run B) | Required by the rubric line: offloading to the NPU or iGPU [G line 136] |
| iGPU offload, measured | Latency of each model on the iGPU | BENCH run B: ACT FP32 IR p50 3.1851 ms; ACT FP16 IR 3.2107 ms; PatchCore (batch 2) 8.4712 ms; PaDiM 1.4847 ms | Measured (run B); not precision-matched (Rubric 4) | None set in the inputs |
| Robotics AI Suite | Used or not, with the file that shows it | RAS workshop recipe, object-detection.yaml lines 38-47 [RAS]; D-070 corrected 15:25Z: the guide's link is JUNCTION2026.md line 122 [G], so our runner is not claimed as a RAS integration [DEC] | Recipe followed (a benchmark method, not a runtime). Runner integration: not claimed. Run B uses our own script with PERFORMANCE_HINT LATENCY (bench_ov.py line 77) [BSRC] | Rubric line names it [G line 136] |
| Policy device, explicit | Device named on each policy load, and the per-query latency of the replay | DEC D-080; RPL replay_GPU.json and replay_CPU.json (latency.update_at_query_ticks, 52 ticks each) | GPU p50 9.29 ms, p95 11.23 ms; CPU p50 121.00 ms, p95 146.88 ms (auto resolves to CPU per D-080). RULES plates still say auto (lines 32 and 41) | Device GPU explicit on every load_model [DEC D-080]; open until the rules file changes |

## Rubric 3: Dataset and Pipeline Quality (20 points)

Judges look at clean teleoperation data and effective fine-tuning in Physical AI Studio [G line 137].

| Metric | How measured | Source file | Value | Target or gate |
|---|---|---|---|---|
| Episodes in the source dataset (team recording, dataset 6d6bdce3) | Count of episodes with a KEEP or DROP decision | R/data/curated/episodes_v1.csv (16 rows, episodes 0 to 15) [CUR, measured in an earlier run] | 16 at 14:51Z | None set; counts rise as the team records [FL 14:00Z] |
| Episodes kept, convention v1 | Count of KEEP rows | CUR (episodes 11, 12 and 15 are KEEP) [measured in an earlier run] | 3 | None set |
| Episodes kept, convention v2 | Count after the v2 rule, which also admits episodes 7 and 14 [DEC D-067] | v2 CSV, not on disk | TBD (expected 5 pick-and-place episodes plus 4 S1 put-downs under D-067; not yet built) | None set |
| Frames kept, v1 | Sum of frames_kept for the KEEP rows: 793 + 832 + 623 (summed by hand for this table) | CUR [measured in an earlier run] | 2248 (derived by summing the three values) | None set |
| Per-plate episodes, v1 | BLUE: 11, 12. GREEN: 15 | CUR [measured in an earlier run] | BLUE 2, GREEN 1 | Train good at least 40 and test defect at least 10 [PLAN section 2, G-DATA row; OD-5] |
| Per-plate episodes, v2 | BLUE: 7, 11, 12. GREEN: 14, 15 [DEC D-067] | v2 CSV, not on disk | TBD | As above |
| Every episode has a verdict and reasons | Each row carries decision and reasons | CUR (16 of 16 rows have a decision; reasons on each) [measured in an earlier run] | 16 of 16 | Clean data [G line 137] |
| Simulation episodes in training | Count of sim episodes in any training set | DEC D-055 and D-069 | 0 by decision | Not used [DEC D-069] |
| Team models in the cell | Count of team models in the loop | DEC D-066 | 0 (the probe loaded fa42ee76 once, as a test) | Not used [INS I-54] |
| Smoke training run (pipeline proof) | Steps, epochs, duration, loss | FL 14:08Z (2056 steps, 8 epochs, 13:51-13:59Z, loss 2.78 to 0.99); MODELS row for 389da0fd (2056 steps) | Measured in an earlier run: 2056 steps, 8 min, loss 2.78 to 0.99 | TRAIN-OK line per version [shared/curation/TRAIN-OK] |
| Imports verified by GET | Episode counts, lengths and texts of each new dataset match the CSV | P2-D-CUR1 step 3; no GET record on disk | TBD | Verified before training [P2-D-CUR1 step 3] |
| Final policy trained on v2 sets | Per-plate models or one text-conditioned model, with steps and loss | models.tsv (no row after 389da0fd at 14:01Z) | TBD | Full run needs a new TRAIN-OK line [TRAIN-OK] |

Note: the S2 and S3 policy targets in P2-D-DATA (at least 30 each, minimum 25) are superseded. OD-9 removed the S2 and S3 policies [PLAN section 1], and OD-12 uses one policy per destination [DEC D-060]. The S1 target of at least 40 (minimum 30) [P2-D-DATA NEEDS] was set for the S1 put-down policy. The v2 sets have no numeric target in the inputs. D-067 keeps four S1 put-downs (3 to 6) in the MT-ALL set [DEC D-067].

## Rubric 4: System Efficiency (20 points)

Judges look at latency per inference step, thermal headroom, and power consumption on Core Ultra [G line 138]. All values in this table are run B (benchmark, not a take result).

| Metric | How measured | Source file | Value | Target or gate |
|---|---|---|---|---|
| Latency per inference step, ACT 389da0fd, IR as exported (FP32 weights), CPU | Batch 1; 20 warm-up and 200 timed iterations; p50, p90, p95 | BENCH run B, act_389da0fd_fp32 on CPU (latency_ms) | p50 83.279 ms; p90 91.7998 ms; p95 93.7206 ms | None set in the inputs |
| Same IR, GPU | As above, explicit device GPU | BENCH run B | p50 3.1851 ms; p90 4.1904 ms; p95 4.5261 ms | None set in the inputs |
| Same IR, NPU | As above, explicit device NPU | BENCH run B | p50 3.8307 ms; p90 4.3542 ms; p95 4.8209 ms | None set in the inputs |
| Latency per inference step, ACT 389da0fd, FP16 weights (CPU, GPU, NPU) | As above | BENCH run B, act_389da0fd_fp16 | p50 80.3801 ms, 3.2107 ms, 3.8594 ms | None set in the inputs |
| Latency, PatchCore detector, native batch 2 (CPU, GPU, NPU) | As above; one inference is a batch of 2 images | BENCH run B, patchcore_detector_native_b2 | p50 107.277 ms, 8.4712 ms, 81.6142 ms per batch | None set in the inputs |
| Latency, PaDiM detector, batch 1 (CPU, GPU, NPU) | As above | BENCH run B, padim_b1_detector | p50 7.378 ms, 1.4847 ms, 6.1837 ms | None set in the inputs |
| Precision match | Same precision for every device row | BENCH run B: INFERENCE_PRECISION_HINT, read after compile (BSRC) | float32 on CPU and float16 on GPU and NPU in all 12 rows. Not matched. Required before a claim (open) | Required before a claim |
| Throughput | Inferences per second, timed run | BENCH run B, throughput_ips | ACT FP32 IR 11.769 (CPU), 291.284 (GPU), 252.008 (NPU); ACT FP16 IR 12.385, 302.549, 250.575; PatchCore (batch 2) 8.819, 116.296, 12.111 batches/s; PaDiM 132.802, 636.854, 158.509 | None set |
| Compile time | Time to compile the IR on each device | BENCH run B, compile_s | ACT FP32 IR 0.462 s (CPU), 0.992 s (GPU), 0.043 s (NPU); ACT FP16 IR 0.504, 0.663, 0.043; PatchCore 0.174, 1.567, 5.847; PaDiM 0.052, 0.464, 2.189 | None set |
| Thermal headroom | Package temperature during timed windows and idle | BENCH run B, telemetry_during_timed and baseline_idle_30s | Idle 30 s baseline: pkg temperature mean 67.6 C, max 95.0 C (30 samples). ACT FP32 IR CPU window: mean 91.6471 C, max 97.0 C (17 samples). ACT FP16 IR CPU window: mean 95.375 C, max 97.0 C (16 samples). All four ACT GPU and NPU windows: 0 samples (0.661 to 0.798 s) | None set in the inputs |
| Latency per inspection | Time per brick inspection | Run folder events.jsonl, inspection event (latency_ms) [LC section 8] | TBD | None set |
| Latency per sorting cycle | Cycle time from the stats event and skill_end events | As in Rubric 1, cycle time row | TBD | None set |
| Package power | Package power in watts | TEL pkg_w (null at 15:08:34.768Z); BENCH power.pkg_w "not measured: RAPL energy_uj is root-only on this laptop"; pkg_w_mean null in all 12 rows | Not measured. Blocked without a non-root power source | Rubric names power consumption [G line 138] |
| Energy per inference | Joules per inference | BENCH energy_per_inference_j (null in all 12 rows) | Not measured. Blocked, as for power | None set |
| Energy per sorting cycle | Joules per cycle, between two timestamps | TEL reader API (energy per cycle, P2-B-OPT step 2); pkg_w is null | Not measured. Blocked, as for power | None set |

Note: the battery reading (bat_power_w in the benchmark) is not system power while the laptop is on AC. The benchmark's power block says so: "AC online; BAT0 power_now is not system power while on AC" [BENCH power block].

## Rubric 5: Creativity (10 points)

Judges look at original ideas beyond the baseline [G line 139]. There is no numeric measure. The evidence is the design, described in architecture.md and shown in the storyboard.

| Idea | Where it is described | Measured? |
|---|---|---|
| Triage that sends confident defects straight to GREEN, with no inspection step [INS I-52; LC section 8] | architecture.md section 1 | No |
| One text-conditioned policy per destination, with inspect-then-act [DEC D-060, OD-12] | architecture.md section 1 | No |
| Handover that keeps the loaded model across a scripted sort [PROBE; DEC D-057] | architecture.md section 2 | Yes, G-A (Rubric 1 table) |
| Two-view inspection: tape from the close view, shape from the top view [DEC D-068] | architecture.md section 1 (INSPECT) | No: detector accuracy TBD (Rubric 1 table) |
| Fail-safe routing: an uncertain brick never goes to BLUE [LC section 8] | architecture.md section 1 | No |

## Gates (PLAN section 2) and their status at 15:11Z

| Gate | Time (UTC) | Status | Source |
|---|---|---|---|
| G-A handover | 14:20Z | PASS (probe 14:13:27Z to 14:13:31Z) | PROBE; DEC D-057 |
| G-M motion | 15:00Z | PASS (round trips 14:29:07Z to 14:31:47Z) | GMB, GMG; DEC D-063 |
| G4-S1 (at least 4 of 5 spot placements) | 15:40Z | TBD | PLAN section 2 |
| G-DATA (train good at least 40, test defect at least 10) | 15:50Z | TBD | PLAN section 2 |
| G-DET (detector frozen) | 16:15Z | TBD | PLAN section 2 |
| G-CHAIN and G-CHECK | 16:30Z and 16:35Z | TBD | PLAN section 2 |
| G4-SORT (each plate at least 4 of 5) | 17:35Z | TBD | PLAN section 2 |
| G-SC2 (16 trials, at least 11 sorted) | 19:00Z | TBD | PLAN section 2 |
| Freeze | 20:00Z | Not reached | PLAN section 2 |
| Take (at least 3 correct sorts) | 21:50-21:58Z | Not reached | PLAN section 2 |

## Open items that block the numbers

1. No run folder exists yet (neither logs/loop nor logs/loop_v2, checked 15:08Z). Every task-success and cycle-time value stays TBD until the folder path is chosen and a run writes events.jsonl.
2. Run B is not precision-matched. ACT is benchmarked as two IR weight variants, and the PatchCore IR is batch 2 [BENCH; BSRC].
3. bench.json is being overwritten by a later run (run_final.log: harness start 15:04:42Z, no exit line at 15:08Z). Cite run B files only.
4. The machine was not quiet by measured load: the 1-minute load before timed windows ranged from 3.18 to 7.82 [BENCH].
5. Package power and energy cannot be read without root [TEL; BENCH].
6. The v2 episode CSV is not on disk; v1 drops episodes 7 and 14 that DEC D-067 admits [CUR; DEC D-067].
7. No runnable loop entry point: tools/run_policy.py is absent [PRUN; checked 15:08Z].
8. The arm state read BUSY at 15:08:09Z (pid 1436289) [ARM].
9. RULES plates set device auto (lines 32 and 41). D-080 requires GPU explicitly. The replay latencies in Rubric 2 are replay values until a live run uses the GPU device.

## Coverage checklist (internal, for review; remove before publishing)

- [x] G lines 129-140 (rubric) read on the laptop copy (earlier run).
- [x] T08 grep line 200 (Studio supports ACT, SmolVLA, Pi0.5) and lines 228-232 (rubric table) in the KB copy (earlier run).
- [x] PLAN section 1 (changes), section 2 (gates and SC targets), section 6 (R1-R12, only R10 and R11 cited) (earlier run).
- [x] DEC D-033, D-042, D-055, D-057, D-060, D-063, D-066, D-067, D-068, D-069; OD-9 and OD-12 (through DEC) (earlier run); D-063 and D-067 re-read at 15:10Z.
- [x] FL 14:00Z, 14:08Z, 11:40Z read (earlier run).
- [x] INS I-52, I-54 read (earlier run).
- [x] PROBE (earlier run); GMB and GMG (re-read at 15:10Z); CUR (all 16 rows, earlier run); TEL tail (15:08Z); BENCH run B (full JSON and summary, this run); BSRC grep (this run); EV and RULES (this run); MODELS, TRAIN-OK, P2-D-DATA NEEDS and item 3, P2-D-CUR1 step 3 (earlier run).
- [x] shared/inspection-labels.tsv: row count only (160 lines, wc) (earlier run).
- [ ] P2-B-DET2 report, P2-R9-RAS report, P2-B-OVRUN report: not read (not in inputs). The P2-B-OVRUN replay JSON files (RPL) were read in the correction pass.
- [x] Correction pass (15:25Z to 15:31Z): RAS recipe lines 38-47 and the InferenceModel grep (0 matches, both RAS copies); JUNCTION2026.md line 122; DEC D-070 (corrected) and D-080; RULES lines 24-44; bench_ov.py line 77; replay_CPU.json, replay_GPU.json and replay_summary.json (CPU block). The physicalai code lines cited by D-080 were not found on disk and are not verified here.
- [ ] R/data/studio-datasets-report/live/: not read in this run (path named from the P2-D-CUR1 input list).
- [x] NPU rows of BENCH: present in run B (this run).
