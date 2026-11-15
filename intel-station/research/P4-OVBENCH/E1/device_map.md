# P4-OVBENCH E1: device map (OpenVINO 2026.4.1; FP16 unless noted)

Status: PARTIAL, written 2026-11-14 about 21:53Z. Laptop intel-HP-OmniBook-Ultra-Laptop-14t-kd000. CPU Intel Core Ultra X9 388H, 16 cores. GPU Intel Graphics [0xb080] (iGPU, xe driver, shared with the desktop). NPU Intel AI Boost.

## 1. Parity rule

- Reference: CPU of the same IR on the same inputs. Inputs: the bench's recorded snapshots, 8 per model (input_source "recorded snapshots"). The ACT reference is the FP32 IR on CPU.
- Output PASS: every value finite, and max abs diff <= 1% of the reference absmax (ACT 0.787; pre_close about 4.5; PatchCore and PaDiM maps 0.01).
- Decision agreement: pre_close uses pred_score > 534.8227 (production threshold, threshold.json). PatchCore and PaDiM use pred_label > 0.5 (the IR's own 0/1 output). Their outputs are normalized to [0, 1], so the P4-FUSION thresholds do not apply to them. ACT has no decision output (n/a).
- Row PASS: every output PASS and every decision agrees. A device that fails is not eligible for that model, whatever its speed.
- Native batch 2 (pre_close, PatchCore; batch-1 reshape fails): per image = call / 2, stated in every row.

## 2. Results (p50 ms; p90 in brackets; async in fps)

### 2a. ACT 389da0fd (batch 1; reference FP32 CPU)

| IR | device | hint | p50 [p90] | async fps | max abs diff | elements > 1% | decisions | verdict |
|---|---|---|---|---|---|---|---|---|
| FP32, production IR | CPU | default, 4 threads | 129.70 | 6.25 | reference | - | n/a | reference |
| FP32, production IR | GPU | default | 4.31 | 228.0 | 0.0984 | 0 of 4800 | n/a | PASS, eligible |
| FP32 | GPU | f32 | 17.09 [18.24] | 60.1 | 1.1e-4 | 0 | n/a | PASS, 4x slower |
| FP16 | GPU | default | 3.93 [4.87] | 210.7 | 0.108 | 0 | n/a | PASS (not production IR) |
| FP16 | NPU | default | 7.90 [8.85] | 112.9 | 0.234 | 0 | n/a | PASS (not production IR; NPU busy 97.8%) |
| FP16 | CPU | default | 118.59 [141.05] | 10.35 | 48.79 | 3962 of 4800 | n/a | FAIL |
| FP16 | CPU | f32 | not timed | - | 48.79 | 4025 of 4800 | n/a | FAIL (the hint does not fix it) |

Production IR: R/work/P2-B-OVRUN/E1/models/389da0fd/openvino/act.xml. Its act.xml and act.bin are byte-identical to R/work/P2-B-OPT/E1/models/389da0fd/act.xml and act.bin (sha256 202001776369 and 930270a190). The production config (GPU, default hint) is the act_ovrun run: max 0.0984, mean 0.0153, 0 elements above 1%.

### 2b. Anomaly pre_close (production scorer anom3; FP16 IR; batch 2; per image = call / 2; reference CPU FP16)

| device | hint | per image p50 [p90] ms | max abs diff (pred_score / anomaly_map) | non-finite | elements > 1% | decisions (score > 534.8227) | verdict |
|---|---|---|---|---|---|---|---|
| CPU | default | 128.81 [185.01] | reference | 0 | - | 0 of 16 flagged (max score 453.9) | reference, production |
| GPU | default (FP16) | 5.05 [5.88] | inf / inf | 858,960 | 858,960 | 0 of 16 agree | FAIL |
| GPU | f32 | 39.19 [41.69] | 5.2e-4 / 6.7e-4 | 0 | 0 | 16 of 16 agree | PASS |
| NPU | default (FP16) | 20.42 [22.84] | 218.2 / 243.5 | 0 | 556,242 (map) | 16 of 16 agree | FAIL |

### 2c. Anomaly PatchCore (batch 2; reference CPU FP16; not in production)

| device | hint | per image p50 [p90] ms | max abs diff (anomaly_map) | elements > 1% (map) | decisions (pred_label) | verdict |
|---|---|---|---|---|---|---|
| CPU | default | 183.56 [232.02] | reference (map saturated at 1.0) | - | 16 of 16 flagged | reference |
| GPU | default | 10.63 [12.76] | 1.0 | 854,740 | 0 of 16 agree (all flipped) | FAIL |
| GPU | f32 | 39.84 (call 79.69, n=38; p90 n/a) | 3.9e-6 | 0 | 16 of 16 agree | PASS on these 8 snapshots (all flagged) |
| NPU | default | 73.29 [96.94] | 0.0029 | 0 | 16 of 16 agree | PASS on this IR; conflicts with P4-FUSION (section 4) |

### 2d. PaDiM (batch 1; reference CPU openvino IR)

| IR | device | hint | p50 [p90] ms | async fps | anomaly_map max / elements > 1% | pred_mask max / elements > 1% | label decisions | verdict |
|---|---|---|---|---|---|---|---|---|
| openvino (anom_padim) | CPU | default | 34.86 [49.31] | 32.0 | reference | reference | 8 of 8 flagged | reference |
| openvino_b1 (anom_padim_b1) | CPU | default | 31.05 [45.23] | 21.8 | 0 / 0 | 0 / 0 | 8 of 8 | PASS vs openvino CPU |
| openvino | GPU | default | 3.01 [5.32] | 334.1 | 1.0 / 397,713 | 1.0 / 301,276 | 8 of 8 | FAIL |
| openvino | GPU | f32 | 16.62 (n=40) | 50.8 | 1.0e-6 / 0 | 1.0 / 1 pixel | 8 of 8 | FAIL (strict: one mask pixel flips; map within 1e-6) |
| openvino | NPU | default | 12.51 [17.52] | 91.3 | 0.316 / 232,341 | 1.0 / 636 | 8 of 8 | FAIL |
| openvino_b1 | GPU | default | 3.70 [4.78] | 292.0 | 1.0 / 397,713 | 1.0 / 301,276 | 8 of 8 | FAIL |
| openvino_b1 | NPU | default | 16.88 [17.18] | 63.0 | 0.316 / 232,341 | 1.0 / 636 | 8 of 8 | FAIL (compiles) |

Compile, cold (s): ACT CPU 0.58, GPU 1.11, NPU 2.56, GPU f32 5.56. pre_close CPU 0.21, GPU 1.59, GPU f32 4.82, NPU 7.41. PatchCore CPU 0.39, GPU 0.98, NPU 0.04 (cache state not verified). PaDiM openvino CPU 0.08, GPU 0.27, NPU 0.03; openvino_b1 CPU 0.10, GPU 0.40, NPU 0.08.

Sources: bench.tsv and metrics.json (stage ovbench-v1, aggregated 21:49:14Z), raw/*.json, raw/*__out.npz, logs/s4_result.txt to logs/s7_result.txt (the full parity matrix v2 is in s7_result.txt).

## 3. Provisional map, updated

- Coordinator's provisional rule "anomaly scoring on NPU": rejected. pre_close NPU FAILS (max 243.5). PaDiM NPU FAILS (map 0.316; 636 mask pixels flip). PatchCore NPU passes on this IR only and conflicts with P4-FUSION (open).
- ACT 389da0fd: GPU, default precision (the production config). PASS, 4.31 ms per call, 228 fps. NPU would need an FP16 export of the production IR (not made). The CPU FP16 IR FAILS.
- Anomaly scoring (pre_close): stays on CPU (current production; no change). The only accelerated option that passes parity is GPU with the f32 precision hint (39.19 ms per image, against 128.81 on CPU). That needs an anom3 change (DEVICES at line 21 and the ViewModel guard at lines 45-48) and a precision hint. Neither was made here; this is an Opus decision.
- PatchCore (not in production): no device eligible yet. GPU f32 passes on the 8 snapshots. NPU passes here but conflicts with P4-FUSION. GPU default fails.
- PaDiM (not in production): CPU only. No device passes the strict rule; GPU f32 is within 1e-6 on the map but flips one mask pixel.
- Loop perception (OpenCV and numpy; no OpenVINO): CPU, unchanged (perception/perception_timing.tsv).

## 4. Checks requested by the coordinator

PaDiM IR on NPU. My 12.5 ms run used R/models/anomaly/padim/weights/openvino/model.xml (sha256 8fd54c04; models_fp16/anom_padim.xml is identical). P4-FUSION run_pipeline.sh feeds infer.py the openvino_b1 IR (sha256 1894a7c9, batch 1), which is my anom_padim_b1 source. It also runs static_ir.py on the openvino IRs first. device_check.json does not record which copy it compiled. My anom_padim_b1 compiled on NPU (16.88 ms), so the IE.Interpolate abort did not reproduce on either source IR. The .bin files of the two PaDiM IRs are identical (sha256 987542881); only the XML graphs differ.

PatchCore conflict, same source IR (R/models/anomaly/patchcore/weights/openvino/model.xml, sha256 0f691450, batch 2). Mine: NPU PASS (max 0.0029); GPU default FAIL (16 of 16 decisions flip). P4-FUSION device_check.json: NPU FAIL (max 344.59); GPU PASS (4.9e-4). Its scores are unclamped (about 237 to 537), while mine are normalized to [0, 1], so the output heads or post-processing differ. Its input set (32 crops) also differs from my 8 snapshots. The cause is not resolved.

## 5. Input set and decision test (limitation)

All parity runs use the 8 recorded snapshots per model. Under the CPU reference they are single-class. pre_close scores are all below 534.82 (max 453.9). PatchCore and PaDiM flag all 8, and PatchCore maps saturate at 1.0. Decision agreement therefore does not test threshold flips. The pre_close IR's own pred_label is 1 for all 16 CPU samples even though its scores are below the production threshold. The IR's label rule is not traced; production uses threshold.json. A labelled good and defect set with scores near 534.82 is needed before the anomaly device choice is final.

## 6. IRs

Included: the ACT production IR (section 2a); the pre_close FP16 IR (R/app/anomaly/anom3/models/v000/pre_close/model.xml, batch 2); the PatchCore IR (batch 2); the PaDiM openvino and openvino_b1 IRs. The E1 copies in models_fp16/ match their R sources for PaDiM and PatchCore (sha256). The ACT FP16 copy (models_fp16/act_fp16.xml; 201 f16 and 15 f32 tensors) was not sha-compared with R/work/P2-B-OPT/E1/models/389da0fd_fp16/act.xml.

Excluded (scratch, not deployed, not benchmarked): R/work/P2-B-DET2/E2 patchcore_aug_cpu and patchcore_plain_cpu; R/work/P2-B-DET2/E3 anom openvino_cpu; R/work/P2-B-ANOM3 scratch and superseded_cache_bug IRs; R/work/P2-B-OPT/E1/models/389da0fd_fp16 (ACT FP16 variant, not sha-compared); the P4-FUSION static_ir.py outputs. Not found: a SmolVLA IR.

## 7. Config lines (current; no change made to R/app)

- R/app/anomaly/anom3/infer.py line 21: DEVICES = ("CPU",), comment "CPU only: no GPU jobs on this laptop (coordinator update, 18:32Z)". Lines 45-48: the ViewModel refuses non-CPU devices.
- R/app/anomaly/anom3/models/v000/pre_close/threshold.json: threshold 534.8227233886719, device GPU, balanced_accuracy 0.9286, auroc 0.869. The metrics.json beside it records scoring_device GPU (open item 5).
- R/app/loop_v2/app/loop/policy_rules.yaml lines 33-34 and 42-43: device GPU (D-080); line 86: device CPU. policy_cycle.py line 53: device "GPU"; lines 60-63 enforce D-080.
- R/app/ovrun/runner.py lines 176, 233, 284-285: InferenceModel backend openvino, --device default GPU, --model-dir default rp.EXPORT. replay.py lines 27-28 point EXPORT at the P2-B-OVRUN IR. The ovrun package has no compile or precision call (grep). The InferenceModel class was not located, so the S6 run assumes it sets no precision hint (open item 10).
- Rejected candidate, not applied: device_map_NPU_candidate_REJECTED.patch (two-line change: DEVICES and the bad-list guard). No GPU candidate was written.

## 8. Power and load

- Power: RAPL is denied on 7 of 7 paths (perf_event_paranoid 4). NPU busy time (npu_busy_time_us) and GPU gt0 idle residency are readable, but they are not power. Power is NOT measured.
- Root command (not run; no sudo used): sudo perf stat -a -e power/energy-pkg/ -- sleep 10, or sudo cat /sys/class/powercap/intel-rapl*/energy_uj before and after each run.
- Load: 1-min loadavg 10.4 to 13.2 during measured runs (gate 14, 16 cores). Other agents' jobs and desktop apps were running; the GPU is shared with the desktop (renderD128 holders: gnome-shell, Discord, chrome, ptyxis). CPU runs used 4 threads, except the first S4 FP32 CPU run, which was rerun at 4 threads in S5.

## 9. Open items

1. Labelled parity set (good and defect samples, scores near 534.82) for pre_close, PatchCore and PaDiM. The current 8 snapshots are single-class.
2. PatchCore NPU versus P4-FUSION (same source IR; 0.0029 here, 344.6 there). Check FUSION's invocation and input crops.
3. PaDiM NPU compile abort in P4-FUSION not reproduced. Check whether FUSION compiled the static_ir.py copy.
4. PaDiM GPU f32 flips one mask pixel. Decide whether a 1-pixel tolerance applies (Opus).
5. threshold.json and metrics.json record scoring_device GPU, but production anom3 runs CPU only. If the threshold was calibrated on GPU FP16 scores, those scores are NaN or inf on this IR (measured here). Check the calibration source.
6. The pre_close IR's own pred_label disagrees with the production threshold on the parity set. Trace its rule.
7. The cause of the CPU FP16 ACT deviation (48.8) is not identified. The f32 hint does not fix it. Production uses the FP32 IR, so this does not block.
8. Power not measured (root command in section 8).
9. RULES plates (device lines) not checked.
10. Confirm that the InferenceModel used by the ovrun runner sets no precision hint (the S6 run assumes so).
11. The anomaly GPU f32 candidate patch is not written. It needs an Opus decision, since anom3 refuses non-CPU devices.

## 10. S9 addendum (2026-11-14): decisions 11, 5, 10; items 2, 3, 4, 6, 7, 8 recorded open

### 11. GPU + f32-hint candidate for the pre_close scorer (patch only; R/app not written)
- Candidate copy: E1/app_copy/ (cp -a of R/app, 83 MB; __pycache__ removed from the copy only).
- Patch: E1/device_map_GPU_f32_candidate.patch, labels a/app/anomaly/anom3/infer.py and b/app/anomaly/anom3/infer.py. Two edits: line 21 DEVICES = ("GPU", "CPU"); lines 46-48 guard accepts CPU and GPU. Line 58, the f32 hint for the GPU compile, is unchanged.
- Apply check: patch -p1 on a fresh copy of infer.py in a temp directory gives a file identical to E1/app_copy/anomaly/anom3/infer.py.
- Test: E1/tools/candidate_parity_test.py, exit 0 (TEST PASS). Its --selftest-flip mode injects one decision flip and the test detects it (exit 0).
- What it runs: the candidate config's outputs from the existing bench run (raw/anom_pre_close_f32hint__GPU__out.npz) against CPU (raw/anom_pre_close__CPU__out.npz) on the 8 recorded snapshot calls (16 images). It does not import infer.py. The bench IR is the production IR (model.xml and model.bin sha256 identical to the production files).
- Result: pred_score max abs diff 5.19e-4 (tolerance 4.54); anomaly_map 6.71e-4 (tolerance 4.48); pred_label and pred_mask exact. Decision flips: 0 at the production threshold 534.822723 and at the CPU recalculated threshold 534.823257.
- Weakness: all 16 CPU snapshot scores are GOOD (357.7 to 453.9; closest 453.87, margin 80.96), so decisions are checked on the GOOD side only. The candidate code itself was not run on GPU in this phase.

### 5. Threshold provenance (pre_close)
- threshold.json records device GPU; metrics.json records scoring_device GPU. Both come from dev at train.py:142 (infer.score_files(out_dir, ..., infer.DEVICES)). ir_precision "FP16" is a constant string written at train.py:162, not a measured value, so the GPU inference precision used at calibration is not recorded.
- File times (UTC): threshold.json, metrics.json, eval_list.tsv and model.xml 17:30:19-20Z; infer.py 18:37:51Z; train.py 18:38:14Z; eval.py 18:39:23Z. The calibration run predates the current scorer and trainer, so the code that produced it is not in the tree.
- Calibration set: eval_list.tsv, 19 held-out images (12 GOOD, 7 DEFECT, folder:bricks_close); all 19 PNGs on disk; all 19 stored scores finite.
- Stored threshold reproduced from the stored scores: 534.8227235 (stored 534.8227234; the difference is 6-decimal rounding in the file). Balanced accuracy 0.928571; tnr 1.0; tpr 0.857.
- The threshold is the midpoint of the gap between the highest GOOD score (506.10) and the lowest detected DEFECT score (563.54). The one missed DEFECT scored 400.41.
- CPU recalculation on the same 19 images, production infer.score_files with devices CPU: threshold 534.8232574; balanced accuracy 0.928571; tnr 1.0; tpr 0.857.
- Stored GPU vs CPU per image: max abs diff 5.8e-4 (max rel 1.1e-6). Decision flips: 0 (stored GPU vs CPU at the stored threshold) and 0 (stored vs recalculated threshold, on CPU).
- Reading: the deviation is the same size as the f32 GPU vs CPU deviation in section 2 (5.2e-4), and none is non-finite. This suggests f32-equivalent GPU scoring, not FP16 drift; it is an inference from the numbers, not a record.
- Both values are reported. The threshold is in-sample (fit and scored on the same 19 images; eval.py:7 calls it optimistic).
- Files: E1/calibration_gpu_vs_cpu.tsv (per image), E1/threshold_cpu_recalc.json.

### 10. Precision hint for ACT (runner source)
- R/app/ovrun/runner.py:176 and :233 call InferenceModel(args.model_dir, backend="openvino", device=args.device) with no precision or config argument. replay.py:128 does the same. runner.py:284 sets the --device default to GPU.
- No precision hint is set for ACT in R/app. The only R/app files that set INFERENCE_PRECISION_HINT are anomaly/anom3/infer.py (line 58) and telemetry/bench_ov.py.
- physicalai OpenVINO adapter (motion-env copy, the only one found; find_spec("physicalai") in ovrun-env returns none): _compile_config sets f32 only for device CPU on ARM machines (openvino.py:47-49, function lines 27-53). compile_model (openvino.py:106-110) receives that config. On this x86_64 laptop no hint is added for CPU or GPU, so ACT on GPU runs with OpenVINO's default precision (not measured separately). Which Python runs ovrun is not established.

### Open items (recorded; no further work per coordinator)
- 2: PatchCore NPU passes on this IR but conflicts with FUSION.
- 3: FUSION PaDiM NPU compile abort not reproduced (static copy?).
- 4: PaDiM GPU f32 flips one mask pixel (tolerance decision).
- 6: pre_close IR pred_label is 1 for all 16 CPU samples while scores are below threshold.
- 7: cause of the CPU FP16 ACT deviation (non-production) not identified.
- 8: power needs sudo and was not run. Command (section 8): sudo perf stat -a -e power/energy-pkg/ -- sleep 10. Opus asks Hassan later.

### R/app observation
- The diagnostics below (ls output appended by this call) show the laptop clock offset and the mtimes of anomaly/anom3/__pycache__ and app/tools/scan_pairs.jsonl. Origin not established.
### S9 diagnostics (ls and clock, appended by the phase 2b follow-up)
laptop local now: 2026-11-15_00:09:43_EET; utc now: 2026-11-14_22:09:43
-- anomaly/anom3/__pycache__ (local time labels):
total 16
-rw-rw-r-- 1 intel intel 14994 2026-11-14_22:54:41 infer.cpython-312.pyc
-- app/tools/scan_pairs.jsonl:
-rw-rw-r-- 1 intel intel 2324 2026-11-15_00:06:25 /home/intel/intel-junction-2026/app/tools/scan_pairs.jsonl
