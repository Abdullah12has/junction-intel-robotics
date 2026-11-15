# P4-FUSION E1: fusion-v1 (CHANGES)

**Status: NOT ADOPTED.** The integration patch is delivered with the switch off (`app/anomaly/fused_config.json`, `"enabled": false`). Stage `fusion-v1`, metrics in `metrics.json` (utc 2026-11-14T21:21Z). Times are UTC.

## 1. What was evaluated

- **Crops:** 32 close-view brick crops after the eye check (46 built; 14 excluded by eye: red gripper fingers, wrong-brick and mat-only crops).
- **Labels:** 30 GOOD and 2 DEFECT. Both DEFECT crops are yellow: any_1740 ep_0006 (verified) and any_1740 ep_0017 (unverified).
  - Primary set: verified labels only, 31 crops (30 GOOD, 1 DEFECT).
  - Sensitivity set: all 32 crops.
- **Colours:** only yellow has DEFECT crops. Blue (3 GOOD), orange (10 GOOD), red (11 GOOD) and white (2 GOOD) have none, so no decision on those colours is assessable.
- **Models:** Anomalib 2.7.0 PatchCore (wide_resnet50_2, layers 2 and 3, coreset) and PaDiM (resnet18, layers 1 to 3), trained on GOOD crops only at 256 px.
- **Validation:** leave-one-episode-out (LOEO). Studio crops are grouped by participant.
- **Fold threshold:** the maximum anomaly score of the fold's training GOOD crops.
- **Fusion candidates:**
  - AND: tape CONFIDENT_DEFECT and anomaly score above the fold threshold.
  - OR: either.
  - LR: logistic regression on the standardised [tape score, anomaly score], class-balanced, fitted on the training folds only.
- **Metrics:** AUROC, AUPR and F1 at the fold threshold. 95% percentile intervals come from a class-stratified bootstrap over crops (2000 draws, seed 0). The intervals are optimistic because crops from one episode are correlated.

## 2. Adoption rule and result

**Rule** (`metrics.json`, `adoption_rule`): a fused candidate is adopted only if all of the following hold.
- On every colour with both classes, its F1 is at least that of the best single method on the same crops (for LR, its AUROC is at least the best single score as well).
- Its overall F1 (and AUROC for LR) is strictly above the best single method.
- At least 5 DEFECT episodes exist.

**Result: NOT ADOPTED on both sets.**
- Primary: 1 DEFECT episode (5 required). Best candidate overall: PatchCore OR, F1 0.500, below the tape verdict's 0.667.
- Sensitivity: 2 DEFECT episodes (5 required). Best candidate overall: PatchCore LR, F1 0.800 and AUROC 1.000. The tape verdict has F1 0.800 and the tape score has AUROC 1.000, so the candidate ties and is not strictly better.
- No candidate passes every check.

## 3. Before and after on the same crops

"Before" is the DET2 tape verdict and each Anomalib model alone. "After" is each fused candidate.

**Sensitivity set (32 crops: 30 GOOD, 2 DEFECT)**

| Method | AUROC [95%] | AUPR | F1 [95%] | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|
| DET2 tape verdict (before) | 0.983 [0.95, 1.00] | 1.000 | 0.800 [0.57, 1.00] | 2 | 1 | 0 | 29 |
| DET2 tape score, continuous (before) | 1.000 [1.00, 1.00] | 1.000 | not applicable | | | | |
| PatchCore alone (before) | 0.533 [0.07, 0.97] | 0.162 | 0.000 | 0 | 1 | 2 | 29 |
| PaDiM alone (before) | 0.150 [0.00, 0.40] | 0.054 | 0.000 | 0 | 11 | 2 | 19 |
| PatchCore AND tape (after) | not applicable | not applicable | 0.000 | 0 | 0 | 2 | 30 |
| PatchCore OR tape (after) | not applicable | not applicable | 0.667 [0.44, 1.00] | 2 | 2 | 0 | 28 |
| PatchCore LR (after) | 1.000 [1.00, 1.00] | 1.000 | 0.800 [0.57, 1.00] | 2 | 1 | 0 | 29 |
| PaDiM AND tape (after) | not applicable | not applicable | 0.000 | 0 | 0 | 2 | 30 |
| PaDiM OR tape (after) | not applicable | not applicable | 0.250 [0.19, 0.36] | 2 | 12 | 0 | 18 |
| PaDiM LR (after) | 1.000 [1.00, 1.00] | 1.000 | 0.800 [0.57, 1.00] | 2 | 1 | 0 | 29 |

**Primary set (31 crops: 30 GOOD, 1 verified DEFECT)**

| Method | AUROC [95%] | AUPR | F1 [95%] | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|
| DET2 tape verdict (before) | 0.983 [0.95, 1.00] | 1.000 | 0.667 [0.40, 1.00] | 1 | 1 | 0 | 29 |
| PatchCore alone (before) | 0.900 [0.77, 1.00] | 0.250 | 0.000 | 0 | 1 | 1 | 29 |
| PaDiM alone (before) | 0.300 [0.13, 0.47] | 0.046 | 0.000 | 0 | 11 | 1 | 19 |
| PatchCore AND tape (after) | not applicable | not applicable | 0.000 | 0 | 0 | 1 | 30 |
| PatchCore OR tape (after) | not applicable | not applicable | 0.500 [0.29, 1.00] | 1 | 2 | 0 | 28 |
| PaDiM AND tape (after) | not applicable | not applicable | 0.000 | 0 | 0 | 1 | 30 |
| PaDiM OR tape (after) | not applicable | not applicable | 0.143 [0.11, 0.22] | 1 | 12 | 0 | 18 |
| PatchCore LR, PaDiM LR (after) | not assessable | | | | | | |

The LR rows are not assessable on the primary set: the only DEFECT episode is held out in its fold and has no training positive.

**Per colour, F1 (only colours with DEFECT crops are assessable)**

| Colour, sensitivity set (GOOD / DEFECT) | Tape verdict | PatchCore alone | PaDiM alone | PatchCore OR | PatchCore LR | PaDiM OR | PaDiM LR |
|---|---|---|---|---|---|---|---|
| yellow (4 / 2) | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.800 | 1.000 |

Red has one tape false positive: S_6d6bdce3-11_U22 start (labelled GOOD, tape score 0.0839, CONFIDENT_DEFECT). No other colour has a DEFECT crop, so the per-colour adoption check cannot be passed for them.

## 4. Held-out confusion table (real crops)

Overall LOEO decisions, from `eval/confusion_loeo.tsv`. TP is DEFECT called DEFECT; FN is DEFECT called GOOD; FP is GOOD called DEFECT; TN is GOOD called GOOD. Per-colour rows are in the same file.

| Set | Method | TP | FN | FP | TN |
|---|---|---|---|---|---|
| sensitivity (2 DEFECT, 30 GOOD) | tape verdict | 2 | 0 | 1 | 29 |
| sensitivity | PatchCore | 0 | 2 | 1 | 29 |
| sensitivity | PaDiM | 0 | 2 | 11 | 19 |
| sensitivity | PatchCore LR | 2 | 0 | 1 | 29 |
| sensitivity | PatchCore OR | 2 | 0 | 2 | 28 |
| sensitivity | PatchCore AND | 0 | 2 | 0 | 30 |
| primary (1 DEFECT, 30 GOOD) | tape verdict | 1 | 0 | 1 | 29 |
| primary | PatchCore | 0 | 1 | 1 | 29 |
| primary | PaDiM | 0 | 1 | 11 | 19 |
| primary | PatchCore OR | 1 | 0 | 2 | 28 |
| primary | PatchCore AND | 0 | 1 | 0 | 30 |

## 5. Device agreement (Anomalib IR, 32 crops)

| Model | Device | Status | Max abs diff vs CPU | Latency per crop |
|---|---|---|---|---|
| PatchCore | CPU | ok | 0 | 102.1 ms |
| PatchCore | GPU | ok | 0.00049 | 25.7 ms |
| PatchCore | NPU | scores disagree | 344.6 | 44.7 ms |
| PaDiM | CPU | ok | 0 | 35.3 ms |
| PaDiM | GPU | ok | 0.00018 | 124.8 ms |
| PaDiM | NPU | compile aborted (LLVM ERROR: Failed to infer result type(s), IE.Interpolate) | not applicable | not applicable |

Latencies are from the 32-crop device check on the laptop, not a benchmark.

Thresholds in `threshold.json` are in-sample: PatchCore 583.217 and PaDiM 468.473, each the maximum over the 30 training GOOD crops.

## 6. Integration patch (`fusion.patch`)

**Base:** a copy of R/app taken 2026-11-14T21:34:42Z, at commit ddc05924 (`app_base_ddc05924/`). **Patched copy:** `app/`. The patch is 10 files, 1586 lines.

- **`app/anomaly/fused.py` (new):**
  - `config()` and `crop_of()`, which give the same crop as `save_crop` for the same bbox and pad.
  - An IR scorer that uses the anom3 preprocessing, loaded by file path so no module name enters `sys.path`.
  - `score()`: returns None when the flag is off, when there is no crop, or when the evidence cannot be computed. In the last case it logs "anomaly unavailable, tape verdict stands".
  - `fused_verdict()` (AND, OR or LR) and `effective_verdict()`.
- **`app/anomaly/fused_config.json` (new):** `enabled` false; model patchcore; rule LR; device CPU; size 256; IR and LR parameter paths under `R/work/P4-FUSION/E1/models`.
- **`app/tools/sort_loop.py`:**
  - `fused.py` is loaded by file path.
  - `decide(pre, shape, views, anomaly=None)`: with no evidence it is unchanged, including the reason text. With evidence the fused rule sets the close verdict, `pre` is not modified, and the reason reads `fused_<RULE>:<tape verdict>,<anomaly_over|anomaly_under>`.
  - `cycle()` sets `EP["anomaly"]` after the pre-grasp close view, and the fast path uses `effective_verdict()`.
- **`app/tools/sort_any.py`:**
  - `grasp()` records `EP["anomaly"]` through `_anomaly_for()` right after `EP["pre_close"]`.
  - `_anomaly_for()` crops the image that the verdict's bbox refers to. For a `diff_first_frame` verdict that is the episode's first close frame; otherwise it is the close frame.
  - The station condition and both `decide()` calls use the evidence.
- **Tests:** `app/tools/tests/test_fusion_replay.py` (new, 5 tests), with `fixtures/fusion_v1/replay_fixture.tsv` and `fusion_folds.json`, MANIFEST rows, the `run_tests.py` module list, and one README line.
- **Episode record:** `EP["anomaly"]` is saved in `episode.json` through the existing `default=str` dump.

## 7. Tests

| Tree | Tests | PASS | FAIL |
|---|---|---|---|
| Snapshot ddc05924, without patch | 36 | 35 | 1: test_scan_candidates_exclude_plates_1754Z (pre-existing) |
| Snapshot ddc05924, with patch | 41 | 40 | same 1 |
| Live R/app at 572e8fff, without patch | 37 | 37 | 0 |
| Live R/app at 572e8fff, with patch | 42 | 42 | 0 |

- The 5 new tests pass on both trees:
  - flag off: `decide()` is unchanged without evidence;
  - a failing IR falls back to the tape verdict;
  - the fused rule sets the close verdict and does not mutate its input;
  - LOEO replay for PatchCore and PaDiM under AND, OR and LR.
- The replay reproduces every LOEO decision on the 32 crops. Decisions called DEFECT: PatchCore AND 0, OR 4, LR 3; PaDiM AND 0, OR 14, LR 3.
- `det2` `test_inspector` returns rc 0 in every run.
- `fusion.patch` applies with rc 0 to a copy of the snapshot, and the result equals the patched copy. It also applies with rc 0 to a copy of the live R/app taken at 2026-11-14T21:36:56Z. Reversing the patch on that state is clean.
- Live state: the upstream commit at 572e8fff passes the suite without the patch, including the test that failed on the snapshot.

## 8. Caveats for Opus

1. **Not adopted.** The switch stays off. Enabling needs a new evaluation with at least 5 DEFECT episodes; only 2 exist.
2. **Crop type mismatch.** The evaluated crops are the close frames that the older pre-close verdict judged. Since the close-view localisation commit, the verdict judges the picked brick on the episode's first close frame (`diff_first_frame`). The patch scores the crop that verdict used, but that crop type was not evaluated.
3. **Held views unchanged.** The fused rule replaces only the close-view verdict. DEFECT from held views (`tape_held_r*`) still decides DEFECT. The LOEO evaluation does not model held views.
4. **Trade-off in the rules.** AND clears the red tape false positive (S_6d6bdce3-11_U22 start) but also misses both DEFECT crops. OR adds false positives: 2 with PatchCore, 12 with PaDiM.
5. **Optimistic intervals.** Crops from one episode are correlated, so the 95% intervals are too narrow. Fold thresholds are out-of-fold; the `threshold.json` values are in-sample.
6. **Unvalidated fit.** `models/fusion_params.json` is the deployment fit on all 32 crops and is marked validated false. Do not enable with it.
7. **NPU unusable.** The NPU scores for PatchCore disagree with CPU (344.6), and the PaDiM compile aborts. Any enablement should use CPU, or GPU (which agrees within 0.0005).
8. **No hardware, no live test of the enabled path.** Nothing touched the robot, cameras, serial ports or the motion server. No test runs the OpenVINO scoring inside the loop; only the missing-IR fallback and the recorded decisions are tested. A bench run with the flag on and no motion is needed before enabling. OpenVINO must be available in the loop's environment (anomalib-env has 2026.4.1).
9. **Small crop set.** Colour matching failed for 59 of 105 frames (`no_colour_match`), which limits the crop count to 32.
10. **Moving base.** The patch base is the 21:34:42Z snapshot. R/app moved to 572e8fff by 21:36:56Z. The patch applies there and passes its suite, but it must be rebased if the tree moves again.

## 9. Files (R/work/P4-FUSION/E1)

- `metrics.json` (stage fusion-v1): `decision`, `best_candidate_overall`, `before_after`, `label_sets`, `tables`, `device_check`.
- `eval/`: `confusion_loeo.tsv`, `table_methods.tsv`, `roc_points.tsv`, `roc_primary.png`, `roc_sensitivity.png`, `crop_table.tsv`, `fusion_folds.json`, `replay_fixture.tsv`, `device_check.json`, `scores.tsv`, `fits.json`.
- `models/`: `fusion_params.json` (deployment fit, validated false), `patchcore/all_good`, `padim/all_good`.
- `fusion.patch`, `app/` (patched copy), `app_base_ddc05924/` (base), `app_base/` (original copy at 96ee440d, superseded), `logs/`.
- Scratch, safe to delete: `verify/`, `verify_live/`, `live_base/`, `app_on_96ee440d/` and `fusion_on_96ee440d.patch` (built on 96ee440d, not valid for the current tree).
