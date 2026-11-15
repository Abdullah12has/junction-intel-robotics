# P2-B-DET2 V1 laptop log (2026-11-14, UTC; laptop clock)

Runs use `envs/data-env/bin/python -I -B`, `nice -n 10`. No robot port, camera or live capture was opened. Live stills are read-only copies from `logs/detlive/` (Opus-taken, 153426).

## Commands and key outputs

1. Test suite (after the live-frame fixes, 15:56:36Z):
   `cd app/anomaly/det2_v1 && python3 -I -B tests/test_inspector.py`
   -> 19 PASS, `failures 0` (16 original + `test_live_153426_no_hands_and_no_rail_bricks`, `test_live_153426_top_shapes_and_verdicts`, `test_live_153426_close_tape_and_clean_bricks`). Re-run at 15:58:17Z: `failures 0`.

2. Seam profile of the 5029 px top blob (`tools/prof_v1.py`, 15:45Z): red 2x2 and white 2x4 join near y 238 to 242; white 2x4 and white 2x2 join near y 298 to 304 with no colour difference; the close-view blob 30879 px is red + white 2x4 + white 2x2.

3. Neck-split debug (`tools/dbg_v1.py`, 15:49Z): the old distance-peak split returned None for the 5092 px blob; white core 3046 px got three seeds 12 px apart (one 2x4 over-seeded).

4. Live frames, V1 final (`tools/live2_v1.py`, 15:56Z):
   `hands 0 plates_top 2 partial_close 3 rejected_close {'plate_or_gripper': 1, 'small': 0}`
   top: red 1421 px 2x2 UNCERTAIN; white 2296 px not_2x2 conf 0.72 UNCERTAIN; white 1232 px 2x2 UNCERTAIN; yellow 1396 px 2x2 UNCERTAIN.
   close: red 9588 px tape 0.0098 UNCERTAIN; white 10613 px tape 0.0127 UNCERTAIN; white 10474 px tape 0.0017 UNCERTAIN (partial); yellow 4724 px tape 0.1383 CONFIDENT_DEFECT.

5. E2 on the same frames (`tools/live_e2_v1.py /home/intel/intel-junction-2026/app/anomaly/det2_e2`, 15:5x Z):
   `hands 0 rejected {'plate_or_gripper': 0, 'small': 0} partial 3`; one top brick (white 1396 px 2x2 UNCERTAIN, the yellow brick); no close bricks.

6. Dataset harness (`tools/eval_v1.py APP OUT_JSON`), runs after2 to after5:
   - after5 (final): refused raw and JPEG 2 each (e07_start, e09_end); top verdicts raw UNCERTAIN 81, JPEG UNCERTAIN 79 + CONFIDENT_DEFECT 3; close verdicts raw GOOD 18 / UNCERTAIN 46 / DEFECT 12, JPEG GOOD 17 / UNCERTAIN 56 / DEFECT 12; top conf_defect_on_2x2 0; top count-match raw 3 evaluable, 11 skipped; raw scan mean 48.7 ms; errors none.
   - E2 baseline (`eval/baseline_e2.json`): refused raw 3, JPEG 5; close DEFECT raw 3, JPEG 2.
   - Output files: `eval/after_v1.json` ... `eval/after5_v1.json`; comparisons `eval/cmp_after2.txt`, `eval/cmp_after3.txt`, `eval/cmp_after5.txt`.

7. Per-frame comparison (`tools/cmp_v1.py EVAL_DIR FILE`): the close DEFECT diff list is in `eval_summary.json` in the sandbox folder. Labels checked: `/home/claude/kb/work/P2-B-DET2/E2/labels/compare_opus_vs_E2.tsv` (sandbox).

8. E2 vs V1 on the e11_start raw PNG (`/home/intel/intel-junction-2026/work/P2-B-DET2/E2/samples/6d6bdce3_e11_start_243722064316.png`): E2 `segment_close` gives white 4420, red 10958 and one plate rejection; V1 before the plate-chroma rule gave three orange pieces (5088, 5270, 9270 px) from the blue-plate blob. Cause of the extra close bricks; fixed for blobs with at least 30% blue or green chroma, not fully resolved for the rest.

9. Static API check of `app/anomaly/det2_v1/scan_once.py`: imports `inspector as I`, calls `I.scan_full(top, close)`, catches `I.HandsPresent`. All present in V1. The script was not executed (it opens the camera and robot path).

10. Fixtures: `tests/fixtures/live_top_153426.jpg` (40035 bytes) and `tests/fixtures/live_close_153426.jpg` (34300 bytes), copied from `logs/detlive/`.

## Not run

- OpenVINO static-reshape port and CPU/GPU/NPU latency: not started.
- Upside-down detection: not implemented (no validated signal).
