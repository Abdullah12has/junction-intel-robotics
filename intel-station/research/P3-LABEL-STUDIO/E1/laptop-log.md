2026-11-14T19:28:17Z CMD nice -n 19 env OMP_NUM_THREADS=4 envs/data-env/bin/python -I work/P3-LABEL-STUDIO/E1/tools/census.py > work/P3-LABEL-STUDIO/E1/out/census_stdout.txt 2> work/P3-LABEL-STUDIO/E1/out/census_stderr.txt
exit 0
2026-11-14T19:30:00Z CMD nice -n 19 env OMP_NUM_THREADS=4 envs/data-env/bin/python -I work/P3-LABEL-STUDIO/E1/tools/build_sheets.py > work/P3-LABEL-STUDIO/E1/out/build_sheets_stdout.txt 2> work/P3-LABEL-STUDIO/E1/out/build_sheets_stderr.txt
exit 0
2026-11-14T19:34:34Z CMD nice -n 19 env OMP_NUM_THREADS=4 envs/data-env/bin/python -I work/P3-LABEL-STUDIO/E1/tools/to_jpg.py
2026-11-14T19:40:02Z CMD nice -n 19 env OMP_NUM_THREADS=4 envs/data-env/bin/python -I work/P3-LABEL-STUDIO/E1/tools/overlap.py > work/P3-LABEL-STUDIO/E1/out/overlap_stdout.txt 2> work/P3-LABEL-STUDIO/E1/out/overlap_stderr.txt
exit 0
2026-11-14T19:41:19Z CMD nice -n 19 env OMP_NUM_THREADS=4 envs/data-env/bin/python -I work/P3-LABEL-STUDIO/E1/tools/montage.py
2026-11-14T19:42:40Z CMD nice -n 19 env OMP_NUM_THREADS=4 envs/data-env/bin/python -I work/P3-LABEL-STUDIO/E1/tools/montage2.py
2026-11-14T19:43:58Z CMD nice -n 19 env OMP_NUM_THREADS=4 envs/data-env/bin/python -I work/P3-LABEL-STUDIO/E1/tools/montage3.py
2026-11-14T19:47:22Z CMD cd /home/intel/intel-junction-2026 && nice -n 19 env OMP_NUM_THREADS=4 envs/data-env/bin/python -I work/P3-LABEL-STUDIO/E1/tools/montage4.py > work/P3-LABEL-STUDIO/E1/out/montage4_stdout.txt 2> work/P3-LABEL-STUDIO/E1/out/montage4_stderr.txt
exit 0
Z19_U19_top_0_100_140_160 (1290, 970) 4
Z20_U20_top_200_230_260_290 (1290, 970) 4
Z21_U20_close_200_230_260_290 (1290, 970) 4
Z22_U24_top_600_620_640_650 (1290, 970) 4
Z23_U21_top_0_60_120_150 (1290, 970) 4
Z24_U21_close_0_60_120_150 (1290, 970) 4
2026-11-14T19:49:24Z CMD cd /home/intel/intel-junction-2026 && nice -n 19 env OMP_NUM_THREADS=4 envs/data-env/bin/python -I work/P3-LABEL-STUDIO/E1/tools/montage5.py > work/P3-LABEL-STUDIO/E1/out/montage5_stdout.txt 2> work/P3-LABEL-STUDIO/E1/out/montage5_stderr.txt
exit 0
U19 n_frames 342 dur_s 11.4 tasks ['move the brick to the left plate'] members [('6d6bdce3', 8)]
U20 n_frames 340 dur_s 11.333 tasks ['move the brick to the left plate'] members [('6d6bdce3', 9)]
U21 n_frames 339 dur_s 11.3 tasks ['move the brick to the left plate'] members [('6d6bdce3', 10)]
U24 n_frames 658 dur_s 21.933 tasks ['move the red brick to the blue plate'] members [('6d6bdce3', 13)]
Z25_U20_top_start_0_60_120_180 (1290, 970) 4
Z26_U20_close_start_0_60_120_180 (1290, 970) 4
Z27_U24_top_end (1290, 970) 4
Z28_U21_top_end (1290, 970) 4
Z29_U21_close_end (1290, 970) 4
Z30_U19_top_end (1290, 970) 4
Z31_U20_top_end (1290, 970) 4
Z32_green_plate_end_crops_U19_U20_U21 (1550, 690) 3
Z33_green_plate_U21_f150_and_end (1030, 690) 2
Z34_blue_plate_U24_f600_and_end (970, 480) 2
2026-11-14T19:51:29Z CMD cd /home/intel/intel-junction-2026 && nice -n 19 env OMP_NUM_THREADS=4 envs/data-env/bin/python -I work/P3-LABEL-STUDIO/E1/tools/montage6.py > work/P3-LABEL-STUDIO/E1/out/montage6_stdout.txt 2> work/P3-LABEL-STUDIO/E1/out/montage6_stderr.txt
exit 0
Z35_U19_gripper_f0_f60_f100 (1640, 540) 3
Z36_U20_gripper_f0_f60_f120_f190 (2070, 540) 4
Z37_U21_gripper_f0_f30_f60 (1640, 540) 3
Z38_U20_green_start_and_end_5x (1110, 450) 2
2026-11-14T19:53:26Z CMD cd /home/intel/intel-junction-2026 && date -u && ls -la data/curated && ls -la data/studio-datasets-report/V1 && ls -la data/studio-datasets/7e99a9e4-*/ (listing; read-only)
exit 0
2026-11-14T19:53:26Z NOTE  curated: MT-ALL-v2, PP-BLUE, PP-BLUE-v2, PP-BOTH, PP-GREEN, PP-GREEN-v2, S1_v0 (dirs + zips), episodes_v1.csv; V1: INVENTORY.md, labels.json, episodes.csv read (hint only). 7e99a9e4 dir mtimes 19:24Z UTC (laptop listing shows UTC+2).
2026-11-14T19:53:40Z CMD (read-only) zip listings of the 7 curated zips via python3 -I zipfile.namelist(); find -maxdepth 2 of the 7 curated dirs (no extraction)

## Run 2 (E1), commands after the first build run. All commands: cd /home/intel/intel-junction-2026; nice -n 19 env OMP_NUM_THREADS=4 envs/data-env/bin/python -I <script>

- 2026-11-14T20:02:07Z build_deliverables.py run 1 (PID 2196793). exit 0, runtime 77.45 s. Printed: episodes_total 41, physical 17, verified 38, mismatched 12, usable_training 23, usable_evaluation 12, usable_calibration_only 5, nothing 1, duplicate_entries 24; checks entries 41 groups 27 each_entry_once true; growth unchanged True; copies same view 21 of 24; mismatch rows 35. Output: out/build_stdout.txt, out/build_stderr.txt, out/build_checks.json.
- 2026-11-14T20:05:01Z tools/summarize_labels.py (PID 2252512), read-only summary of labels_episodes.jsonl. Printed rows 41, unique entries 41, trust training 23 / calibration 5 / evaluation 12 / nothing 1. Found 4 copies (a4ba6ede:0-3) with frame_check same_view False for 0-2 but trust training: fixed in run 2 by a frame verdict.
- 2026-11-14T20:06:07Z tools/offset_check.py (PID 2271412). exit 0, 50.19 s. Ffmpeg frames of a4ba6ede:0-3 and their representatives at frames 60, 150, n/2 for both cameras, pixel MAE at offsets -3..+3. Output: out/offset_check.json, out/offset_check_stdout.txt. Best offset: a4ba6ede:0 0 (top 2.34, close 4.12); :1 -1 (top 3.23, close 2.24; offset 0 gives 4.01 and 3.01); :2 0 (3.21, 3.25); :3 0 (2.47, 3.29).
- 2026-11-14T20:08:48Z tools/reenc_montage.py (PID 2329067). exit 0, 2 s. Output out/reenc_montage.jpg: rows 1-4 a4ba6ede:0-3 against representative at frame 150 (top and close); row 5 18468a09:6 against 6d6bdce3:12 at frame 708. Viewed: rows 1-4 show the same bricks and arm pose; row 5 shows the gripper over the BLUE plate.
- 2026-11-14T20:09:42Z tools/crop_trunc.py (PID 2349648). exit 0. Output out/trunc_crop_top_f708.jpg (top view crop, copy and representative). Viewed: a red brick is held between the jaws above the BLUE plate in both.
- 2026-11-14T20:09:56Z tools/summarize_checks.py (PID 2350776). Printed task text for all 41 entries, alignment flags (none), lag_best 3 frames for all 41, saturation flags (11 entries), census flags (7e99a9e4 frozen joints), grip events of the 17 representatives. Read-only.
- 2026-11-14T20:11:20Z copies of run-1 outputs to out/run1_2002Z/ and out/metrics_run1_2002Z.json (PID 2378900). exit 0.
- 2026-11-14T20:11Z to 20:12Z edit_block on tools/build_deliverables.py (9 edits): REENCODED_VISUAL and truncated constants; OFFCHK loader from out/offset_check.json; frame verdict (identical_decode / reencoded_same_scene / unverified); truncated entries set outcome carry_over, decision NONE, decision_correct unknown; originals_in_dataset and dataset notes; mismatch kinds video_reencoded, video_offset, text_conflict; metrics keys and previous/delta from out/metrics_run1_2002Z.json.
- 2026-11-14T20:12:36Z build_deliverables.py run 2 (PID 2402269). exit 0, 4.12 s. Output out/build2_stdout.txt, out/build2_stderr.txt (empty). Printed: verified 41 (17 own frames, 20 identical_decode, 4 reencoded_same_scene), copies_unverified 0, mismatched 12, task_text_mismatch 4, od10_decision_wrong 9, usable 23 / 12 / 5 / 1, incomplete_or_cut 6, text_plate_conflicts_physical 1, physical trust training 8 / evaluation 5 / calibration 3 / nothing 1, reencoded_copies 4, video_offset_entries 1, duplicate_entries 24; mismatch rows 41.
- 2026-11-14T20:13:15Z tools/check_run2.py (PID 2417430). Printed delta (verified +3, all other count deltas 0), trust counts, frame verdict counts, the truncated and a4ba6ede rows, mismatch kinds.
- 2026-11-14T20:13:44Z tools/verify_facts.py (PID 2426753). exit 0. Output out/verify_facts_stdout.txt. CHECK 1: info.json total_episodes and total_frames equal meta rows and data rows for all 9 Studio datasets. CHECK 2: raw parquet action rows of 9 copy pairs equal their representative windows (max abs diff 0.0). CHECK 3: 7e99a9e4 near-constant shoulder_pan, shoulder_lift, wrist_roll (std 0.04, 0.03, 0.01), gripper constant 16.731, elbow_flex and wrist_flex vary (std 2.29, 7.27).
- 2026-11-14T20:14:04Z tools/p17_montage.py (PID 2433256). exit 0. Output out/p17_montage.jpg (top and close at 0, 93.3, 186.7, 266.7 s of 7e99a9e4:0). Viewed: same bricks in the same places at all four times, both plates empty.
- 2026-11-14T20:15:14Z tools/summarize_curated.py (PID 2460660). Printed per-set counts: MT-ALL-v2 9 (3 issues), PP-BLUE 2 (0), PP-BLUE-v2 3 (1), PP-BOTH 3 (1), PP-GREEN 1 (1), PP-GREEN-v2 2 (2), S1_v0 4 (0); 24 episodes, 24 matched, 0 unmatched.
- 2026-11-14T20:15Z sandbox STATUS.md refreshed to IN-PROGRESS (coordinator request).
- Desktop Commander: one read_file (out/reenc_montage.jpg) and one edit_block returned "still running"; polled with get_tool_result, not re-run.
- 2026-11-14T20:25Z sandbox copies made: labels_episodes.jsonl (41 lines; sha256 0dcb15be26ffa4ff5846d38e4e3f7685ee440510c41332b536aa6ce2b82bc1d9 on both sides), datasets.json, metrics.json, mismatches.tsv, laptop-log.md. The last four were cut from one temporary packed file (out/_copy_pack_tmp.txt, deleted after use); each piece matched its laptop sha256 before it was written.
- 2026-11-14T20:25Z correction in the sandbox: an earlier write of labels_episodes.jsonl held 12 of the first 20 laptop lines (6d6bdce3:2 to :9 missing). Lines 13 to 41 were added from the laptop read and checked line by line by sha256.
- 2026-11-14T20:25Z trust_report.md: the header sentence now says evidence files (frames, montages, out/ JSON, tools/) stay on the laptop; same edit in both copies, sha256 4b47f82ad4f1884e4756556ca00f52ff5afee8a68d4f60287e308cf1206f049e on both sides. edit_block returned the file's first lines instead of a confirmation; the result was checked by sha256 and the edit was applied once.
