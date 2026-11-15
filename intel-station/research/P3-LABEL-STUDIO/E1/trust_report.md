# Studio datasets: trust report (P3-LABEL-STUDIO, E1)

Stage studio-labels-v1, run 2, 2026-11-14T20:12Z (run 1 at 20:02Z; metrics.json "previous"). Laptop copy: R/work/P3-LABEL-STUDIO/E1/trust_report.md. Machine-readable: labels_episodes.jsonl (41 rows), datasets.json (9 Studio, 7 curated, V1 hint), mismatches.tsv (41 rows), metrics.json. Evidence files (frames, montages, out/ JSON, tools/) stay on the laptop in R/work/P3-LABEL-STUDIO/E1/ and are not copied here; paths are given below.

Trust levels: training = one entry per physical episode may train; evaluation = outcome, OD-10 decision or instruction check only, never training; calibration = joint and gripper sanity and grasp events only, no placement label; nothing = do not use.

## 1. Bottom line
- 41 Studio entries in 9 datasets are 17 physical episodes. Only 6d6bdce3 (16 episodes, P1 to P16) and 7e99a9e4 (one raw batch, P17) hold original recordings. The other seven Studio datasets (24 entries) are copies or windows of 6d6bdce3 episodes and add no physical episode.
- Training: P1, P2, P3, P4, P5, P7, P9, P16 (8 physical episodes; 23 entries counting copies).
- Evaluation only: P6, P8, P13, P14, P15 (5 physical; 12 entries). Their recorded task text or OD-10 decision disagrees with the frames; the physical outcome is verified.
- Calibration only: P10, P11, P12 (grasped and held, never placed) plus two truncated copies of P3 (5 entries).
- Nothing: P17 (7e99a9e4:0), a raw batch with placeholder text and no pick.
- Verified: 41 of 41 entries. 17 by my own frame inspection; 20 copies pixel-identical to their representative; 4 copies (a4ba6ede:0 to :3) re-encoded, checked by eye and by pixel offset.
- Mismatched: 12 entries. Task text: 4 (P1 representative, P13, P14, P15). OD-10 decision: 9 entries (P6, P8, P13 and their copies). P13 is in both groups.

## 2. Datasets
| dataset | entries | originals | training | evaluation | calibration | nothing | what it holds |
|---|---|---|---|---|---|---|---|
| 6d6bdce3 | 16 | 16 (P1 to P16) | 7 | 6 | 3 | 0 | original recording; V1 counted 11 |
| 18468a09 | 9 | 0 | 6 | 2 | 1 | 0 | windows of P7, P2, P9, P5, P1, P4, P3, P6, P8 |
| e2eae6c8 | 3 | 0 | 2 | 0 | 1 | 0 | copies or windows of P1, P4, P3 |
| 800d7dc2 | 3 | 0 | 2 | 1 | 0 | 0 | copies of P4, P3, P8 |
| a4ba6ede | 4 | 0 | 4 | 0 | 0 | 0 | re-encoded copies of P7, P2, P9, P5 |
| d2fd0a80 | 2 | 0 | 2 | 0 | 0 | 0 | copies of P4, P3 |
| ba9eafba | 2 | 0 | 0 | 2 | 0 | 0 | copies of P6, P8 |
| 51100357 | 1 | 0 | 0 | 1 | 0 | 0 | copy of P8 |
| 7e99a9e4 | 1 | 1 (P17) | 0 | 0 | 0 | 1 | raw batch, 280.2 s |

Facts common to all nine (datasets.json, verify_facts CHECK 1): fps 30; two cameras, serial 243622061421 = top and 243722064316 = close, AV1, 640x480, 30/1; joint names shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos; units not stated in info.json. info.json total_episodes and total_frames equal the meta and data row counts in every dataset. Duration totals: 6d6bdce3 338.6 s (16 episodes, 11.3 to 43.5 s); 18468a09 185.1 s; e2eae6c8 83.1 s; a4ba6ede 85.6 s; 800d7dc2 74.9 s; d2fd0a80 54.2 s; ba9eafba 30.2 s; 51100357 20.8 s; 7e99a9e4 280.2 s.

## 3. The 17 physical episodes
Grasp is a gripper close (to_lo) and release an open (to_hi) from the census gripper events; times are seconds in the representative entry.

| P | representative | brick | what the frames show | plate | OD-10 | decision correct | task text | trust |
|---|---|---|---|---|---|---|---|---|
| P1 | 6d6bdce3:7 | orange 2x2, no tape | grasp 11.7; released into BLUE at 40.4 | BLUE | BLUE | yes | representative says "inspection spot" (mismatch); copies say "blue plate" | training (via 18468a09:4); representative evaluation |
| P2 | 6d6bdce3:4 | red 2x2 marked 2, no tape | regrasps at 7.1 and 9.0; put-down on mat, release 21.3 | SPOT | n/a | n/a | match | training |
| P3 | 6d6bdce3:12 | red 2x2 marked 2, no tape | grasp 14.0; released into BLUE at 24.1 | BLUE | BLUE | yes | match | training |
| P4 | 6d6bdce3:11 | red 2x2, no tape | grasp 9.7; released into BLUE at 20.0 | BLUE | BLUE | yes | match | training |
| P5 | 6d6bdce3:6 | red 2x2, no tape | grasp 5.6; put-down on mat, release 14.2 | SPOT | n/a | n/a | match | training |
| P6 | 6d6bdce3:14 | red 2x2, no tape | grasp 7.6; released into GREEN at 12.1 | GREEN | BLUE | no | match | evaluation |
| P7 | 6d6bdce3:3 | yellow 2x2, no tape | grasp 8.2; put-down top-centre, release 20.4 | SPOT | n/a | n/a | match | training |
| P8 | 6d6bdce3:15 | red 2x2, no tape | grasp 11.9; released into GREEN (right edge) at 16.4 | GREEN | BLUE | no | match | evaluation |
| P9 | 6d6bdce3:5 | yellow 2x2, no tape | grasp 7.6; put-down lower right of cluster, release 18.3 | SPOT | n/a | n/a | match | training |
| P10 | 6d6bdce3:0 | blue 2x2, no tape | grasp 7.9; held to the end (22.4), never placed | none | BLUE | n/a | "inspection spot": not placed | calibration |
| P11 | 6d6bdce3:1 | red 2x2 marked 2, no tape | grasp 2.5; held to 19.3, never placed | none | BLUE | n/a | not placed | calibration |
| P12 | 6d6bdce3:2 | red 2x2, no tape | grasp 3.6; held to 18.1, never placed | none | BLUE | n/a | not placed | calibration |
| P13 | 6d6bdce3:8 | white 2x2, no tape | starts holding a brick; released onto GREEN beside the red 2x2 at 5.0 | GREEN | BLUE | no | "left plate" (BLUE): mismatch | evaluation |
| P14 | 6d6bdce3:9 | white 2x4 | starts holding; released onto the GREEN right rim at 6.3 | GREEN | GREEN (2x4 = DEFECT) | yes | "left plate": mismatch | evaluation |
| P15 | 6d6bdce3:10 | blue 2x2 with silver/grey tape | starts holding; released into GREEN at 4.7 | GREEN | GREEN (tape = DEFECT) | yes | "left plate": mismatch | evaluation |
| P16 | 6d6bdce3:13 | red 2x2, no tape | grasp 10.8; released into BLUE at 17.8, brick stays in the plate | BLUE | BLUE | yes | match | training |
| P17 | 7e99a9e4:0 | none | no pick (section 5) | none | n/a | n/a | placeholder "raw batch - relabel per pick" | nothing |

## 4. Per-entry trust (41 entries)
| entry | P | trust | pixel check vs representative | note |
|---|---|---|---|---|
| 6d6bdce3:0 | P10 | calibration | own frames | held, never placed |
| 6d6bdce3:1 | P11 | calibration | own frames | held, never placed |
| 6d6bdce3:2 | P12 | calibration | own frames | held, never placed |
| 6d6bdce3:3 | P7 | training | own frames | spot put-down |
| 6d6bdce3:4 | P2 | training | own frames | spot put-down, two regrasps |
| 6d6bdce3:5 | P9 | training | own frames | spot put-down |
| 6d6bdce3:6 | P5 | training | own frames | spot put-down |
| 6d6bdce3:7 | P1 | evaluation | own frames | text "inspection spot" vs BLUE; copies say "blue plate" |
| 6d6bdce3:8 | P13 | evaluation | own frames | "left plate" vs GREEN; OD-10 BLUE, decision GREEN is wrong |
| 6d6bdce3:9 | P14 | evaluation | own frames | "left plate" vs GREEN; OD-10 GREEN (2x4) is right |
| 6d6bdce3:10 | P15 | evaluation | own frames | "left plate" vs GREEN; OD-10 GREEN (tape) is right |
| 6d6bdce3:11 | P4 | training | own frames | BLUE, decision correct |
| 6d6bdce3:12 | P3 | training | own frames | BLUE, decision correct |
| 6d6bdce3:13 | P16 | training | own frames | BLUE, decision correct |
| 6d6bdce3:14 | P6 | evaluation | own frames | GREEN; OD-10 BLUE, decision wrong |
| 6d6bdce3:15 | P8 | evaluation | own frames | GREEN; OD-10 BLUE, decision wrong |
| 7e99a9e4:0 | P17 | nothing | own frames | raw batch, no pick |
| 18468a09:0 | P7 | training | identical, window 113:639 of 6d6bdce3:3 | |
| 18468a09:1 | P2 | training | identical, window 21:663 of :4 | |
| 18468a09:2 | P9 | training | identical, window 46:578 of :5 | |
| 18468a09:3 | P5 | training | identical, window 0:454 of :6 | |
| 18468a09:4 | P1 | training | identical, window 31:1241 of :7 | text "orange brick to the blue plate" (matches BLUE) |
| 18468a09:5 | P4 | training | identical, window 57:630 of :11 | |
| 18468a09:6 | P3 | calibration | identical, window 0:711 of :12 | cut 0.4 s before release; carry_over |
| 18468a09:7 | P6 | evaluation | identical, window 1:387 of :14 | GREEN; OD-10 wrong |
| 18468a09:8 | P8 | evaluation | identical, window 0:520 of :15 | GREEN; OD-10 wrong |
| 51100357:0 | P8 | evaluation | identical, full copy of :15 | GREEN; OD-10 wrong |
| 800d7dc2:0 | P4 | training | identical, window 57:850 of :11 | |
| 800d7dc2:1 | P3 | training | identical, full copy of :12 | |
| 800d7dc2:2 | P8 | evaluation | identical, full copy of :15 | GREEN; OD-10 wrong |
| a4ba6ede:0 | P7 | training | re-encoded: MAE 2.34 top, 4.12 close at offset 0 | same actions; same scene by eye at frames 150 and 352 |
| a4ba6ede:1 | P2 | training | re-encoded: MAE 4.01 top, 3.01 close at offset 0; 3.23 and 2.24 at offset -1 | video one frame earlier than its rows |
| a4ba6ede:2 | P9 | training | re-encoded: MAE 3.21 top, 3.25 close at offset 0 | same scene by eye |
| a4ba6ede:3 | P5 | training | re-encoded: MAE 2.47 top, 3.29 close at offset 0 | same scene by eye |
| ba9eafba:0 | P6 | evaluation | identical, window 1:387 of :14 | GREEN; OD-10 wrong |
| ba9eafba:1 | P8 | evaluation | identical, window 0:520 of :15 | GREEN; OD-10 wrong |
| d2fd0a80:0 | P4 | training | identical, window 57:850 of :11 | |
| d2fd0a80:1 | P3 | training | identical, full copy of :12 | |
| e2eae6c8:0 | P1 | training | identical, window 31:1241 of :7 | text "orange brick to the blue plate" |
| e2eae6c8:1 | P4 | training | identical, window 57:630 of :11 | |
| e2eae6c8:2 | P3 | calibration | identical, window 0:711 of :12 | cut 0.4 s before release; carry_over |

Counts: training 23 entries (8 physical), evaluation 12 (5), calibration 5 (3), nothing 1 (1). For training, count each physical episode once; for P1 use 18468a09:4 (its text matches), not the representative. The MAE values on the re-encoded rows are means over frames 60, 150 and n/2 per camera (out/offset_check.json).

## 5. Evidence and checks
- Pixel copies: 20 copies have MAE 0.0 at frames 60 and n/2 on both cameras against their representative (out/build_checks.json, copy_frame_checks, run 2).
- Actions: nine copy pairs checked on the raw parquet, max absolute difference 0.0 (out/verify_facts_stdout.txt, CHECK 2). The census found all 24 copies by exact row match.
- Re-encoded copies: pixel MAE at offset 0 is 2.3 to 4.1 while the actions are identical. Offsets -3 to +3 do not give a lower MAE, except a4ba6ede:1 (best at -1) (out/offset_check.json). Same scene by eye, rows 1 to 4 of out/reenc_montage.jpg (frame 150, both cameras). Close-camera detail differs slightly during fast motion (a4ba6ede:0, frame 352).
- Truncated copies 18468a09:6 and e2eae6c8:2 end at 23.7 s; the representative releases at 24.1 s. At frame 708 the brick is between the jaws above the BLUE plate in both (out/trunc_crop_top_f708.jpg). Outcome carry_over; decision unknown.
- Timestamps: video time = from_timestamp + frame_index / 30. frame_index is contiguous, video span and meta length match the rows for all 41 entries (census flags).
- Joint sanity: action leads observation.state by 3 frames in all 41 entries; all actions inside [-100, 100] (body) and [0, 100] (gripper). Eleven entries reach a normalized limit (shoulder_lift at -100, or wrist_flex or elbow_flex at +100); flagged only, not a trust reason.
- 7e99a9e4: four joints near-constant (std 0.01 to 0.04), gripper constant at 16.731; elbow_flex and wrist_flex move (std 2.29 and 7.27). Frames at 0, 93, 187 and 267 s, top and close: the same bricks in the same places, both plates empty (out/p17_montage.jpg). No pick, so trust nothing.
- Growth: 7e99a9e4 unchanged between the census (19:28Z) and now (20:02Z): both mp4 sizes and 8405 data rows (out/build_checks.json, growth_7e99a9e4).

## 6. Curated sets (read only; 7 sets, 24 episodes)
- All 24 curated episodes match a Studio entry exactly (action MAE 0.0; out/curated_check.json).
- Per set (episodes; episodes with issues): MT-ALL-v2 9 (3); PP-BLUE 2 (0); PP-BLUE-v2 3 (1); PP-BOTH 3 (1); PP-GREEN 1 (1); PP-GREEN-v2 2 (2); S1_v0 4 (0).
- Eight curated labels are wrong. Six say "move the red brick to the green plate" for a clean red 2x2, where OD-10 says BLUE (P6 in MT-ALL-v2:7 and PP-GREEN-v2:0; P8 in MT-ALL-v2:8, PP-BOTH:2, PP-GREEN:0 and PP-GREEN-v2:1). Two are truncated copies labelled as placements (MT-ALL-v2:6, PP-BLUE-v2:2). Do not use the curated labels as OD-10 ground truth.
- episodes_v1.csv marks six rows DROP (episodes 7, 8, 9, 10, 13, 14). Five of the reasons are motion or rest based. Episode 13 (P16) says "no brick grasped", which is wrong: grasp at 10.8 s and release on BLUE at 17.8 s.

## 7. V1 report (hint only; no label taken from it)
- V1 counted 11 episodes for 6d6bdce3 at 13:35Z; there are 16 now.
- V1 release plates for episodes 3 to 10 (SPOT, SPOT, SPOT, SPOT, BLUE, GREEN, GREEN, GREEN) agree with the frames.
- V1 ep 8 says the red brick is on the green plate; the released brick is the white 2x2 (the red 2x2 was already there). V1 ep 10 did not record the silver or grey tape.

## 8. Open issues and limits
1. Orientation is "not recorded" for most bricks; the P3-LABEL-OURS values (upright, upside_down, leaning, on_side) are not filled. Not used for trust.
2. info.json states no joint units. The observed ranges fit the normalized SO-101 units in P3-COMMON (body -100..100, gripper 0..100). Not checked against hardware here.
3. decision_correct is n/a for SPOT put-downs (OD-10 defines only BLUE and GREEN) and unknown for the two truncated entries.
4. P13, P14 and P15 start while holding a brick (hidden between the jaws behind the wrist mount), so their task text cannot be checked at the start.
5. The visual labels of the 17 representatives come from frame viewing done earlier (zoom/Z1 to Z39, sheets/U01 to U27). In this run I re-checked the census timings, the copies and P17, not every visual label again. Labels come from sampled frames (start, grasp, release, end, zoom crops), not every frame.
6. Re-encoded copies (a4ba6ede) are not byte-identical: count each physical episode once.
7. P1 text conflict: 6d6bdce3:7 says "inspection spot", while the same episode in 18468a09:4 and e2eae6c8:0 says "blue plate". The BLUE placement is verified.
8. The 6d6bdce3 name in V1 INVENTORY.md is stale ("Smoke test - inspection skill - 5 episodes"); not used.

## 9. Metrics, run 2 against run 1 (metrics.json previous and delta)
- verified 38 to 41 (+3, improved). Run 1 counted a copy as verified only when its pixel MAE was under 4.0 at two frames; a4ba6ede:0 to :2 failed that and a4ba6ede:3 passed. Run 2 adds a verdict for the re-encoded copies by eye and by pixel offset: 17 own + 20 identical + 4 re-encoded.
- mismatched 12 to 12; task_text_mismatch 4; od10_decision_wrong 9; usable 23 / 12 / 5 / 1; duplicate entries 24. No trust level changed.
- mismatches.tsv rows 35 to 41: +4 video_reencoded, +1 video_offset (a4ba6ede:1), +1 text_conflict (P1 representative).
- Entry changes: 18468a09:6 and e2eae6c8:2 outcome success to carry_over, decision BLUE to NONE (unknown), task_text_ok match to "incomplete (cut before release)".

## 10. Coverage checklist (R11)
- [x] Studio 18468a09 (9 entries) [x] 6d6bdce3 (16) [x] e2eae6c8 (3) [x] a4ba6ede (4) [x] ba9eafba (2) [x] 800d7dc2 (3) [x] d2fd0a80 (2) [x] 51100357 (1) [x] 7e99a9e4 (1)
- [x] All 41 entries in section 4 and labels_episodes.jsonl; 17 representatives viewed by frame; 20 copies pixel-identical; 4 re-encoded copies checked by eye and pixel offset.
- [x] Curated MT-ALL-v2, PP-BLUE, PP-BLUE-v2, PP-BOTH, PP-GREEN, PP-GREEN-v2, S1_v0; 24 of 24 episodes in datasets.json episode_detail.
- [ ] Zip archives inside the curated folders: listed only, not extracted (not needed; the directory form matched all 24 episodes).
- [x] V1 INVENTORY.md, episodes.csv, labels.json (hint only).
- [ ] V1 calib.json, findings.json, gaps.json, inventory.json, sheets/, STATUS.md, verdict.json: not used (hint only; not needed for release plates).
- [x] Protocol and spec read: P3-COMMON.md, RULES.md, LESSONS.md, LAPTOP.md, P3-LABEL-STUDIO.md, P3-LABEL-OURS.md (schema).
- [x] Deliverables: datasets.json, labels_episodes.jsonl, mismatches.tsv, trust_report.md, metrics.json, STATUS.md, laptop-log.md.

## 11. Self-check (R12): facts re-verified in this run
1. info.json total_episodes and total_frames equal meta and data rows for all 9 Studio datasets (verify_facts CHECK 1, re-read from the mirror).
2. Nine copy pairs: raw action rows equal their representative windows, max absolute difference 0.0 (CHECK 2).
3. 20 copies pixel-identical at frames 60 and n/2 (build_checks.json, run 2).
4. a4ba6ede:0 to :3 pixel MAE at offset 0 and offsets -3 to +3 (offset_check.json, script run this session).
5. a4ba6ede:1 best pixel offset -1 (top 3.23 vs 4.01; close 2.24 vs 3.01).
6. Truncated copies: brick between the jaws above the BLUE plate at frame 708 (viewed); representative release at 24.1 s (census gripper event).
7. Census gripper events of the 17 representatives re-read: P16 grasp 10.83 and release 17.8; P13 release 5.0; P3 release 24.1.
8. 7e99a9e4 joint deviations and constant gripper 16.731 (CHECK 3, raw parquet).
9. 7e99a9e4 has no pick at 0, 93, 187 and 267 s (viewed out/p17_montage.jpg).
10. Curated: 24 of 24 exact matches; per-set issue counts (datasets.json, summarize_curated).
11. lag_best 3 frames for all 41 entries; no alignment flags; no action out of bounds (summarize_checks).
12. Trust counts per entry and per physical episode recomputed (check_run2): 23 / 12 / 5 / 1 and 8 / 5 / 3 / 1.
13. mismatches.tsv: 41 rows; kinds counted (check_run2).
14. 7e99a9e4 growth unchanged (build_checks.json, run 2).
Logged in laptop-log.md, entries 2026-11-14T20:05Z to 20:15Z. STATUS.md holds the same list.
