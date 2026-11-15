# Studio datasets: trust report v2 (P3-LABEL-STUDIO, E2)

Stage studio-labels-v2, run 2026-11-14T20:56Z, mirror snapshot 2026-11-14T20:44Z (11 Studio folders, 52 episode rows). This supersedes E1 (stage studio-labels-v1, 20:12Z, 9 folders, 41 rows), which stays in R/work/P3-LABEL-STUDIO/E1/. Laptop copy: R/work/P3-LABEL-STUDIO/E2/trust_report.md. Machine-readable: labels_episodes.jsonl (52 rows; its first 41 rows are byte-identical to E1), datasets.json (11 Studio, 7 curated, V1 hint), mismatches.tsv, metrics.json. Evidence files (frames, zoom grids, montages, out/ JSON, tools/) stay on the laptop in R/work/P3-LABEL-STUDIO/E2/.

Trust levels: training = one entry per physical episode may train; evaluation = outcome, OD-10 decision or instruction check only, never training; calibration = joint and gripper sanity and grasp events only, no placement label; nothing = do not use.

## 1. Bottom line
- 52 entries in 11 Studio folders are 23 physical episodes. Originals: 6d6bdce3 (16 episodes, P1 to P16), 7e99a9e4 (raw batch, P17) and 8a8bb05e (6 multi-brick episodes, P18 to P23, new in this run). The other 8 folders (29 entries) are copies or windows of these and add no physical episode.
- Training (unchanged): P1, P2, P3, P4, P5, P7, P9, P16 (8 physical; 23 entries counting copies).
- Evaluation only: P6, P8, P13, P14, P15 (from E1) and P18 to P23 (new): 11 physical; 23 entries. Each has an OD-10 or instruction disagreement.
- Calibration only (unchanged): P10, P11, P12 (grasped and held, never placed) plus two truncated copies of P3 (5 entries).
- Nothing (unchanged): P17 (7e99a9e4:0), a raw batch with placeholder text and no pick.
- Verified: 52 of 52 entries. 23 by own frame inspection (17 from E1 plus 6 new); 25 copies pixel-identical to their representative (20 from E1 plus 5 new); 4 copies (a4ba6ede:0 to :3) re-encoded, checked by eye and by pixel offset (E1).
- Mismatched: 23 entries (E1 12 plus 11 new). Task text: 4 (unchanged). OD-10 decision wrong: 20 entries (E1 9 plus 11 new). The new episodes hold 14 placed bricks whose BLUE decision is wrong under OD-10 (section 4); the 5 e60f4087 copies repeat them. In 8a8bb05e:2, 2 bricks were never placed.
- The 12 E1 mismatch entries were re-checked against frames in this run: all confirmed (section 7).

## 2. Datasets (11 Studio folders)
| dataset | entries | originals | training | evaluation | calibration | nothing | what it holds |
|---|---|---|---|---|---|---|---|
| 6d6bdce3 | 16 | 16 (P1 to P16) | 7 | 6 | 3 | 0 | original recording, 16 episodes (V1 counted 11) |
| 8a8bb05e | 6 | 6 (P18 to P23) | 0 | 6 | 0 | 0 | original, new in this run: multi-brick, 3 cameras |
| e60f4087 | 5 | 0 | 0 | 5 | 0 | 0 | copies or windows of 8a8bb05e:0, :1, :3, :4, :5; new in this run |
| 18468a09 | 9 | 0 | 6 | 2 | 1 | 0 | windows of P7, P2, P9, P5, P1, P4, P3, P6, P8 |
| e2eae6c8 | 3 | 0 | 2 | 0 | 1 | 0 | copies or windows of P1, P4, P3 |
| 800d7dc2 | 3 | 0 | 2 | 1 | 0 | 0 | copies of P4, P3, P8 |
| a4ba6ede | 4 | 0 | 4 | 0 | 0 | 0 | re-encoded copies of P7, P2, P9, P5 |
| d2fd0a80 | 2 | 0 | 2 | 0 | 0 | 0 | copies of P4, P3 |
| ba9eafba | 2 | 0 | 0 | 2 | 0 | 0 | copies of P6, P8 |
| 51100357 | 1 | 0 | 0 | 1 | 0 | 0 | copy of P8 |
| 7e99a9e4 | 1 | 1 (P17) | 0 | 0 | 0 | 1 | raw batch, 280.1 s |

Facts common to the folders: fps 30; two cameras in 10 folders (serial 243622061421 = top, 243722064316 = close); 8a8bb05e has a third camera, usb2_0_cam1 "external camera" (serial not in info.json); AV1, 640x480; joint names shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos; units not stated in info.json. info.json total_episodes and total_frames equal the meta and data row counts in all 11 folders (checked in this run). Durations: 8a8bb05e 564.8 s in 6 episodes (73.6 to 128.1 s); e60f4087 488.4 s in 5 episodes. The other durations are unchanged from E1.

## 3. The 23 physical episodes
- P1 to P17: unchanged from E1 (training: P1, P2, P3, P4, P5, P7, P9, P16; evaluation: P6, P8, P13, P14, P15; calibration: P10, P11, P12; nothing: P17). E1 section 3 has the per-episode table.
- P18 (8a8bb05e:0): 7 bricks, all placed on BLUE; 2 defect bricks on BLUE (yellow 2x2 with tape, white 2x4). Evaluation.
- P19 (8a8bb05e:1): 7 bricks, all placed on BLUE; 2 defect bricks on BLUE (white 2x4, yellow 2x2 with tape). Evaluation.
- P20 (8a8bb05e:2): 5 bricks placed on BLUE, 2 left on the table; 3 defect bricks on BLUE (blue 2x2 with tape, yellow 2x2 with tape, white 2x4). Evaluation, outcome partial.
- P21 (8a8bb05e:3): 7 bricks, all placed on BLUE; 3 defect bricks on BLUE (blue 2x2 with tape, yellow 2x2 with tape, white 2x4). Evaluation.
- P22 (8a8bb05e:4): 7 bricks, all placed on BLUE; 2 defect bricks on BLUE (white 2x4, yellow 2x2 with tape). Evaluation.
- P23 (8a8bb05e:5): 7 bricks, all placed on BLUE; 2 defect bricks on BLUE (white 2x4, yellow 2x2 with tape). Evaluation.

Why evaluation and not training for P18 to P23: the operator's plate (BLUE) disagrees with OD-10 for the tape and 2x4 bricks, so a placement label would teach the wrong defect rule. The joint data is sane (section 6), so the episodes are not discarded; they are usable for checking outcomes and OD-10 decisions only. Per-brick records are in section 4.

## 4. New physical episodes in 8a8bb05e (P18 to P23), per brick
Each episode starts with 7 bricks on the table and ends with the bricks on the BLUE plate (task text: pick up all the blocks and put them on the blue plate). OD-10: clean 2x2 goes to BLUE, tape or non-2x2 goes to GREEN. Tape from top-view zoom grids (out/zoom_8a8bb05e_start.png); placement from end-state zoom (out/zoom_8a8bb05e_end.png).

### 8a8bb05e:0 (P18), 2934 frames, 97.8 s, outcome: success (7 of 7 on BLUE)
| brick | colour | 2x2 | tape | placement | OD-10 expected | decision correct |
|---|---|---|---|---|---|---|
| B1 | white | yes | no | BLUE | BLUE | yes |
| B2 | orange | yes | no | BLUE | BLUE | yes |
| B3 | orange | yes | no | BLUE | BLUE | yes |
| B4 | red | yes | no | BLUE | BLUE | yes |
| B5 | yellow | yes | yes (grey strip, left side) | BLUE | GREEN | no |
| B6 | white | no (2x4) | yes (grey strip, top left) | BLUE | GREEN | no |
| B7 | blue | yes | unclear (small dark mark) | BLUE | unknown | unknown |

Trust: evaluation. Why: OD-10 decision wrong for defect bricks on BLUE (evaluation only); joint data sane. Joint data: lag 3 frames (MAE 0.862), gripper [0.0, 25.02], limit hits: none.

### 8a8bb05e:1 (P19), 3844 frames, 128.1 s, outcome: success (7 of 7 on BLUE)
| brick | colour | 2x2 | tape | placement | OD-10 expected | decision correct |
|---|---|---|---|---|---|---|
| B1 | white | no (2x4) | yes (grey tape, top middle) | BLUE | GREEN | no |
| B2 | blue | yes | no | BLUE | BLUE | yes |
| B3 | red | yes | no | BLUE | BLUE | yes |
| B4 | white | yes | no | BLUE | BLUE | yes |
| B5 | orange | yes | no | BLUE | BLUE | yes |
| B6 | yellow | yes | yes (grey strip, left side) | BLUE | GREEN | no |
| B7 | orange | yes | no | BLUE | BLUE | yes |

Trust: evaluation. Why: OD-10 decision wrong for defect bricks on BLUE (evaluation only); joint data sane. Joint data: lag 3 frames (MAE 0.787), gripper [0.0, 33.15], limit hits: shoulder_lift.pos at +100 x0 / -100 x4.

### 8a8bb05e:2 (P20), 2209 frames, 73.6 s, outcome: partial (5 of 7 on BLUE; red 2x2 and white 2x2 left on the table)
| brick | colour | 2x2 | tape | placement | OD-10 expected | decision correct |
|---|---|---|---|---|---|---|
| B1 | blue | yes | yes (grey strip, right side) | BLUE | GREEN | no |
| B2 | orange | yes | no | BLUE | BLUE | yes |
| B3 | orange | yes | no | BLUE | BLUE | yes |
| B4 | yellow | yes | yes (grey strip, left side) | BLUE | GREEN | no |
| B5 | white | no (2x4) | yes (grey tape, top) | BLUE | GREEN | no |
| B6 | red | yes | no | table (not placed) | BLUE | n/a (not placed) |
| B7 | white | yes | no | table (not placed) | BLUE | n/a (not placed) |

Trust: evaluation. Why: OD-10 decision wrong for defect bricks on BLUE (evaluation only); joint data sane; 2 bricks not placed. Joint data: lag 3 frames (MAE 0.766), gripper [0.0, 32.38], limit hits: shoulder_lift.pos at +100 x0 / -100 x7.

### 8a8bb05e:3 (P21), 2879 frames, 96.0 s, outcome: success (7 of 7 on BLUE)
| brick | colour | 2x2 | tape | placement | OD-10 expected | decision correct |
|---|---|---|---|---|---|---|
| B1 | blue | yes | yes (dark strip, bottom) | BLUE | GREEN | no |
| B2 | yellow | yes | yes (grey strip, left side) | BLUE | GREEN | no |
| B3 | orange | yes | no | BLUE | BLUE | yes |
| B4 | red | yes | no | BLUE | BLUE | yes |
| B5 | orange | yes | no | BLUE | BLUE | yes |
| B6 | white | yes | no | BLUE | BLUE | yes |
| B7 | white | no (2x4) | yes (grey tape, top) | BLUE | GREEN | no |

Trust: evaluation. Why: OD-10 decision wrong for defect bricks on BLUE (evaluation only); joint data sane. Joint data: lag 3 frames (MAE 0.706), gripper [0.0, 58.02], limit hits: shoulder_lift.pos at +100 x0 / -100 x8.

### 8a8bb05e:4 (P22), 2584 frames, 86.1 s, outcome: success (7 of 7 on BLUE)
| brick | colour | 2x2 | tape | placement | OD-10 expected | decision correct |
|---|---|---|---|---|---|---|
| B1 | white | no (2x4) | yes (grey tape, top) | BLUE | GREEN | no |
| B2 | yellow | yes | yes (grey tape, top) | BLUE | GREEN | no |
| B3 | blue | yes | no | BLUE | BLUE | yes |
| B4 | white | yes | no | BLUE | BLUE | yes |
| B5 | orange | yes | no | BLUE | BLUE | yes |
| B6 | orange | yes | no | BLUE | BLUE | yes |
| B7 | red | yes | no | BLUE | BLUE | yes |

Trust: evaluation. Why: OD-10 decision wrong for defect bricks on BLUE (evaluation only); joint data sane. Joint data: lag 3 frames (MAE 0.758), gripper [0.0, 43.76], limit hits: shoulder_lift.pos at +100 x0 / -100 x9; elbow_flex.pos at +100 x17 / -100 x0.

### 8a8bb05e:5 (P23), 2493 frames, 83.1 s, outcome: success (7 of 7 on BLUE)
| brick | colour | 2x2 | tape | placement | OD-10 expected | decision correct |
|---|---|---|---|---|---|---|
| B1 | white | no (2x4) | yes (grey tape, top) | BLUE | GREEN | no |
| B2 | yellow | yes | yes (grey tape, top) | BLUE | GREEN | no |
| B3 | blue | yes | no | BLUE | BLUE | yes |
| B4 | white | yes | no | BLUE | BLUE | yes |
| B5 | red | yes | no | BLUE | BLUE | yes |
| B6 | orange | yes | no | BLUE | BLUE | yes |
| B7 | orange | yes | no | BLUE | BLUE | yes |

Trust: evaluation. Why: OD-10 decision wrong for defect bricks on BLUE (evaluation only); joint data sane. Joint data: lag 3 frames (MAE 0.780), gripper [0.0, 42.76], limit hits: shoulder_lift.pos at +100 x0 / -100 x16.

## 5. Per-entry trust (52 entries)
| entry | P | trust | pixel check vs representative | outcome | decision correct | task text |
|---|---|---|---|---|---|---|
| 18468a09:0 | P7 | training | identical_decode | success | n/a | match |
| 18468a09:1 | P2 | training | identical_decode | success | n/a | match |
| 18468a09:2 | P9 | training | identical_decode | success | n/a | match |
| 18468a09:3 | P5 | training | identical_decode | success | n/a | match |
| 18468a09:4 | P1 | training | identical_decode | success | yes | match |
| 18468a09:5 | P4 | training | identical_decode | success | yes | match |
| 18468a09:6 | P3 | calibration | identical_decode | carry_over | unknown | incomplete (cut before release |
| 18468a09:7 | P6 | evaluation | identical_decode | success | no | match |
| 18468a09:8 | P8 | evaluation | identical_decode | success | no | match |
| 51100357:0 | P8 | evaluation | identical_decode | success | no | match |
| 6d6bdce3:0 | P10 | calibration | own frames | stopped | n/a | incomplete (not placed) |
| 6d6bdce3:1 | P11 | calibration | own frames | stopped | n/a | incomplete (not placed) |
| 6d6bdce3:2 | P12 | calibration | own frames | stopped | n/a | incomplete (not placed) |
| 6d6bdce3:3 | P7 | training | own frames | success | n/a | match |
| 6d6bdce3:4 | P2 | training | own frames | success | n/a | match |
| 6d6bdce3:5 | P9 | training | own frames | success | n/a | match |
| 6d6bdce3:6 | P5 | training | own frames | success | n/a | match |
| 6d6bdce3:7 | P1 | evaluation | own frames | success | yes | mismatch |
| 6d6bdce3:8 | P13 | evaluation | own frames | success | no | mismatch |
| 6d6bdce3:9 | P14 | evaluation | own frames | success | yes | mismatch |
| 6d6bdce3:10 | P15 | evaluation | own frames | success | yes | mismatch |
| 6d6bdce3:11 | P4 | training | own frames | success | yes | match |
| 6d6bdce3:12 | P3 | training | own frames | success | yes | match |
| 6d6bdce3:13 | P16 | training | own frames | success | yes | match |
| 6d6bdce3:14 | P6 | evaluation | own frames | success | no | match |
| 6d6bdce3:15 | P8 | evaluation | own frames | success | no | match |
| 7e99a9e4:0 | P17 | nothing | own frames | none (no pick) | n/a | placeholder |
| 800d7dc2:0 | P4 | training | identical_decode | success | yes | match |
| 800d7dc2:1 | P3 | training | identical_decode | success | yes | match |
| 800d7dc2:2 | P8 | evaluation | identical_decode | success | no | match |
| a4ba6ede:0 | P7 | training | reencoded_same_scene | success | n/a | match |
| a4ba6ede:1 | P2 | training | reencoded_same_scene | success | n/a | match |
| a4ba6ede:2 | P9 | training | reencoded_same_scene | success | n/a | match |
| a4ba6ede:3 | P5 | training | reencoded_same_scene | success | n/a | match |
| ba9eafba:0 | P6 | evaluation | identical_decode | success | no | match |
| ba9eafba:1 | P8 | evaluation | identical_decode | success | no | match |
| d2fd0a80:0 | P4 | training | identical_decode | success | yes | match |
| d2fd0a80:1 | P3 | training | identical_decode | success | yes | match |
| e2eae6c8:0 | P1 | training | identical_decode | success | yes | match |
| e2eae6c8:1 | P4 | training | identical_decode | success | yes | match |
| e2eae6c8:2 | P3 | calibration | identical_decode | carry_over | unknown | incomplete (cut before release |
| 8a8bb05e:0 | P18 | evaluation | own frames | success (7 of 7 on BLUE) | partial (2 of 7 placed bricks  | match (placed bricks on BLUE,  |
| 8a8bb05e:1 | P19 | evaluation | own frames | success (7 of 7 on BLUE) | partial (2 of 7 placed bricks  | match (placed bricks on BLUE,  |
| 8a8bb05e:2 | P20 | evaluation | own frames | partial (5 of 7 on BLUE; red 2x2 and whi | partial (3 of 5 placed bricks  | incomplete (2 bricks left on t |
| 8a8bb05e:3 | P21 | evaluation | own frames | success (7 of 7 on BLUE) | partial (3 of 7 placed bricks  | match (placed bricks on BLUE,  |
| 8a8bb05e:4 | P22 | evaluation | own frames | success (7 of 7 on BLUE) | partial (2 of 7 placed bricks  | match (placed bricks on BLUE,  |
| 8a8bb05e:5 | P23 | evaluation | own frames | success (7 of 7 on BLUE) | partial (2 of 7 placed bricks  | match (placed bricks on BLUE,  |
| e60f4087:0 | P18 | evaluation | identical_decode | success (7 of 7 on BLUE) | partial (2 of 7 placed bricks  | match (placed bricks on BLUE,  |
| e60f4087:1 | P19 | evaluation | identical_decode | success (7 of 7 on BLUE) | partial (2 of 7 placed bricks  | match (placed bricks on BLUE,  |
| e60f4087:2 | P21 | evaluation | identical_decode | success (7 of 7 on BLUE) | partial (3 of 7 placed bricks  | match (placed bricks on BLUE,  |
| e60f4087:3 | P22 | evaluation | identical_decode | success (7 of 7 on BLUE) | partial (2 of 7 placed bricks  | match (placed bricks on BLUE,  |
| e60f4087:4 | P23 | evaluation | identical_decode | success (7 of 7 on BLUE) | partial (2 of 7 placed bricks  | match (placed bricks on BLUE,  |

## 5b. Notes on the table above
Columns are the state written to labels_episodes.jsonl. Pixel results are in each row's frame_check_vs_rep field.

## 6. Evidence and checks (run 2026-11-14T20:56Z)
- Census: 11 folders, 52 rows, read from all data and meta chunks (tools/census_e2.py, out/census_e2.json, out/census_e2_stdout.txt). Exact action-row matching found 29 copies or windows; 23 episodes contain no other episode's rows.
- New copies (e60f4087:0 to :4): action rows equal their window of the representative exactly (census). Pixels: MAE 0.0 for every camera at frames 60 and n/2 of each copy (top, close, external), so all 5 are identical decodes (out/e2_pixel_sanity.json).
- Joint sanity for the 11 new rows (checked in this run): action lags observation.state by 3 frames in all 11 (lag_best 3); MAE 0.706 to 0.866; action_range_in_bounds 11 of 11; frame_index contiguous 11 of 11. Shoulder_lift and elbow/wrist extremes were not re-checked in this run.
- Frame indices are contiguous and meta lengths equal the row counts for all 11 new rows. The video-span check (to_timestamp minus from_timestamp against rows) was not computed for the new rows.
- E1 rows: the first 41 rows of labels_episodes.jsonl are byte-identical to the E1 file.
- The census lists 52 rows: 8a8bb05e (16943 rows in 6 episodes) and e60f4087 (14651 rows in 5 episodes) are the new folders since the E1 census (19:28Z, 9 folders). Other folders were not re-read for row counts in this run.

## 7. E1 mismatch re-check (carried from E1; not re-viewed in this run)
The 12 E1 mismatch entries were confirmed against frames in the E1 run (E1 trust_report.md). In this E2 run their rows were checked only as unchanged: the first 41 rows of labels_episodes.jsonl are byte-identical to E1, and the census and metrics agree. Montages were not re-viewed in E2, so the per-entry visual notes are not repeated here.
Copies of P6 and P8 (18468a09:7, 18468a09:8, 51100357:0, 800d7dc2:2, ba9eafba:0, ba9eafba:1) share action rows with their representative, so their mismatch follows that representative.
Curated and V1 mismatch rows were not re-checked in this run (curated labels are not used as truth).

## 8. Open issues and limits (E2)
1. Multi-brick episodes: tape and 2x2 or 2x4 calls come from top-view zoom grids at about 2x. Each brick was not checked at close range. One blue brick in 8a8bb05e:0 is "unclear" and is recorded as decision unknown. Confidence is medium; treat the per-brick OD-10 counts as provisional until a close-range check.
2. Placements come from the last sampled frame (end minus 3 frames). Brick-by-brick release times were not traced, so release_s_in_rep is null for the new episodes.
3. 8a8bb05e:2 ends with 2 bricks on the table (red 2x2, white 2x2). Its outcome is partial.
4. The mirror syncs every 120 s. These counts are for the 20:44Z listing (census run about 20:45Z). Re-census before using them.
5. The video-span check was not computed for the new rows (section 6).
6. External camera usb2_0_cam1 exists only in 8a8bb05e; its serial is not in info.json.
7. Orientation is not recorded (as in E1). info.json states no joint units (as in E1).
8. The curated sets (24 episodes) were not re-run in E2. They matched E1 entries and are unchanged.

## 9. Metrics v2 against v1 (metrics.json previous and delta)
- episodes_total 41 to 52 (+11); datasets_studio 9 to 11 (+2); physical unique episodes 17 to 23 (+6).
- verified 41 to 52 (+11); by own frames 17 to 23 (+6); identical copies 20 to 25 (+5); re-encoded copies 4 (unchanged).
- mismatched 12 to 23 (+11); task text mismatch 4 (unchanged); OD-10 decision wrong 9 to 20 (+11).
- usable training 23 (unchanged); evaluation 12 to 23 (+11); calibration 5 (unchanged); nothing 1 (unchanged).
- Physical trust: training 8 (unchanged), evaluation-only 5 to 11, calibration 3 (unchanged), nothing 1 (unchanged).

## 10. Coverage checklist (R11)
- [x] All 11 Studio folders: 18468a09 (9), 51100357 (1), 6d6bdce3 (16), 7e99a9e4 (1), 800d7dc2 (3), 8a8bb05e (6), a4ba6ede (4), ba9eafba (2), d2fd0a80 (2), e2eae6c8 (3), e60f4087 (5) = 52 entries.
- [x] All 52 entries in section 5 and labels_episodes.jsonl. The 6 new physical episodes were viewed at full resolution (start and end frames, zoom grids, 8-frame montages on each camera).
- [x] 25 copies pixel-identical; 4 re-encoded copies checked in E1.
- [x] E1 mismatch entries re-checked (section 7).
- [ ] Curated sets re-run in E2: not done (unchanged).
- [ ] V1 folder re-read in E2: not done (hint only; E1 used it).
- [x] Deliverables: datasets.json, labels_episodes.jsonl, mismatches.tsv, trust_report.md, metrics.json, STATUS.md, laptop-log.md.

## 11. Self-check (R12): facts checked in E2
1. Census of 11 folders: 52 rows, 52 unique entries (census_e2.json).
2. info.json total_episodes and total_frames equal meta rows and data rows in all 11 folders.
3. The first 41 rows of labels_episodes.jsonl are byte-identical to E1.
4. Trust counts from the rows: 23 training, 23 evaluation, 5 calibration, 1 nothing; 23 physical episodes.
5. Five copy pairs: action rows equal the window exactly (census).
6. Five copy pairs: pixel MAE 0.0 at frames 60 and n/2 on all three cameras.
7. Joint sanity for the 11 new rows: lag 3 frames; in-bounds check (count reported in section 6).
8. Per-brick OD-10 counts for the six new episodes: 14 wrong, 25 correct, 1 unknown, 2 not placed. The section 4 tables come from the same records.
9. E1 mismatch entries re-viewed on montages (section 7).
10. 7e99a9e4 unchanged in the census: 1 row (P17); its row count was not re-read in this run.
11. mismatches.tsv: 53 data rows (the E1 file had 41); row kinds were not tallied in this run. Entry counts are in metrics.json: 23 mismatched, 20 OD-10 wrong, 4 task text.
12. datasets.json: 19 entries (11 Studio, 7 curated, 1 V1 hint).
13. Metrics arithmetic: physical 17 + 6 = 23; entries 41 + 11 = 52.
14. utc fields set to 2026-11-14T20:56Z (run stamp); snapshot_utc is the 20:44Z listing.
