# P2-B-ANALYZE report (analyst E1): pick-anywhere runs any_1731 and any_1740

Times are UTC (epoch base 2026-11-14T00:00:00Z). Row ids are `run aN (ep K)`: aN is the attempt index, unique within the run; K is the episode number, which repeats (section 7, rec 3). A row is one attempt: a loop-log line and/or a save record in logs/sort/<run>/cycles.jsonl.

## 1. Outcome counts (one row per attempt)
| run | rows | success | approach_fail | grasp_fail | estop | stopped | carry_over | unknown |
|---|---|---|---|---|---|---|---|---|
| any_1731 | 5 | 0 | 0 | 0 | 2 | 1 | 2 | 0 |
| any_1740 | 24 | 4 | 6 | 0 | 5 | 6 | 3 | 0 |
| all | 29 | 4 | 6 | 0 | 7 | 7 | 5 | 0 |

Pick attempts (approach started): 17, of which successes: 4. Carry-over rows: 5. Wait-stop rows: 7.

## 2. Failure taxonomy (rows that are not success)
| category | n | example rows |
|---|---|---|
| stopped: while waiting | 7 | 1731 a5 (ep 4), 1740 a2 (ep 2), 1740 a3 (ep 2), 1740 a13 (ep 11) |
| carry-over: placed | 4 | 1731 a4 (ep 3), 1740 a7 (ep 5), 1740 a10 (ep 8), 1740 a18 (ep 14) |
| approach: IK fail | 3 | 1740 a4 (ep 2), 1740 a5 (ep 3), 1740 a11 (ep 9) |
| approach: brick not visible at hover | 2 | 1740 a12 (ep 10), 1740 a15 (ep 12) |
| estop: ascent or lift (dz>0) | 2 | 1731 a1 (ep 1), 1740 a6 (ep 4) |
| estop: transit or plate move (move) | 2 | 1731 a2 (ep 2), 1740 a9 (ep 7) |
| approach: could not centre at hover | 1 | 1740 a20 (ep 16) |
| carry-over: crash | 1 | 1731 a3 |
| estop: descent (dz<0) | 1 | 1740 a17 (ep 14) |
| estop: transit move (tmove) | 1 | 1740 a16 (ep 13) |
| estop: xy move (approach or servo step) | 1 | 1740 a14 (ep 11) |

## 3. Timing (seconds; median and max)
| measure | n | median | max |
|---|---|---|---|
| wait, successful cycles | 4 | 1.05 | 1.3 |
| pick, successful cycles | 4 | 46.05 | 53.5 |
| inspect, successful cycles | 4 | 0.0 | 0.0 |
| place, successful cycles | 4 | 20.5 | 51.9 |
| return, successful cycles | 4 | 2.5 | 15.0 |
| cycle, successful cycles | 4 | 69.0 | 120.4 |
| inside pick: approach moves before close (sum of bursts) | 10 | 15.36 | 20.13 |
| inside pick: close (gripper from open to closed) | 10 | 0.9 | 0.92 |
| inside pick: lift start after close | 10 | 2.27 | 2.44 |
| inside pick: attempt line to first servo line (log) | 14 | 14.0 | 211.0 |
| inside pick: servo iterations per attempt (before + after yaw) | 22 | 5.0 | 10.0 |

inspect_s measures only the station re-look (sort_any.py cycle()); 0 of 4 successful cycles had one. place_s covers place(plate).

## 4. Grasp readings (grip_closed after the close command; held = 1.1 or more)
| class | n | min | median | max | grip_lifted median (n) |
|---|---|---|---|---|---|
| empty | 0 | None | None | None | None (0) |
| held | 10 | 1.5 | 1.9 | 2.29 | 1.57 (8) |

## 5. Estimate-to-hover error vs distance from the scan centre
| scan distance (mm) | n | median error (mm) | max error (mm) |
|---|---|---|---|
| 0 to 20 | 1 | 4.3 | 4.3 |
| 20 to 40 | 1 | 15.9 | 15.9 |
| 40 to 60 | 0 | None | None |
| 60 and more | 5 | 42.2 | 49.3 |

Pearson r of error against distance: 0.798 (n=7). Hover tip size minus FK tip height (mm): n=7, median -1.6, min -9.9, max 0.1.

## 6. Hand watchdog: the 41 estop images, reviewed by eye, per view
| view | images | REAL_HAND | FALSE_POSITIVE | UNKNOWN | false-positive rate (FP / (FP+REAL)) |
|---|---|---|---|---|---|
| top | 17 | 16 | 1 | 0 | 5.9% |
| close | 9 | 2 | 7 | 0 | 77.8% |
| wrist | 15 | 0 | 13 | 2 | 100.0% |
| all | 41 | 18 | 21 | 2 | 53.8% |

hand_watch.log: 6617 HAND? lines from 16:12:22 to 18:13:07 (lines per UTC hour: 16h 4695, 17h 355, 18h 1567), 0 unparsed, 54 trigger rows. Last estop image 18:13:07.
Loop estop lines: {'any_1731': 10, 'any_1740': 49}. Estop events (lines within 20 s grouped): {'any_1731': 6, 'any_1740': 9}. Image classes on loop lines: {'REAL_HAND': 13, 'UNKNOWN': 6, 'FALSE_POSITIVE': 40}.

## 7. Recommendations
1. Stop retrying sooner. 1740 a17 (ep 14) logged 40 retry lines in 29 s (18:02:53 to 18:03:22, last retry 40); recover() loops range(40) (sort_any.py line 570). Hand over to a person after a few retries.
2. Save every attempt. 4 of 22 pick and carry-over attempts have no save record (rows: 1731 a3, 1740 a6 (ep 4), 1740 a9 (ep 7), 1740 a14 (ep 11)); their loop-log lines are the only record. Call save_ep() on every exit path.
3. Key saves by attempt, not episode number. 1731: 0 of 4 rows share an episode number with another row; 1740: 11 of 24 rows share an episode number with another row; most shared: ep 2 (3 rows). 5 rows (1740 a2 (ep 2), 1740 a3 (ep 2), 1740 a17 (ep 14), 1740 a21 (ep 17), 1740 a23 (ep 18)) read another save's episode.json, which save_ep() overwrites on every call.
4. Handle Stop on the carry-over start path. 1731 a3 raised an uncaught grasp_auto.Stop from inspect_pose() at 17:36:15; the process exited and the next session began 72 s later at 17:37:27.
5. One label source. 1731 a2 (ep 2): episode.json says GOOD, cycles.jsonl says DEFECT; write one label field.
6. Gate the wrist and close views. False positives: wrist 13 of 13 (100.0%), close 7 of 9 (77.8%), top 1 of 17 (5.9%). Require a top-view trigger or a second frame before a wrist or close trigger stops the loop.
7. Add a watchdog heartbeat. hand_watch.py writes a HAND? line only when is_hand() is true (line 84), so silence after 18:13:07 cannot show the watchdog was running; the loop logged until 18:25:13 (12 min later, 0 loop estop lines after it).
8. Limit picks to the near scan zone or recalibrate the scan map. Estimate-to-hover error: median 10.1 mm within 40 mm of the scan centre (n=2), 42.2 mm at 60 mm or more (n=5); Pearson r 0.798 (n=7).
9. Reject unreachable bricks before the approach. 3 of 6 approach failures are IK failures (1740 a4 (ep 2), 1740 a5 (ep 3), 1740 a11 (ep 9)); one logged error reads 'ik fail moving [-7.2, 79.7] mm at scan height'.
10. Shorten the approach. Pick attempts spent a median 15.36 s moving before close (n=10) and a median 14.0 s from the attempt line to the first servo line (max 211.0 s, n=14); successful picks took a median 46.05 s.

## 8. Data integrity
- any_1731: 4 loop attempt lines; 4 save records (1 without a loop line); 1 attempts with no save record; 0 rows with another save's episode.json; 1 label conflicts; 1 traceback lines; 0 unparsed lines; last stamped loop line 17:40:40.
- any_1740: 18 loop attempt lines; 21 save records (6 without a loop line); 3 attempts with no save record; 5 rows with another save's episode.json; 0 label conflicts; 0 traceback lines; 0 unparsed lines; last stamped loop line 18:25:13.

## 9. Coverage checklist
- [x] logs/sort_any_any_1731.out: 59 lines; 0 unparsed (section 8)
- [x] logs/sort/any_1731/cycles.jsonl: 4 records; each matched to a loop line or listed as its own row
- [x] logs/sort/any_1731/ep_*: 4 folders; episode.json and joints.csv read where present (json_state column)
- [x] logs/sort_any_any_1740.out: 341 lines; 0 unparsed (section 8)
- [x] logs/sort/any_1740/cycles.jsonl: 21 records; each matched to a loop line or listed as its own row
- [x] logs/sort/any_1740/ep_*: 18 folders; episode.json and joints.csv read where present (json_state column)
- [x] logs/monitor/hand_watch.log: 6617 lines; 0 unparsed (section 6)
- [x] logs/monitor/estop_*.jpg: 41 files, each viewed and classified; all listed in estops.csv
- [x] app/tools/sort_any.py: cycle(), save_ep() and main() read for the fields used; recover() located by grep (file last modified 2026-11-14T18:53:02Z, 674 lines)
- [x] app/tools/kin.py: lines 8-71 read; fk() and MAT_Z used here; the import reads files only (URDF, project_robots.json)
- [x] app/pick_v2/kinematics.py: import chain checked; hardware-term grep run (see laptop-log.md)
- [x] app/tools/scan_calib.json: q_scan through kin.fk() gives SCAN_XY; target_scan gives TS (main())
- [x] docs/PICK-ANYWHERE.md line 61 states the held threshold (gripper reading >= 1.1); HELD_MIN = 1.1 here
- [ ] app/tools/grasp_auto.py, app/tools/pick_any.py, app/tools/hand_watch.py: not re-read in full in this pass
