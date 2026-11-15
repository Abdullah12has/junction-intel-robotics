# P3-DATA-CENSUS census (worker E1)

Device: intel-HP-OmniBook-Ultra-Laptop-14t-kd000 (a22743b6-09bb-4707-802d-1b6f3ceca871). Root R = /home/intel/intel-junction-2026. All access read-only.
Catalogue snapshot: 2026-11-14T19:53:16Z (census_build.py run 7, rc 0). File inventory of logs/, data/, run/: 19:29Z (raw/files_all.tsv). Live-tree re-checks at 20:30Z to 20:33Z are in section 9.

## 1. Counts
- 670 rows = 639 live + 31 twin rows (flag deleted_19:48Z).
- 603 unique rows = live rows that are not copies (rule in section 3).
- Inventory scope: 3304 files, 38604389883 bytes, 641 directories. One symlink (logs/ovrun/current) is not counted as a unit.

## 2. Totals per kind and source
| Kind | Source | All rows | Live | Unique |
|---|---|---|---|---|
| episode | friend sim | 150 | 150 | 150 |
| episode | ours | 22 | 22 | 22 |
| episode | ours/twin-chat | 3 | 0 | 0 |
| video | friend sim | 7 | 7 | 7 |
| video | team/friend dataset | 50 | 50 | 30 |
| video | ours | 312 | 312 | 312 |
| video | ours/twin-chat | 12 | 0 | 0 |
| capture_session | ours | 38 | 38 | 38 |
| capture_session | ours/twin-chat | 3 | 0 | 0 |
| dataset | friend sim | 5 | 5 | 3 |
| dataset | ours | 1 | 1 | 1 |
| dataset | team/friend dataset | 24 | 24 | 10 |
| dataset | ours/twin-chat | 7 | 0 | 0 |
| image_set | team/friend dataset | 7 | 7 | 7 |
| image_set | ours | 13 | 13 | 13 |
| image_set | ours/twin-chat | 6 | 0 | 0 |
| report | friend sim | 5 | 5 | 5 |
| report | ours | 4 | 4 | 4 |
| report | team/friend dataset | 1 | 1 | 1 |
| Total | | 670 | 639 | 603 |

Totals by kind (all rows): episode 175, video 381, capture_session 41, dataset 37, image_set 26, report 10.
Row size sums (size_MB as written in the CSV, live rows): video 34351.71, capture_session 29522.78, dataset 9038.67, episode 191.08, image_set 60.28, report 10.16; total 73174.68 MB. These rows overlap (monitor MKV files appear both as capture sessions and as video rows; zip archives and the sim tar.gz repeat folder contents), so the sum does not equal the scope bytes.

## 3. Copies and the unique rule
36 live rows are not unique:
- V001-V020: team copies of the 20 curated videos (flags COPY_OF:18468a09 x18, COPY_OF:a4ba6ede x2).
- D003, D005, D006, D009, D011, D012, D015: curated copies of studio datasets (flag VIDEO_COPY_OF_STUDIO).
- D004, D007, D008, D010, D013, D014, D016: zip archives of curated folders (flag ZIP_MATCH_FOLDER).
- D020: live copy of the sim tar.gz D017 (SHA equal to primary). D021: live mirror of the raw sim attempts D018 (MIRROR_OF_PRIMARY).
The CSV has no unique column. The unique and copy_of fields are in raw/unit_detail.jsonl. The CSV flags identify 35 of the 36 copies; D020 is identified only through unit_detail.jsonl.
Check: 172 episode + 349 video + 38 capture + 14 dataset + 20 image_set + 10 report = 603.

## 4. Video hours and frames
- Unique videos: 349; 11.51 h (11.506 h); 1145965 frames.
- Live videos including copies: 369; 12.08 h; 1208217 frames.
- All video rows including twin: 381; 12.16 h.
- Composition: V021-V027 friend-sim LeRobot camera videos (7); V028-V057 the 30 videos of the nine studio datasets; V058-V363 wrist segments (306); V364-V369 gm and monitor files; V370-V381 twin (deleted). The 300 sim attempt videos (overhead.mp4 and wrist.mp4 for attempts 0000-0149) are inside D018 and have no separate rows.

## 5. UTC timeline
Live rows with a start time: 360 of 639. The other 279 rows (datasets, reports, undated sim and studio items) have none.

| Hour (UTC) | Rows | By kind |
|---|---|---|
| 13 | 60 | video 59, capture 1 |
| 14 | 48 | video 39, capture 9 |
| 15 | 60 | video 60 |
| 16 | 82 | video 63, capture 16, image 3 |
| 17 | 80 | episode 15, video 63, capture 2 |
| 18 | 30 | episode 4, video 26 |

Key times, from names and CSV:
- Wrist session 1: 13:01:30 to 14:12:03 (C001, 71 segments, 70.6 min).
- Wrist session 2: 14:33:11 to 18:26:30 (C002, 235 segments, 233.2 min). All 306 wrist segments total 303.79 min.
- gm session: name stamp 143045 only (no date, no start time in the CSV).
- Ovrun runs: 14:51:53 to 14:55:00 (C027-C034).
- Monitor session A: 16:12:18 to 17:03:48. Monitor session B: 17:03:51 to 18:26:30.
- Depth captures C006-C020: 16:29:04 to 16:52:47.
- E-stop frames: 16:19:05 to 18:13:07 (I017-I019, names are epoch seconds).
- Sort runs: any_1731 17:31:54 to 17:40:40; any_1740 17:40:56 to 18:25:13 (from census_tables.json).
- Twin captures (deleted): C041 17:34:53, C040 17:41:02, C039 17:50:23.
- Telemetry, detection scans, sim and studio datasets carry no time.

## 6. Gaps
- No wrist recording from 14:12:03 to 14:33:11 (1268 s, 21 min 8 s): between V128 and V129.
- Session 2, two joins longer than 1 s: V233 ends 16:17:23, V234 starts 16:17:27 (4 s; V233 is 12.82 s long). V280 ends 17:03:49, V281 starts 17:03:51 (2 s; V280 is 22.61 s long). C002's flag reports 2 gaps, max 3.2 s, which agrees in count.
- 88 of the 305 wrist joins show a 1 s step in whole-second stamps (for example V060 ends 13:04:29, V061 starts 13:04:30). With 60 s segments these are read as rounding, not gaps. 214 joins are 0 s.
- Monitor session A ends 17:03:48 and session B starts 17:03:51 (3 s). V281 starts in the same second as session B.
- Short wrist segments: V128 (33.44 s), V233 (12.82 s), V280 (22.61 s), V363 (39.07 s, last segment).
- Frame-level gaps from the ffprobe flags: V282 GAPS2 (max 0.23 s); V364 GAPS74 (max 2.50 s); V366 GAPS23050 (max 0.33 s); V368 GAPS23179 (max 0.33 s); V367 GAPS58858 (max 0.20 s); V369 GAPS35069 (max 0.47 s). The "miss" totals inside these flags are the builder's values and were not re-derived. V364's miss value (92.0 s) exceeds its clip length (74.5 s), so it is not a usable total.
- Twin data removed at 19:48:00Z (section 8).

## 7. Needs-labelling counts
Values are the task counts in metrics.json. CSV checks are noted.
- T1, sort episodes without a success label: 7. Checked: E153, E154, E158, E161, E165, E168, E172.
- T2, sort episodes without colour or plate: 14 of 22. Checked: 8 carry both colour and plate.
- T3, unique videos to label: 349 videos, 11.51 h, 1145965 frames.
- T4, anomaly rows with blank defect labels: 123. CSV check: D001 reads defect-label(0/123 filled).
- T5, depth sessions: 15 (C006-C020). Checked in the CSV.
- LeRobot episodes without an outcome field: 191 (150 sim in D019, 41 in studio datasets D022-D030). Their labels are task text only (task-text(n)), so colour appears only in the task text for the same 191 episodes. The sim raw dataset D018 carries success labels; its LeRobot export does not.

## 8. Twin scope
- 31 rows, source ours/twin-chat: episodes E173-E175; videos V370-V381; capture sessions C039-C041; datasets D031-D037; image sets I021-I026. All carry deleted_19:48Z, unique False, no links. They are excluded from live and unique counts and appear only in the all-row totals.
- Coordinator instructions: 19:39Z asked for the twin catalogue (source ours/twin-chat). 19:49Z (D-102) reported the deletion.
- R/twin was deleted at 19:48:00Z (manifest header). The directory was gone at 19:50:01Z and still absent at 20:30:00Z.
- Record: R/work/deleted/twin_manifest_20261114T194800Z.txt, 12195 lines, 1139483 bytes, sha256 a5cf581e4c7d3fd9f9dc5b213a66adf614472116a49df0451881cd2893279a57 (checked 20:30:00Z).
- Pre-deletion facts come from build run 3, which read the twin before 19:48:00Z and finished at 19:47:59Z.
- Counting error: E173 (V1) notes "0 image files". The manifest lists 2700 .jpg and 900 .png under twin/work/TW-REC/V1/episodes (plus 1 parquet). The 0 count was wrong.
- E174 (V2) and E175 (V3): the manifest has 3642 entries under each tree but none under their episodes folders (checked 20:32:36Z). The manifest contains no image entries for them there. Their folders were not checked further.
- The twin data itself cannot be re-counted; the manifest is the only record.

## 9. Live-process caveats and changes since the inventory
- Process table read at a time not measured (log L14). Processes running then: sync_datasets.sh (Studio snapshot), sync_sim.sh (mirror of data/sim every 120 s), data.inventory --watch, arm_free_watch.sh, telemetry.py, watch_sim.sh. No wrist or monitor recorder appeared in the table.
- Live flags: C037 telemetry (LIVE_MTIME<15min); R010 studio report live (LIVE_MTIME<15min); I020 run/ (LIVE_SNAPSHOT); R005, R006 and D021 sim live mirror (LIVE_MIRROR).
- At 20:32:07Z versus the 19:29Z inventory: 25 new files (1974964241 bytes), 9 changed files (+1839655 bytes), 0 removed.
- The 25 new files are two Studio dataset folders that are not in the catalogue: data/studio-datasets/8a8bb05e-6f2c-44f5-bf4b-745c2028f7cf (12 files, 987671563 bytes) and data/studio-datasets/e60f4087-d620-4f1d-885f-1620cefd54bd (13 files, 987292678 bytes). They were added after the snapshot and need a catalogue update.
- Changed files: logs/telemetry/20261114.jsonl (7769781 to 9618132 bytes); logs/sync_datasets.log; logs/sync_sim.log; logs/sim_gate.log; logs/sim_gate.out; data/studio-datasets-report/live/watch.log; run/arm_state; run/sync_datasets.pid; run/sync_sim.pid. All nine were last modified after 19:53:16Z.
- Sim mirror at build time: 150 of 150 attempts match; SHA256 manifests 762/762 ok (primary and live).

## 10. Scope bytes discrepancy (open)
- Inventory 19:29Z (raw/files_all.tsv, re-summed 20:30Z): 3304 files, 38604389883 bytes. metrics.json uses this value.
- census_tables.json (build 19:53Z): 3304 files, 38605150795 bytes. Difference: 760912 bytes.
- Not resolved. Build-time per-file sizes were not recorded. The nine changed files all changed after the build, so they cannot account for the difference by their current sizes.

## 11. Video timing findings
- 15 fps finding: the monitor pair from 161218Z (V366 close, V368 top) has r_frame_rate 15/1 (RFR15_1), 46362 frames over 3090.73 s (14.93 fps), and relative timestamps starting at 0. The protocol statement (30 fps, absolute timestamps) does not hold for this pair (log L15).
- Monitor pair 170351Z: close V367 implies 30.30 fps (147602 frames over 4959.13 s); top V369 implies 15.15 fps (147594 frames). Both have absolute timestamps and many non-monotonic timestamps (53554 and 68876). Durations match 17:03:51 to 18:26:30. The two cameras in this session differ by a factor of two in implied rate.
- gm close_green_143045 (V364): 74.50 s, 150 frames, 2.00 fps (RFR2_1), 74 gaps (max 2.50 s), no absolute time.
- gm wrist_green_143045 (V365): 75.00 s, 2252 frames, 1280x720 mjpeg (wrist segments are 640x480). r_frame_rate 359/12 (about 29.92 fps); frame count gives 30.03 fps. Name-versus-view mismatch: the spot check shows a fixed overhead view with no gripper fingers. spot_checks.md flags NAME_ROLE_BY_VIEW; the CSV flags do not carry that flag.
- Wrist fps field: 31.25 (FPS31.2 on 307 rows including V365). Frame counts over the stated durations give about 30.0 fps (V325: 1802 frames over 60.01 s = 30.03 fps).
- Timestamps: relative (PTS_REL) on V001-V280, V364, V365, V366 and V368; absolute (PTS_ABS) from V282 onward on the wrist segments, and on V367 and V369. Absolute timestamps begin at V281 (17:03:51) in the count (85 PTS_ABS rows).

## 12. Spot checks (details in spot_checks.md)
- 649 video files (349 unique plus 300 sim attempt videos): two frames each at 25% and 75%, 1298 PNG frames, 1298 of 1298 extracted, 0 failures.
- 55 contact sheets of 12 items each (sheet 55 holds one item).
- 20 image folders I001-I020, one picked image each (2 image sheets). I021-I026 are twin rows and were not checked.
- Numeric checks: no frame with mean luma below 10; no flat frame (std below 4); no identical 25%/75% pair; no missing frame; mean luma range 15.0 to 151.9.
- Viewing: every sheet was viewed at least once. Sheets 01-07 and 29 were viewed again in the final check. The other sheet descriptions come from the first viewing and were not re-checked.
- Camera roles: serial 243622061421 is overhead; 243722064316 is close/wrist. Sim camera1 is overhead; camera2 is close gripper.
- One name-versus-view mismatch: V365.

## 13. Flags on live rows (token counts)
FPS tokens 311; PTS_REL 284; PTS_ABS 85; NO_TIME 59 (plus NO_TIMESTAMP on I015); MIRROR_OK 150; COPY_OF 20; NO_LABEL 14; NAME_TIME 10; FAILED 9; DERIVED_FROM_FRIEND_DATASETS 8; VIDEO_COPY_OF_STUDIO 7; ZIP_MATCH_FOLDER 7; GAPS on 6 rows; NONMONO on 5 rows; JOINTS_MISSING 5; JOINTS_END_EARLY 4; SHORT_SEGMENT 4; RFR tokens 4; LOG_ONLY 4; NO_T_START 3; NO_EPISODE_JSON 2; LIVE_MTIME 2; LIVE_MIRROR 2; SHA_EQUAL_TO_PRIMARY 2; GAPS_BETWEEN_SEGMENTS 1; EMPTY_FILES 1; LABELS_BLANK 1; MIRROR_OF_PRIMARY 1; LIVE_SNAPSHOT 1.

## 14. Files
- Catalogue: catalogue.csv (670 rows, 16 columns; sha256 8c0e82b43c4c06ef8b3a8fff586fa4441a1e2a36673ba63c854dec1c9b3dab5f), catalogue.json (same rows; sha256 f662654f7ee57161002134b06ffbaacc5b8601ee715d2683e3d764ab5c404cef).
- metrics.json (sha256 3b04c87545f3448273318de515930856877caedeafad14adde94e214214687ea); spot_checks.md; census.md; STATUS.md.
- Raw builder outputs in raw/ (unit_detail.jsonl, census_tables.json, files_all.tsv, frame_stats.tsv, image_pick.tsv, sheet_index.tsv); frames in frames/.
