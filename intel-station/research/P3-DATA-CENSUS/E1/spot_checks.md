# Spot checks (E1, P3-DATA-CENSUS)

## What was checked
- 649 video files (349 unique catalogue video units plus 300 friend-sim attempt videos, overhead.mp4 and wrist.mp4 of attempts 0000-0149). Two frames per file, at 25% and 75% of the file duration, extracted by seek only and scaled to 192x144. 1298 PNG frames in R/work/P3-DATA-CENSUS/E1/frames/ (1298 of 1298 extracted, 0 failures; raw/extract_log.tsv).
- Contact sheets of 12 items each: frames/sheets/sheet_01.png to sheet_55.png (sheet 55 holds one item). Caption on each frame gives the catalogue ID or sim key and the file name; index in raw/sheet_index.tsv.
- One picked image per image folder (20 folders): frames/images/image_sheet_01.png and image_sheet_02.png; pick list in raw/image_pick.tsv.
- Numeric checks on all 1298 frames (raw/frame_stats.tsv): no frame with mean luma below 10, no flat frame (std below 4), no identical 25%/75% pair, no missing frame. Mean luma range 15.0 to 151.9.

## Viewing record
- Every sheet (01-55) and both image sheets were returned as images and were looked at. Some sheets were later removed from the working context by the harness to free space ("media removed ... already seen"); their observations below were recorded when they were viewed. Sheets 01-07 and 29 were re-viewed during the final check; the image sheets were viewed once in the first pass.
- Observations below cover what was seen. Where a sheet was viewed only once, the note is from that viewing and was not re-checked afterwards.

## Camera roles (name versus view)
- RealSense 243622061421 = overhead (top) view. Confirmed on sheets 01-04 for every studio-dataset file in the range V028-V057: V028-V030, V034, V036-V039, V044, V046, V048, V050, V052, V054, V055. Consistent with an overhead name.
- RealSense 243722064316 = close view with the gripper fingers at the bottom edge. Confirmed on sheets 01-03: V031-V033, V035, V040-V043, V045, V047, V049, V051, V053, V056, V057. Consistent with a wrist name.
- Friend-sim LeRobot camera1 (V021-V024) = overhead view of the workspace. Friend-sim LeRobot camera2 (V025-V027) = close gripper view. The camera1 and camera2 names carry no role; the views match overhead and wrist respectively. Recorded as NAME_ROLE_BY_VIEW (names do not state the role; the role is read from the view).
- Friend-sim attempt files (sheets 30-55): overhead.mp4 = workspace from above in every attempt viewed; wrist.mp4 = orange gripper close up in every attempt viewed. Names match the views.
- gm wrist_green_143045.mkv (V365, sheet 29, re-viewed): fixed overhead view of the bowls and bricks with no gripper fingers in either frame. The name says wrist; the view does not. Flag NAME_ROLE_BY_VIEW for this file. gm close_green_143045.mkv (V364) shows a close view of bricks (name matches).
- Monitor files (sheet 29-30): close_20261114T161218Z.mkv (V366) and close_20261114T170351Z.mkv (V367) show a close view with the red gripper finger at the top (name matches). top_20261114T161218Z.mkv (V368) and top_20261114T170351Z.mkv (V369) show the workspace from above (name matches).

## Wrist segments (data/wrist/raw; names give UTC start, one segment per minute at this stage)
- V058-V068 (13:01:30-13:11:30; sheets 04): bright scenes with red gripper fingers at the bottom (V058-V062), then dim (V063-V068, mean luma 17-29, not black). No mismatch.
- V069-V080 (13:12:30-13:23:30; sheet 05, re-viewed): V069 dark frame with the gripper; V070 blurred bluish frame with a white object; V071-V079 a grey equipment structure with cables, seen from the wrist camera, nearly static; V080 dark blue at the right.
- V081-V092 (13:24:30-13:35:30; sheet 06): V081 blue cast with the gripper; V082-V088 near-identical dark frames with the bowl edge and red fingers; V089-V090 grey-violet haze; V091-V092 bricks held in the gripper over the bowl.
- V093-V104 (13:36:30-13:47:30; sheet 07, re-viewed): V093 red brick on a green bowl; V094 bricks with red fingers and a blue bowl edge; V095-V104 near-identical pairs of an orange brick on a dark purple-blue table with the cyan bowl edge and red fingers.
- V105-V152 (13:48-14:56; sheets 08-11; first viewing): static bricks and bowl view (V105-V114); equipment room (V115); bricks held over the bowl (V116); dark red-brown scene (V117-V119); a bench with electronics, cables and a white box, a hand visible at V120 (V120-V126); bricks and bowl (V127-V128); dark brown bowl views (V129-V136); blue-brown scene with bricks held over the bowl from V139 (V137-V140); bowl, one orange brick and a keyboard edge (V141-V152).
- V153-V236 (14:57-16:19; sheets 12-18; first viewing): bowl with orange and blue bricks, mostly near-identical runs; a dark table with a white object (V207-V216); a person seated at a desk in a room view (V217; a face is visible; not identified and not recorded further); bowl and bricks in dark views (V218-V224); a hand holding a blue brick (V226).
- V237-V284 (16:20-17:07; sheets 19-22; first viewing): blue, white and yellow bricks on a dark table; a brick in the gripper; bench with equipment (V271); a paper-like flat card held over the bowl from V272 (near identical pairs).
- V285-V332 (17:07-17:54; sheets 23-26; first viewing): card-like item over the bowl (V285-V287); bricks on the dark table with the gripper (V288-V308); large orange-pink brick held in the gripper (V309-V316); brick in the bowl and green and blue bowls (V317-V320); bricks on green and blue bowls (V321-V332).
- V333-V363 (17:55-18:25; sheets 27-29; first viewing, sheet 29 re-viewed): bricks on bowls and the dark table (V333-V356); V357 w_181951Z bright with red fingers and a brick; V358-V362 near-identical dark frames with green and blue bowl edges; V363 w_182551Z dark with a red-orange brick (last wrist segment of session 2).

## Friend-sim attempts (sheets 30-55, first viewing; sheet 55 one item)
- Attempts 0000-0149, overhead.mp4 and wrist.mp4 in every sheet viewed (sheets 30-55). Overhead = workspace from above (bowls, bricks, arm, keyboard at left). Wrist = orange gripper close up over the bricks. Names match. No black or frozen frames seen. SIM0149 wrist (sheet 55) matches.

## Image folders (one picked image per folder; both image sheets, first viewing)
- I001-I003 data/anomaly/real/held/crops (gw, mat, raw): close crops of a yellow brick in the gripper.
- I004 data/anomaly/real/label_sheets: contact grid of labelled crops with ID text.
- I005-I007 data/anomaly/real/spot/crops (gw, mat, raw): close crops of a blue bowl and an orange brick.
- I008 data/anomaly/real/spot/overview: pre and post workspace overview.
- I009 data/anomaly/synthetic/test/defect, I010 .../test/good, I011 .../train/good: synthetic brick renders on a dark background (defect status not judged).
- I012 data/inspection/views: 2x2 grid of inspection camera views.
- I013 data/studio-datasets-report/V1/sheets: dense thumbnail sheets.
- I014 data/studio-datasets-report/V2: workspace overview with a marker check image.
- I015 logs/depth/servo [loose files]: bricks in a bowl with the gripper.
- I016 logs/gm: bricks on a dark table.
- I017 logs/monitor [estop_top]: top view, arm and bowls.
- I018 logs/monitor [estop_close]: close view, brick in the gripper.
- I019 logs/monitor [estop_wrist]: wrist view over a green bowl.
- I020 run (mserver): close view of bricks with a gripper finger.
Each picked image matches its folder name.

## Result
- Visual spot checks: no black, frozen or missing frames found. Name-versus-view mismatch: one flag, V365 (gm wrist_green_143045.mkv, looks like a fixed overhead view). Camera roles by serial and by sim name recorded above.
- Open limitations: visual checks cover every sheet at least once, but sheet descriptions for sheets 08-26 and 28 are from the first viewing and were not re-checked after the image was removed from context.
