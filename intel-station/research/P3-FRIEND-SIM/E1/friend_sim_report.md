# Friend's sim vs real cell (P3-FRIEND-SIM, worker E1, stage friend-sim-v2)

Status: PARTIAL. Data snapshot: the live mirror was inventoried at 19:29-19:31Z and re-checked at 20:18Z (the tar is unchanged). compare.json and metrics.json are as of 20:01Z. The twin column is removed (R/twin deleted at 19:48Z, D-102). The live mirror keeps syncing every 120 s until 2026-11-16 06:00Z, so these figures describe a snapshot. JPG binaries exist on the laptop only; their sha256 values are in side_by_side/MANIFEST.txt. The authoritative copies are in R/work/P3-FRIEND-SIM/E1/ on the laptop; the sandbox copies are checked by sha256.

Scope: sim = R/data/sim/live (matches R/data/sim/2026-11-14 on every file checked). Real = Studio dataset 6d6bdce3 (16 episodes, 10158 frames; image checks sample 1 frame per second from all four chunk-000 files per camera, 340 samples per view) and R/logs/sort/any_*/ep_*/joints.csv (15 usable files). Gate: app/simgate/sim_gate.py, copied unmodified to tools/sim_gate_base.py; its output is identical to the existing gate JSON.

## 1. Bottom line
- Version: the live mirror matches the 2026-11-14 copy on all 150 attempts (five files each), the tar (sha256 36fbe792...93780b), the 762-entry manifest and all seven source files. The 2026-11-14 copy also holds a 14-file extract-lerobot folder that the live mirror does not.
- Gate re-run: a (SmolVLA co-training) NO-GO; b (ACT pre-training or augmentation) NO-GO; d (synthetic detector) NO-GO; c (geometry FK cross-check) GO; e (default: keep sim out of training) GO. This matches D-073: NO-GO for any training use, GO only as an FK cross-check.
- Top view: close to the real cell. Plate spacing agrees: 333.1 px in the sim and 333.8 px in the real view (ratio 1.002). Through the sim camera model, 343.4 mm and 343.5 mm (the sim's true bowl spacing is 341.3 mm). Both real plates sit about 6-7 px from the sim's plate centroids (blue 5.9 px, green 7.1 px; 6.2 mm and 7.0 mm through the model). The model also places the sim's own plate centroids 6.6-6.8 px (7-8 mm) from the bowl centres, so the mm figures carry about 7 mm of model bias; the px differences are the reliable numbers. A shared shift like this fits a camera offset in the sim model or both plates being placed about 6 mm off; the data cannot separate the two.
- Close view: not true to reality. The friend's comment says the close camera was set to match the observed close view and that its mounting is unverified (station_scene.py lines 76-77). The real green plate centroid sits 76 px from the sim's (gate: 81 px). Real close-view centroids are approximate: the green plate is cut by the top and right edges of the frame.
- Brightness: the sim is 41.7% darker in the top view and 51.0% darker in the close view (mean luma 51.9 vs 89.1, and 38.0 vs 77.4).
- Joints: the sim's wrist limit (91.6 units) is below the real maximum (97.4); the sim never reaches the real rest pose (lift -98.5, elbow 96.8, gate reference); the sim gripper span is 10.4% larger (ratio 1.104); the start-pose convention differs.
- Motion: the sim's five arm joints reach 2.4-4.6x the real p95 speed. The real arm joints are still for 69-84% of frames; the sim's for 30-65%.
- Routing: 73 of 150 sim episodes contradict OD-10 (40 2x2 bricks sent to green, 33 2x4 bricks sent to blue). The sim has no tape, so a defect is a non-2x2 brick only.
- Usable for: a top-view plate-layout sanity check only. Not usable for training or detector data (decisions a, b, d).

## 2. Version identity (inventory)
- The 150 attempts in each root are identical for metadata.json, trajectory.npz, trace.json, overhead.mp4 and wrist.mp4 (sha256 per file; inventory_compare.json and compare.json version_identity).
- Tar, manifest and source: tar sha256 equal in both roots; SHA256.json verifies 762/762; the seven source files are unchanged; recording_context keys are unchanged.
- Re-check at 20:18Z: the live tar is still sha256 36fbe79242afa86ed29247fd32327041780e0937e586b1c014962bc74b93780b. The live SHA256.json is sha256 9fc8a07ef5c83435961e1cfa875d93ff953237ed9077fc397c0263d10ac2e566 (first recorded hash for that file). Live still holds 763 files, 1.3 GB, unchanged in count since 19:25Z.
- Only in R/data/sim/2026-11-14: extract-lerobot/dataset-150, 14 files (examples: a 13 KB stats JSON, a 3.9 MB data parquet, a 0.6 MB episode-metadata parquet). Not analysed.
- Videos: overhead.mp4 and wrist.mp4 are h264, 640x480, 30 fps, 150 each (L04, L05). The camera named wrist is not on the arm: station_scene.py removes it from the gripper body and appends it to the world body (lines 78-80), so it does not move with the arm.
- Episodes: 360-367 frames for 147 attempts and 403-405 frames for three. Median 362 frames (12.07 s), mean 12.11 s, longest 13.5 s (L17). The 361-frame figure in my first draft came from a hard-coded value in make_metrics.py and is corrected here.
- All 150 attempts are accepted, success and grasp-verified. 82 bricks 2x2, 68 2x4. Destinations: 75 blue, 75 green. Six target colours, 25 each. Actions in absolute radians. MuJoCo 4.0.1.

## 3. Gate re-run
- Passed: provenance (762/762); top camera green offset 7.2 px (limit 10 px); release detected and plate-matched for 150/150; accepted, success and grasp verified for 150/150; episode checks; FK cross-check (sim TCP to plate, median 0.39 cm for both plates).
- Failed: units (release pan residual blue 16.5 deg, green 1.7 deg; tolerance 3 deg); gripper (span ratio 1.104 against 10%; closed offset 3.49 units against 1); rest (no sim frame near the real rest); wrist (21.4% of sim frames pinned at the MJCF limit, gate method); brightness (relative tolerance 15%); close camera (81.3 px against 10 px); geometry (blue 3.79 cm, green 2.03 cm against 2.0 cm); OD-10 routing (od10_clean false); no tape or defect class in the sim (tape_class_in_sim false).
- Limit of the gate: sim_gate.py line 437 reads only chunk-000/file-000.mp4 for each real view (about 42 s of the 338.6 s dataset). Its real-view values, top 7.2 px and close 81.3 px, come from that file alone. Section 4 uses all four files.
- The gate was not modified. The image-based plate check and the all-files real pass are in tools/sim_real_compare.py (a separate tool); they are proposed for the gate (section 13).

## 4. Plate positions and camera geometry (top view)
Method: HSV masks (blue hue 80-105, green hue 50-90, saturation and value thresholds as in tools/sim_real_compare.py), 5x5 opening, largest component, centroid. Real: all four chunk-000 files per camera, sampled at 1 frame per second (340 samples per view). Sim: 1 fps over 150 attempts (1804 blue and 1797 green detections). Millimetres via the sim overhead camera model (look-at (-0.025, -0.46, 0.61) to (-0.025, -0.22, 0), fovy 40).

| Quantity | Sim | Real | Real minus sim |
|---|---|---|---|
| Blue centroid, px (median) | 240.2, 151.9 | 246.0, 150.7 | 5.82, -1.20 px (5.94 px; 6.16 mm through the model) |
| Green centroid, px (median) | 572.4, 177.5 | 578.4, 181.1 | 6.08, 3.58 px (7.06 px; 7.02 mm through the model) |
| Plate spacing | 333.1 px; 343.4 mm through the model (truth 341.3 mm) | 333.8 px; 343.5 mm through the model | 0.2% in px |
| Apparent area ratio, real/sim | blue 1.06, green 0.99 | | |

- Model check: the sim's own plate centroids sit 6.6 px (blue) and 6.8 px (green) from the centroids the model predicts for the bowl centres (7.8 and 7.4 mm through the model). The camera model is therefore not exact even for the sim, and the mm values carry about 7 mm of bias. The comparison uses only the px differences, which come from the same centroid method on both sides.
- Per-file stability (L26): blue x medians 244.3, 245.9, 245.5, 246.6 px across files 000-003 (spread 2.3 px); green x 578.5, 578.1, 578.5, 578.7 px (spread 0.6 px); green y 181.5, 181.9, 180.7, 177.7 px (spread 4.2 px).
- Outliers: at most six frames per file sit more than 20 px from the file median. The outliers include cross-colour hits (the blue plate read as green near x 240; the green plate read as blue near x 578). I did not classify the rest. They inflate the sd figures (blue x sd 40 px, green 32 px) but not the medians. Per-frame filtering has not been done.
- Plate pair geometry: spacing agrees within about 1 mm (ratio 1.002 in px). The line between the plates is rotated about 0.8 deg more in the real view; over the 333 px spacing that is about 4.6 px at the far plate. Two points cannot separate camera roll from plate placement, and no full camera calibration exists.
- D-069's "top camera within about 8 px" still holds (7.2 px by gate; 5.9-7.1 px here).

Release points and FK:
- Gate geometry (base frame, from the gate JSON): sim plate centres (0.1688, -0.1438) m blue and (0.1988, 0.1962) m green. Mean real release TCPs, from forward kinematics of the real release joint readings through the sim's MJCF: (0.1857, -0.1099) m blue and (0.2095, 0.2134) m green. Their sd is 4.4, 5.7 and 8.0 mm for blue and 15.7, 13.8 and 21.4 mm for green. The FK distance from each sim plate centre to its mean real release is 3.79 cm blue and 2.03 cm green. The mean real release points are 324.2 mm apart, against 341.3 mm for the sim's bowl centres, so the real releases sit 17 mm closer together.
- These real TCPs depend on the unit and zero convention, which is not validated (D-073 gives the zero as the range midpoint), so they inherit that uncertainty. The FK distance shows where bricks were released relative to the plate centre; it does not measure plate placement. This supports reading D-073's "blue plate 3.8 cm off" as a release offset. Inference; a per-release image check would confirm it.
- Blue pan residual: release pan medians are 48.37 units (sim) and 32.18 (real), a gap of 16.2 units, which is 16.5 deg at 56.16 units per rad. Green: -50.89 (sim) and -52.56 (real), 1.7 deg. A constant joint zero offset would move both plates by the same angle, so it cannot explain a blue residual about ten times the green one. The blue plate centre is 22.2 cm from the robot base, so 16.5 deg is about 6.4 cm of arc. The blue bowl's outer rim radius is 8.7 cm (station_scene.py add_bowl profile, about 17.4 cm across), and the blue mean release is 3.8 cm from the plate centre, which fits inside the rim. Inference only: this is consistent with off-centre blue releases, but I did not check where the blue bricks landed.

Real releases (tools/real_release_check.py, Studio joint state; pan in units):
- Blue side (pan +32.1 to +33.6): eps 7, 11, 12 (gate set) and ep 13 (D-059 says no brick; D-067 excludes it).
- Green side (pan -51.7 to -54.7): eps 8, 10, 14, 15 (gate set). Lifts 24.5 to 53.4 across the listed releases.
- Eps 0-6 release at pan -5 to -13, and ep 9 at pan -44 with lift -9. Both groups fall outside the two plates' pan groups.
- D-067's v2 sets are PP-BLUE 7, 11, 12 and PP-GREEN 14, 15. It leaves out eps 8 and 10 with no per-episode reason in the ledger, so the gate's green set (8, 10, 14, 15) is wider than D-067's.
- Gate sets agree with the sign of the release pan for every gate episode.

## 5. Close view and camera poses
- Green plate (median): sim 555.0, 137.5 (n = 1804); real 557.9, 61.3 (n = 318). Real minus sim: 76.2 px (gate 81.3 px). D-069 had 32-72 px.
- Close-view plate spacing (median centroids): sim 497.2 px, real 518.6 px (ratio 1.04). Green area ratio, real over sim: 0.66. Both depend on the clipping and cross-colour hits described below, so treat them as approximate.
- Blue plate: sim 58.3, 157.9 (x sd 25 px); real 46.4, 147.0 (x sd 82 px; per-file medians 41-50 px). Not usable for the offset.
- The real green plate is cut by the top and right edges of the frame, and detections shift (cross-colour hits at x 528-620). The sim green plate is cut by the right edge and the sim blue plate by the left edge (sim_close_a0000_f0.jpg, section 9). Treat the 76 px as approximate.
- Sim close camera (station_scene.py lines 78-82, the camera named wrist): fixed at (0.025, -0.45, 0.22), looking at (0.065, -0.22, 0.015), fovy 40. Lines 76-77 say it was set to match camera2's observed close view and that its mounting is unverified; line 3 says the geometry was visually fitted.
- Real close camera: pose not measured. Framing differs from the sim: the real bricks in the foreground are larger, and the bowls sit at different image positions (section 9).
- Overhead camera (lines 73-75): (-0.025, -0.46, 0.61) to (-0.025, -0.22, 0), fovy 40. The real top pose is checked at two plate centroids only (section 4).
- Robot base: no real base reference, so not measured. The sim base is at (0.0388, 0.0388, 0) with quaternion (0.7071, 0, 0, -0.7071) (station_scene.py lines 57-58).

## 6. Brightness and colour
- Top: sim luma 51.94 (1804 frames) vs real 89.07 (340 frames); sim 41.7% darker. Dark fraction (share of pixels with luma under 40; sim_real_compare.py line 109): sim 64.5%, real 43.3%.
- Close: sim 37.97 vs real 77.44; sim 51.0% darker. Dark fraction 68.0% vs 29.4%.
- Real luma by file: top 90.5, 90.1, 89.0, 87.7; close 78.8, 80.5, 75.3, 75.6. The gap is not a one-file effect.
- Saturation S: top sim 80.3 vs real 84.4 (sim lower); close sim 98.1 vs real 93.8 (sim higher). The sim is not uniformly more saturated.
- D-069's 32% darker cannot be reproduced; its method is not recorded. The gate's own sample (63 sim frames; 30 real frames from file-000) gives 51.8 vs 90.2 (top) and 38.3 vs 78.9 (close), consistent with the figures above.
- A possible cause is the light setup (station_scene.py lines 83-86: a light at (-0.4, -0.2, 1.5) with diffuse 0.65, and a headlight with ambient 0.4). Not tested.

## 7. Joint units, zero convention and ranges
- Conversion (as in sim_gate.py and D-073): normalized = rad x K, K = [56.16, 57.47, 59.14, 55.26, 34.08, 42.57] units per rad (gripper encoder-linear, an assumption); zero at the range midpoint; sign +1.
- Ranges, units: lift sim -42.8 to 54.6 vs real -98.9 to 61.9; elbow -56.9 to 62.4 vs -76.5 to 96.9; wrist 33.2 to 91.6 vs 29.7 to 97.4.
- Real rest (gate reference): lift -98.5, elbow 96.8. Sim frames near rest: 0%.
- Wrist: MJCF limit 1.658 rad = 91.62 units. 24.4% of sim frames sit within 0.5 units of it (gate method 21.4%). No real frame reaches 98 (max 97.37).
- Gripper: open max sim 29.8 vs real 25.1; min sim 4.0 vs real 0.5. Span ratio 1.104 (gate).
- Start pose: sim frame 0 is the same constant home [0, -28.7, 47.3, 33.2, 0, 21.3] in all 150 attempts (std 0; L25). Real first rows (median) [-12.7, 12.8, -23.7, 89.7, -62.2, 24.6]. These are different start conventions, so this does not test the zero convention.
- Wrist roll median: sim -0.9 vs real -62.1. Grasp orientation or convention; not separable here.

## 8. Motion and speed (30 Hz)
- p95 speed, units/s (pan, lift, elbow, wrist, roll): sim 31.2, 35.2, 57.1, 39.4, 24.6; real 9.6, 14.9, 18.1, 10.3, 5.4. Ratios 3.26, 2.36, 3.15, 3.82, 4.56. Gripper: sim 40.8, real 0.0 (no ratio).
- Still fraction (share of frames under 0.5 units/s), pan to gripper: sim 0.65, 0.30, 0.31, 0.45, 0.59, 0.70; real 0.82, 0.70, 0.70, 0.69, 0.84, 0.96.
- Jerk RMS, sim/real ratios for the five arm joints: 2.2, 3.0, 1.4, 2.5, 3.7 (gripper 8.5). The sim is less smooth on this measure. The real signal was resampled from about 28.6 Hz to 30 Hz and is quantised, which can change its jerk, so treat the magnitude as approximate. D-069 says "motion smoother than real" (ledger, 14:52Z); the speed and jerk figures here point the other way.
- Episode length: sim median 12.07 s (12.0-13.5 s). Studio episode videos average 21.16 s (16 episodes); joints.csv files average 51.1 s (15 files, 23,156 frames). The two real measures disagree, so the real episode length is unresolved.
- Friend's TRAINING-STATUS.md (first 1800 of about 5100 bytes read): 150 episodes; 54,474 frames per camera (equal to the sum of the sim episode lengths here); 150 of 562 preflight runs succeeded; the file says this does not demonstrate broad grasp reliability.

## 9. Visual check
Viewed: frames/real_top_t0.jpg, frames/real_close_t0.jpg, frames/sim_overhead_a0000_f0.jpg and frames/sim_close_a0000_f0.jpg (first viewed 19:35Z; re-viewed about 20:16Z), and both side-by-side composites (viewed about 20:16Z). The f150 frames were extracted but not viewed.
- Top, sim (sim_overhead_a0000_f0.jpg): black mat; blue bowl at left and green bowl at right, the green cut by the right edge; the black robot base at top centre with an orange gripper reaching into the scene; five bricks (white 2x2, red 2x4, yellow 2x2, purple 2x2, orange 2x2); a black laptop at top left; a dark box at the left; a grey bar at the bottom; orange-brown table edges.
- Top, real (real_top_t0.jpg): the same plate layout, with the blue bowl at upper left and the green bowl at right, cut by the right edge; a laptop keyboard at top left; black cables across a wooden table, which is visible at left and bottom; grey tape and black clamps along the bottom edge; a white-and-grey tool at the bottom centre; bricks of several colours (red, yellow, blue, orange, white, including a white 2x4); the robot's orange gripper with red fingers at top centre; brighter overall.
- Close, sim (sim_close_a0000_f0.jpg): the camera looks down at an oblique angle onto the black mat; the blue bowl is cut by the left edge and the green bowl by the right edge; an orange gripper jaw at top centre; a cyan brick partly behind the gripper; a white 2x2 at top left; a red 2x4 and a yellow 2x2 in the foreground; purple and orange bricks at the bottom edge; a brown table edge at the top.
- Close, real (real_close_t0.jpg): dark grey mat, not black; the blue bowl at the left edge; the green bowl at top right, cut by the top and right edges; red, yellow, blue and two orange 2x2 bricks near the top; a white 2x2 at the right; a large white 2x4 in the foreground; red gripper parts at the top, left of centre.
- Framing differs from the sim: the real bricks in the foreground are larger, and the bowls sit at different image positions. I did not measure the real camera pose.
- Both composites carry markers for the detected centroids and for the sim-model predictions. In the top composite the markers sit on the plate centres. In the close composite the real green marker lies on the green bowl and the real blue marker on the blue bowl.

## 10. Usable for and not usable for
- Usable: a top-view check of plate layout (spacing within about 1 mm; both plates about 6-7 px off as a pair, if the sim camera model holds). Usable as a check of the sim's own MJCF reach (0.39 cm), which says nothing about the real arm's kinematics.
- Not usable for SmolVLA or ACT training, or as detector data (gate decisions a, b, d). Reasons: joint and wrist limits differ; gripper span +10%; close camera not matched (76-81 px, different framing); brightness 42-51% darker; routing contradicts OD-10 in 73 of 150; no defect class; episodes about 12 s against about 21 s real (the real length is unresolved, section 8).

## 11. Comparison table
compare.json, comparison_table (24 rows: measure | friend_sim | real | note). Every value in those rows was read from the JSON by tools/patch_d102.py, not typed by hand. The row "top view, plate spacing" shows 324.2 mm in the real column; that is the spacing between the mean real release TCPs (section 4), not a plate spacing.

## 12. Twin (D-102)
- R/twin was deleted at 19:48Z (D-102). The last read-only check (19:45:13Z) found all five twin data directories empty (0 files each); app/twin had 35 .py files. A grep of the twin scene and calibration code (19:39Z) found one docstring match (calib/dataset_io.py line 3). Nothing was read or written under R/twin beyond these listings and that grep.
- Per the coordinator (19:49Z), the twin column is removed from compare.json and this report. Ours will be R/sim/twin, which Opus will compare once it passes its gates.

## 13. Requests to Opus
1. Plate geometry: add an image-based plate check (section 4) to the gate's geometry criterion. The FK release check measures release offsets and cannot show plate placement, so D-073's 3.8 cm needs review.
2. Blue release positions: check where blue bricks land on the bowl (FK offset 3.8 cm from the plate centre; pan residual 16.5 deg) before reading 16.5 deg as a sim unit error.
3. D-069's "32% darker": the method is needed. Current values: 41.7% top, 51.0% close (all four files).
4. Gate real-view sample: sim_gate.py line 437 uses file-000 only. The top (7.2 px) and close (81.3 px) values rest on about 42 s. Consider all four files (tools/patch_real_all_files.py shows the change).
5. Friend items: wrist limit (91.6 vs 97.4 units); gripper span (+10%); close-camera mounting (the friend's comment says unverified); routing against OD-10 (73 of 150); episode lengths (360-405 frames, not one fixed length).
6. Process: pid 3021971 is a bash launcher whose command line starts app/simgate/watch_sim.sh; it had been running 4 h 35 min at my check. D-073 records the automatic watch as denied for the agent. I did not start it and did not check its children; please confirm who started it. Also the P2-D-DATA inventory watch (pids 443624 and 443626, 6 h 14 min) is running; not touched.
7. Per-frame plate filter: reject detections whose hue fits the other plate, and frames with area far from the median, to tighten the sd figures.
8. Mirror: R/data/sim/live syncs every 120 s until 2026-11-16 06:00Z. The tar and SHA256.json were unchanged at 20:18Z. Re-run the version check before any training decision.

## 14. Verification and process notes
- selfcheck.py: 22 of 22 PASS (selfcheck.json; L24).
- verify_claims.py (L25): saturation top sim 80.34, real 84.40; close 98.11, 93.75; sim frame 0 identical in all attempts; arm-joint p95 ratio 2.36-4.56.
- Corrections to my first draft: episode length (now 360-405 frames, not 361); brightness frame counts 1950 and 339 (now 1804 sim, 340 real); "sim is more saturated" (now mixed); "2-4x speed" (now 2.4-4.6x on arm joints; real gripper p95 is 0); "jerk 2-3x" (now 1.4-3.7x on arm joints); real plate spacing 345.4 mm and offsets 4.3 and 7.3 px from file-000 alone (now 343.5 mm and 5.9 and 7.1 px from all four files). The file-000 sample covered about 42 s, not the dataset.
- Patches: tools/patch_real_all_files.py (real pass extended to four files; the v1 script is backed up in logs/); tools/patch_d102.py v2 (twin removed, duration corrected, table built from values). make_metrics.py line 76 still hard-codes 361/30 (superseded by the patch; not edited). Do not re-run finalize.py, which still writes the twin column.
- v1 outputs (file-000 only) kept: logs/compare.v1-file000-only.json and logs/metrics.v1-file000-only.json.
- Process notes: one call to mcp__claude-code-remote__list_repos (not on my tool list) was denied by the auto-mode classifier; I did not retry. Two read_file calls (f5aae643, 092d7b0b) stalled and were replaced by shell copies (no side effects). One inventory start_process timed out on the device; a process listing showed it had completed (no duplicate run). The first run of sim_real_compare.py had no output argument and failed before writing; it was re-run with the folder.
- Memory: no stored memory was used.

## 15. Files
Laptop, R/work/P3-FRIEND-SIM/E1/: compare.json, metrics.json, friend_sim_report.md, selfcheck.json, real_release_check.json, inventory_compare.json, inv_live.json, inv_base.json, twin_status.json, gate/2026-11-14-95fa40dbf3.json, side_by_side/top_sim_a0000_f0_vs_real_t0.jpg, side_by_side/close_sim_a0000_f0_vs_real_t0.jpg, frames/ (six JPGs), tools/ (scripts), logs/ (L03-L27 plus v1 backups).
Sandbox, /home/claude/kb/work/P3-FRIEND-SIM/E1/: compare.json, metrics.json, friend_sim_report.md, STATUS.md, laptop-log.md, side_by_side/MANIFEST.txt.

## 16. Coverage and gaps
Done: version identity (all 150 attempts, tar, manifest, sources, recording_context; live tar re-checked 20:18Z); gate re-run (identical); plate positions (all four real files per camera, 1 fps sampling); camera poses (two points, not calibrated); brightness and saturation (all four files); joint units, ranges and start pose; motion (speed, still fraction, jerk); routing; release positions and FK; visual check of four frames and two composites; twin status (empty at 19:45Z; removed per D-102).
Not done or partial: extract-lerobot (14 files, only in the 2026-11-14 copy) not analysed; the dataset tar not extracted; summary.json and verification.json (top level) not read; TRAINING-STATUS.md partly read; source files expert.py, so101_env.py, collect.py, setup_assets.py and uv.lock not read (hashes verified); per-frame outlier filtering not done; robot base pose not measured; top camera pose checked at two plates only; close view not matched; close-view spacing and area ratios approximate; f150 frames not viewed; per-release brick positions not checked; the real episode length unresolved (joints.csv vs Studio); no comparison with R/sim/twin (not built yet).
