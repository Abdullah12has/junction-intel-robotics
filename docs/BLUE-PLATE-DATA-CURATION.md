# Blue-plate demonstrations: curation handoff

Date: 14 November 2026, Helsinki. Status: curated export completed and imported into Studio. Review used sampled start/mid/end and exact-tail frames, not a frame-by-frame certification of every grasp.

## User intent

Human demonstrations of picking up all blocks and placing them on the blue plate. User describes the blue plate as on the right from their station viewpoint; it appears on the left in the overhead image. Identify the destination by colour, not image left/right. Do not flip camera images.

User requested a cheaper subagent to review, clean and crop the data. Review delegated to `gpt-6-luna`; parent independently checked numeric integrity and sampled final frames. Interpret crop as conservative temporal trimming, not spatial image cropping. Keep all three views and preserve original recordings.

## Original dataset and backup

- Studio project: `ab767473-d6fd-47ff-812e-8239c455dabb`.
- Dataset: `8a8bb05e-6f2c-44f5-bf4b-745c2028f7cf`, **ALL-BLOCKS-BLUE - three cameras**.
- Environment: `3507e384-4abf-488d-be91-0d235fee2182`.
- Task: `pick up all the blocks and put them on the blue plate`.
- Laptop source: `/home/intel/.local/share/physicalai/datasets/8a8bb05e-6f2c-44f5-bf4b-745c2028f7cf`.
- After user confirmed the batch finished, stopped its recording session through Studio (HTTP 204) to finalize the cache and publish the episodes. Do not reuse that session ID for future controls.
- Finalized source contains **6 episodes, 16,943 frames at 30 FPS**, three 640×480 video streams, LeRobot v3 format.
- Mac backup: `/Users/abdullah/Desktop/junction/station-backups/2026-11-14-blue-plate/original/`.
- Backup verified: **12 files, 987,671,563 bytes, zero checksum mismatches**. Manifest `original-sha256.json` and result `backup-verification.json` are beside the backup folder.

## Review findings

| Studio episode | Source index | Frames | Duration (s) | Decision |
|---|---:|---:|---:|---|
| E1 | 0 | 2934 | 97.80 | Keep: sampled final frame shows blocks on blue plate |
| E2 | 1 | 3844 | 128.13 | Keep: sampled final frame shows blocks on blue plate |
| E3 | 2 | 2209 | 73.63 | Exclude from success-only copy: red and white blocks remain outside blue plate at the end |
| E4 | 3 | 2879 | 95.97 | Keep: sampled final frame shows placement completed |
| E5 | 4 | 2584 | 86.13 | Keep: sampled final frame shows blocks on blue plate |
| E6 | 5 | 2493 | 83.10 | Keep: sampled final frame shows blocks on blue plate |

All source actions and observations are finite and within the expected normalized ranges. Row counts and episode/frame indices are consistent. These checks do not alone establish real-time synchronization or certify every intermediate grasp.

Several runs still move at the end: do not apply an arbitrary tail trim. Reviewed leading trims: source 0 removes 46 frames, source 1 removes 26, source 5 removes 11; sources 3 and 4 keep their starts. These retain approximately 0.5 seconds before the first >1-normalized-unit state change, with sampled visual confirmation. No tail or internal-pause cuts. Expected curated result: **5 episodes, 14,651 frames**, excluding source 2 and removing 83 leading frames total. Preserve the incomplete episode in the original dataset, not as a successful full-task demonstration. Review images currently live in `/tmp/blue_review_frames/`; these are temporary, not the durable dataset backup.

Validated laptop output: `/home/intel/junction-handoff/blue-plate-curation/ALL-BLOCKS-BLUE-curated-5`.

## Completed output

- Studio name: **ALL-BLOCKS-BLUE - curated three cameras**.
- New dataset ID: `e60f4087-d620-4f1d-885f-1620cefd54bd`.
- Studio path: `/home/intel/.local/share/physicalai/datasets/e60f4087-d620-4f1d-885f-1620cefd54bd`.
- Mac copy: `/Users/abdullah/Desktop/junction/station-backups/2026-11-14-blue-plate/curated/`.
- Curation details and source checksums: `curation-manifest.json` inside the cleaned output.
- Parent independently verified 14,651 rows across 5 episodes, exact retained action/state values, contiguous global indices, zero-based episode timestamps, all three cameras' adjusted start offsets, and recomputed action/state means and standard deviations.
- Studio CLI import succeeded; the episode API lists all five episodes with three video references each.
- Video payloads are retained byte-for-byte. Trimming is expressed through episode metadata and selected rows, avoiding lossy re-encoding. Shared physical video files still contain excluded intervals, but those intervals are not referenced by the curated dataset episodes.

## Remaining scope

No training requested or launched in this task. This curation screens outcomes and removes short idle lead-ins; it does not prove sensor acquisition synchronization or simulation-to-real generalization. Retain the original six episodes and backup for further review.

## Recording setup context

Three streams: RealSense `243722064316` (close), RealSense `243622061421` (overhead), and `USB2.0_CAM1` (wrist). Both RealSense devices were verified as USB 3.2 after the overhead camera was reconnected. The separate wrist FFmpeg recorder was stopped using `/home/intel/junction-cell/recording/STOP` so Studio could own that camera. Recheck ownership before any future camera/session changes.
