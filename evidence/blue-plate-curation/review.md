# Blue plate recording review

Reviewed 2026-11-14. This is a read-only review of the original Studio dataset and a record of the separate curated copy.

## Source and curated copy

- Source dataset: `8a8bb05e-6f2c-44f5-bf4b-745c2028f7cf` (`ALL-BLOCKS-BLUE - three cameras`)
- Source: `/home/intel/.local/share/physicalai/datasets/8a8bb05e-6f2c-44f5-bf4b-745c2028f7cf`
- Curated LeRobot copy: `/home/intel/junction-handoff/blue-plate-curation/ALL-BLOCKS-BLUE-curated-5`
- Imported Studio dataset: `e60f4087-d620-4f1d-885f-1620cefd54bd`
- Original remains unchanged. All 12 original files (987,671,563 bytes) were SHA-256 checked after curation against the recorded source hashes.

## Review result

The source contains six episodes and 16,943 frames at 30 fps. Each episode has the requested task text, `pick up all the blocks and put them on the blue plate`. The three camera keys are preserved at 640×480: RealSense `243722064316`, external `USB2.0_CAM1`, and RealSense `243622061421` (overhead).

Overhead video was sampled at each episode's start, middle, near-end, and true tail. Start frames and proposed lead-in cut frames were also checked for episodes with a stationary opening. The final overhead frames show all blocks on the blue plate for source episodes 0, 1, 3, 4, and 5. Episode 2 is a partial failure: its final frame still shows a red block and a white block outside the blue plate, near the green plate. Episode 2 remains in the untouched source and is excluded from the success-training copy.

The overhead image places the blue plate on image-left and the green plate on image-right. Review and task labels use plate color; no image flip or spatial crop was applied. The USB2.0_CAM1 sample is a very tight close-up and is less useful for whole-workspace review, but its original camera stream is preserved.

## Curation

Only verified stationary lead-in frames were removed. Candidate cut frames matched the initial overhead pose, and the cut leaves about 0.5 seconds before the first adjacent action/state change greater than one normalized unit:

| Source episode | Outcome | Trimmed lead-in | Curated length |
|---:|---|---:|---:|
| 0 | Successful | 46 frames / 1.533 s | 2,888 |
| 1 | Successful | 26 frames / 0.867 s | 3,818 |
| 2 | Partial failure; held out | — | — |
| 3 | Successful | 0 | 2,879 |
| 4 | Successful | 0 | 2,584 |
| 5 | Successful | 11 frames / 0.367 s | 2,482 |

No episode tails or internal pauses were removed because motion continued near the recorded ends. The curated copy contains five episodes and 14,651 frames. Video files were copied byte-for-byte; metadata time offsets were adjusted for the three lead-in trims. All camera feature names were retained. Numeric statistics were recomputed from retained rows; camera summaries were aggregated from per-episode source statistics, with counts adjusted for retained lengths. The three short lead-in trims were not decoded again for pixel-stat recomputation.

## Integrity checks

- Curated copy loaded successfully through Studio's read-only `InternalLeRobotDataset` loader.
- Episode ranges are contiguous; frame indices restart at zero per episode, timestamps are monotonic at 30 fps, and global row indices are contiguous.
- Actions and observations are finite. Source state/action differences remain intact; values were not clamped or otherwise altered.
- All seven copied video files match their source SHA-256 hashes. The files were not re-encoded or spatially cropped. A shared video file may still physically contain the excluded episode's adjacent interval, but the curated episode table does not reference that interval.

Detailed mapping, source hashes, and camera offsets are in `curation-manifest.json` beside the curated copy.
