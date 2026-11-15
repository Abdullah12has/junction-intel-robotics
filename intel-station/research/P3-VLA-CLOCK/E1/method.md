# P3-VLA-CLOCK E1: method and result (clock-offsets-v1)

Builder E1. Script: R/work/P3-VLA-CLOCK/E1/tools/clock_offsets.py (modes inventory, analyze, visual, visual2). Interpreter: R/envs/lerobot-env/bin/python -I, nice -n 19, OMP_NUM_THREADS=4. Read only: R/logs/sort/any_1731 and any_1740 (joints.csv), R/logs/monitor/{top,close}_20261114T*.mkv, R/data/wrist/raw/w_*.mkv. Nothing else opened. No robot, camera device, serial port or Studio.

## Clock references (verified with ffprobe, probe_files.log)
- Monitor top and close files from 170351Z: absolute pts (start_time 1794675831.767 s). Wrist files from 170351Z on: absolute (start 1794675831.757 s). Earlier wrist files (before 170351Z) are relative; none is used here, because every episode window in this set lies after 17:31Z and all camera coverage is absolute (inventory.log).
- Container duration on absolute files is an end time, not a length; not used.

## Signals
- Joint speed: |d pos0..5 / dt| on joints.csv (about 28 Hz), 3-sample smoothed, on a 10 ms grid.
- Camera motion energy, primary: mean absolute difference of consecutive 160x120 gray frames (blurred, mean removed). Each frame pair is stamped at the midpoint of its two timestamps. The v1 run stamped it at the later frame, which biased lags by about half a frame; v1 is kept as offsets_v1_frame_end_timing.csv and is not used.
- Camera energy, secondary: fraction of pixels changed by more than 12 gray levels. Used as a robustness check (lag_frac_ms).

## Lag estimate
- Pearson cross-correlation of joint speed against camera energy over lags -500 to +500 ms (10 ms steps, parabolic refinement of the peak). Sign: lag = camera time minus joint time for the same motion. Positive means the camera stamps the motion later.
- Split-half lags: the same correlation on the first and second half of each window (lag_h1_ms, lag_h2_ms).
- Event estimate: threshold crossings (30 percent of the 10th to 95th percentile range) of joint speed and of camera energy, paired within 0.5 s; median paired difference is event_lag_ms. Agreement with the correlation lag counts as "yes" when within 50 ms.

## Reliability rule (fixed before the results were read)
A pair counts as reliable when the camera is on the absolute clock, the correlation peak is at least 0.4, and the lag is not at the search edge. Close fails the peak rule for 12 of 15 episodes and is reported as not determined.

## Results (metrics.json, stage clock-offsets-v1, offsets.csv: 45 rows, 15 episodes x 3 cameras)
- Episodes with joints.csv: 15 (any_1731 ep_0001, ep_0002; any_1740 ep_0001, 0004, 0006, 0007, 0009 to 0017). Skipped 7 with no joints.csv (any_1731 ep_0003, 0004; any_1740 ep_0002, 0003, 0005, 0008, 0018).
- Wrist: 15 of 15 reliable. Median lag +7.5 ms (IQR 1.0 to 11.9 ms, range -15.7 to +25.6). Median peak 0.66. Split-half difference median 10.0 ms. Event agrees within 50 ms in 14 of 15.
- Top: 11 of 15 reliable (not reliable: any_1731 ep_0001, ep_0002; any_1740 ep_0001, ep_0004). Median lag +21.5 ms (IQR 20.1 to 30.4, range 7.4 to 45.4). Median peak 0.71. Split-half difference median 15.5 ms. Event agrees in 11 of 11 with an estimate.
- Close: 3 of 15 reliable. Median peak 0.46 over all, but the lag scatters from -382 to -9 ms in the reliable set. Not determined; see below.
- Top minus wrist, reliable episodes (11): median +19.7 ms, std 13.2 ms.
- Within one frame (33.3 ms) of the median: wrist 15 of 15, top 11 of 11 reliable.

## Conclusion
1. Wrist and top each fit one near-constant offset within about one frame: wrist about +7 ms (camera stamps 0 to 1 frame after joints), top about +22 ms (about 0.7 frame). Per-episode offsets are not needed at the one-frame level; the remaining episode scatter is about the split-half noise (10 to 16 ms).
2. Close cannot be assigned an offset from this method. Its lag is not stable across episodes, and the frames show the red fingers only at the top edge with little whole-frame change (visual check, ep_0017), so whole-frame energy carries little arm signal. A close-specific method (fingertip or brick tracking) is needed before a close offset is claimed.
3. Relation to the P2 estimate: P2 R4 gave wrist -103 ms (goal5) and -161 ms (pos5) for ep_0017 using a jaw-gap half-travel method. This method gives -2.5 ms for the same episode, a difference of +101 and +158 ms (metrics.json delta). The P2 result is not reproduced. The P2 method is the more fragile one (1 px red-mask gap in ep_0015; its own limit is 2 frames), so this estimate should replace it for the wrist, pending an independent check.

## Visual check (required: at least 3 episodes)
- Frame strips with the joint-speed trace (plots/visual_*.png, first slow ramp selection) for ep_0017 wrist, top and close, ep_0006 wrist and top, ep_0012 wrist and top, ep_0010 close (8 strips in total). Viewed: ep_0017 wrist, top and close; ep_0006 and ep_0012 were generated but not viewed.
- Difference-tile strips (plots/visual2_*.png, joint onset = largest joint-speed onset in the window) were generated for ep_0017, ep_0006 and ep_0012 (wrist, top, close). Their per-pair energy values are in visual2.log. Not viewed as images in this pass; the numbers are the evidence used.
- Reading: in ep_0017 wrist, pair energy stays near 0.8 until the midpoint at -14 ms and rises to 5.8 and 7.4 at +54 and +86 ms; in top it rises at +72 ms. The image change therefore appears about 50 to 90 ms after the joint threshold crossing, not at 0 to 20 ms. The joint onset is defined on a slow ramp, so the visual onset is ambiguous to about 1.5 frames. The visual check does not confirm the correlation lag to better than about one frame, and it cannot confirm a close offset.

## Comparison with P3-TWIN-CAMS (requested by Opus)
Not done in this pass. The file work/P3-TWIN-CAMS/E1/clock_offsets.json sits in another task's folder, and this task's isolation rule (R1) and L-029 forbid reading it unless the spec lists it as an input. Opus: publish it to shared/ or add it as an input, then the comparison (top +1 frame, close 0 to 1 frame in theirs; top +22 ms, wrist +7 ms, close undetermined in mine) can be done.

## Limits
- One session, one run of each mode; no repeat on other recordings.
- The joint speed is a finite difference of 28 Hz data; the pos channels come from the follower and lag the command, but both sides of the comparison use measured joints, so this does not bias the offset.
- Energy-based correlation is sensitive to scene changes that are not the arm (lighting, people). No hand check was done on these episodes.
