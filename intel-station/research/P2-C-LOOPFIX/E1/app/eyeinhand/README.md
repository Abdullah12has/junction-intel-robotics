# Eye-in-hand wrist camera model (offline, I-61)

Fits the wrist camera (pinhole + one radial term), its pose in the TCP frame of `app/tools/kin.py`, and five joint
offsets, from the sort loop's own recordings. No calibration target is needed. The observations are the target
brick's blob centre:

- in still-arm wrist frames, taken from the continuous MJPEG segments (absolute pts since 17:04Z) and matched to
  `joints.csv`;
- in each episode's scan frame (`pre_wrist.jpg` at `q_scan`).

## Run (offline; scipy lives in the anomalib env)

```
OMP_NUM_THREADS=4 nice -n 19 envs/data-env/bin/python -B app/eyeinhand/collect.py      # -> work/eyeinhand/obs.json + frames
OMP_NUM_THREADS=4 nice -n 19 envs/anomalib-env/bin/python -B app/eyeinhand/calib3.py   # -> work/eyeinhand/wrist_cam_model_v3.json, scan_check.jpg, assoc.json
```

## Result at 18:47Z (8 episodes, 126 observations)

- In-sample reprojection: median 4.0 px; scan frames 3.8 px.
- Fitted values: f 392.8, cx 336.7, cy 232.9, k1 −0.152.
- Camera position in the TCP frame: (0.1, 61.8, −24.1) mm.
- Joint offsets (normalized units): [0.0, −5.6, −2.5, −2.8, 0.0].
- `scan_check.jpg`: the brick position fitted from the hover and descent frames, projected into every scan frame,
  falls on the brick the loop chose in all 7 episodes (11–23 px from the blob centre).
- **Runtime-path leave-one-episode-out** (scan blob at `q_scan` → TCP hover target, compared with the
  servo-converged hover): 1.2–24.3 mm, median 11.5 mm. The learned similarity scan map now in `sort_any.py` scored
  13.8 mm median on its own leave-one-out. With 8 positions the joint/camera parameters generalize only modestly.

## Switch criteria (not used by the loop yet)

Refit when there are at least 20 episodes. Make `wristcam.tcp_target` the scan map in `sort_any.candidates` only
when the runtime-path LOO median is ≤ 6 mm and the max is ≤ 15 mm. The servo stays the closed-loop correction
either way.
