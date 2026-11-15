PARTIAL 2026-11-14T20:44Z

# P3-TWIN-CAMS E2 status (final)

Outcome: no camera passes G3. G3 is parked until the robot returns (5-minute checkerboard capture). R/sim/twin/cameras/top.json and close.json are unchanged (E1 state).

Method A, held-brick correspondences (3,874 frames, 5 episodes, TCP = kin.fk, brick centre TCP + 1.6 mm, clock +0.0333 s):
- s22, zero joint offsets, global PnP-RANSAC, Huber, LOEO: pooled median 5.452 px, p90 12.982 px over 773 frames. FAIL. Full camera unphysical (fx = fy at bound, k1 at -0.3, 4.2 m away, below mat).
- s23, plate-centre bootstrap + held-brick ICP, LOEO: pooled median 16.153 px, p90 28.382 px over 2,071 frames. FAIL. Full camera 530 mm high, tilt 32 deg, cx and k1 at bounds.
- s24 (final, requested by Opus): physical intrinsics priors (fx = fy 615 +- 40 px, principal point within 30 px of centre, |k1| <= 0.15), 5 arm joint offsets free (prior and sigma 3 from the wrist-camera fit), camera pose free, Huber, LOEO by episode. LOEO pooled median 20.112 px, p90 28.236 px, 1,846 held-out frames, in-gate fraction 0.477. FAIL.
  - Full-data fit: camera 450 mm above the mat, tilt 32.4 deg, depth along axis 532.5 mm (prior 589 mm, 20 deg): does not match the prior. fx = fy at 655 (bound), cx 308, cy 270 (bound), k1 0.15 (bound).
  - Joint offsets fit: [20.01, 0.25, 10.73, -12.27, 20.00] against prior [0.013, -5.642, -2.486, -2.758, 0.001]. Four of five are at or near the +-20 bounds, 3 to 6.7 sigma from prior. Degenerate.
- Common cause across s22, s23, s24: the held-brick TCP-to-blob association does not agree with a plausible top camera, even with free joint offsets. Likely causes: colour-blob contamination (cables, gripper head, plates), the TCP + brick-centre model, or the top clock offset. Not resolved.

Not done (by request, G3 parked): method B as a separate fit; method C (MuJoCo only in envs/sim-env; no fitted camera to compare); method D overlays; close and wrist held passes; clock_offsets.json not updated (top +0.0333 s stands).

Mesh sha claim: NOT verified. 13 STL files present (binary; triangle counts 1254 to 53994). R/work/P3-TWIN-ROBOT/E1/STATUS.md is absent on the laptop, so the LFS match could not be checked.

Next (when the robot returns): a 5-minute checkerboard capture for top and close intrinsics and extrinsics, then re-run G3.

E2 files: R/work/P3-TWIN-CAMS/E2/out/s22_top_zero.json, s23_top.json, s24_top.json and logs; g3_report.json; metrics.json; laptop-log.md; STATUS.md.
