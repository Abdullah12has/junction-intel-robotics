# P3-TWIN-CAMS E2 laptop log (UTC)

- 20:33Z session restart. Re-check: out/ held 2 s21 files only; tools s21, s22, s23 present; no E2 process running.
- 20:34Z s22 zero run 1: crash, initial RANSAC pose outside least-squares bounds. Fix: clip x0 into bounds (edit_block, tools/s22_held_fit.py).
- 20:35Z s22 zero run 2: OFF sweep, LOEO. Pooled median 5.452 px, p90 12.982 px over 773 frames (G3 FAIL). Full fit at bounds, camera 4.2 m away, below the mat (out/s22_top_zero.json, out/s22_zero.log).
- 20:36Z s23 bootstrap + ICP: 192 bootstrap starts, 5 candidates with plate rms < 25 px, best hit fraction 0.037. LOEO pooled median 16.153 px, p90 28.382 px over 2,071 frames (G3 FAIL) (out/s23_top.json, out/s23.log).
- 20:36Z MuJoCo check: only envs/sim-env has mujoco. Method C not run.
- 20:36Z mesh check: 13 STL present, binary; triangle counts from header. R/work/P3-TWIN-ROBOT/E1/STATUS.md absent, so sha claim not verified.
- 20:38Z STATUS.md, g3_report.json, metrics.json written (sandbox and laptop). Camera files R/sim/twin/cameras not changed.
