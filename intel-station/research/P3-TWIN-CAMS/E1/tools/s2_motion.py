#!/usr/bin/env python3
"""P3-TWIN-CAMS E1 step 2: per-frame global motion energy and brightness of one recording (sequential decode).

Decoded frame i (presentation order) gets the i-th smallest absolute pts from the ffprobe packet list
(out/pts_<cam>.csv, written by ffprobe -show_entries packet=pts_time,flags). Read-only on the recording.
Run: envs/data-env/bin/python -I tools/s2_motion.py top|close
"""
import sys

import cv2
import numpy as np

R = "/home/intel/intel-junction-2026"
E1 = f"{R}/work/P3-TWIN-CAMS/E1"
VID = {"top": f"{R}/logs/monitor/top_20261114T170351Z.mkv",
       "close": f"{R}/logs/monitor/close_20261114T170351Z.mkv"}


def main():
    cam = sys.argv[1]
    cv2.setNumThreads(4)
    with open(f"{E1}/out/pts_{cam}.csv") as fh:
        pts = np.sort(np.array([float(line.split(",")[0]) for line in fh if line.strip()]))
    n = len(pts)
    cap = cv2.VideoCapture(VID[cam])
    motion = np.zeros(n, np.float32)
    bright = np.zeros(n, np.float32)
    prev = None
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32)
        if i < n:
            bright[i] = g.mean()
            if prev is not None:
                motion[i] = np.abs(g - prev).mean()
        prev = g
        i += 1
    gaps = np.diff(pts)
    big = np.flatnonzero(gaps > 0.05)
    missing = int(np.sum(np.round(gaps[big] / (1.0 / 30.0)) - 1))
    np.savez(f"{E1}/out/motion_{cam}.npz", pts=pts[:min(i, n)], motion=motion[:min(i, n)],
             bright=bright[:min(i, n)])
    print(cam, "decoded", i, "pts", n, "pts_first", round(pts[0], 3), "pts_last", round(pts[-1], 3))
    print(cam, "gaps>50ms", len(big), "missing_frames_est", missing, "largest_gap_s",
          round(float(gaps.max()), 3) if gaps.size else None)


if __name__ == "__main__":
    main()
