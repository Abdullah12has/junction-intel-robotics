#!/usr/bin/env python3
"""P3-TWIN-CAMS E1 step 2b: peek frames for viewing at six still-arm windows (seek by pts index, read-only).

Frame for a window = decoded frame whose pts is nearest the window centre (joint time, offset not yet known; the arm
is still over the window so a small clock offset does not change the pose).
Writes frames/peek_<cam>_<k>.jpg (full resolution, BGR as decoded) and prints the windows used.
Run: envs/data-env/bin/python -I tools/s2b_peek.py
"""
import json

import cv2
import numpy as np

R = "/home/intel/intel-junction-2026"
E1 = f"{R}/work/P3-TWIN-CAMS/E1"
VID = {"top": f"{R}/logs/monitor/top_20261114T170351Z.mkv",
       "close": f"{R}/logs/monitor/close_20261114T170351Z.mkv"}


def pick_windows(segs, k=6):
    long_ = [s for s in segs if s["dur"] >= 1.0]
    pick = np.linspace(0, len(long_) - 1, k).round().astype(int)
    return [long_[i] for i in pick]


def main():
    cv2.setNumThreads(4)
    with open(f"{E1}/out/still_windows.json") as fh:
        segs = json.load(fh)["segments"]
    chosen = pick_windows(segs)
    for cam in ("top", "close"):
        with open(f"{E1}/out/pts_{cam}.csv") as fh:
            pts = np.sort(np.array([float(line.split(",")[0]) for line in fh if line.strip()]))
        cap = cv2.VideoCapture(VID[cam])
        for k, s in enumerate(chosen):
            tc = 0.5 * (s["t0"] + s["t1"])
            idx = int(np.argmin(np.abs(pts - tc)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = cap.read()
            if not ok:
                print(cam, k, "read failed at index", idx)
                continue
            cv2.imwrite(f"{E1}/frames/peek_{cam}_{k}.jpg", fr)
            print(cam, k, "episode", s["episode"], "t_centre", round(tc, 3), "pts_used", round(float(pts[idx]), 3),
                  "pose", s["pose_median"])
    for k, s in enumerate(chosen):
        print("window", k, "episode", s["episode"], "t0", round(s["t0"], 3), "t1", round(s["t1"], 3))


if __name__ == "__main__":
    main()
