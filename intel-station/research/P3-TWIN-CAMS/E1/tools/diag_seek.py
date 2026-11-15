#!/usr/bin/env python3
"""P3-TWIN-CAMS E1 step 9 diagnostic 4 (read-only): is OpenCV seek (CAP_PROP_POS_FRAMES) frame-accurate on the
recordings? Compares, for a few target frame indices, the frame returned after a seek with the frame obtained by
sequential decoding, and reports the mean absolute pixel difference and the timestamp of each.
Run: envs/data-env/bin/python -I tools/diag_seek.py <top|close>   (uses cv2 only)
"""
import sys

import cv2
import numpy as np

R = "/home/intel/intel-junction-2026"
E1 = f"{R}/work/P3-TWIN-CAMS/E1"
VID = {"top": f"{R}/logs/monitor/top_20261114T170351Z.mkv", "close": f"{R}/logs/monitor/close_20261114T170351Z.mkv"}
TARGETS = [300, 2000, 6000, 12000]


def main(cam):
    pts = np.loadtxt(f"{E1}/out/pts_{cam}.csv", delimiter=",", usecols=0)
    print(cam, "pts count", len(pts), "first", round(float(pts[0]), 4), "last", round(float(pts[-1]), 4), flush=True)
    cap = cv2.VideoCapture(VID[cam])
    print(cam, "CAP_PROP_FRAME_COUNT", cap.get(cv2.CAP_PROP_FRAME_COUNT), "fps", cap.get(cv2.CAP_PROP_FPS), flush=True)
    seq = {}
    n = 0
    last = max(TARGETS)
    while n <= last:
        ok, img = cap.read()
        if not ok:
            break
        if n in TARGETS:
            seq[n] = (img.copy(), cap.get(cv2.CAP_PROP_POS_MSEC))
        n += 1
    print(cam, "sequential frames read", n, flush=True)
    for t in TARGETS:
        cap2 = cv2.VideoCapture(VID[cam])
        cap2.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, img = cap2.read()
        pos_after = cap2.get(cv2.CAP_PROP_POS_FRAMES)
        if not ok or t not in seq:
            print(cam, "target", t, "read failed or missing", flush=True)
            continue
        mad = float(np.mean(np.abs(img.astype(np.float32) - seq[t][0].astype(np.float32))))
        print(cam, "target", t, "seek MAD vs sequential", round(mad, 2), "pos after seek", pos_after,
              "pts[t]", round(float(pts[t]), 4) if t < len(pts) else None, "seq msec", round(seq[t][1], 1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
