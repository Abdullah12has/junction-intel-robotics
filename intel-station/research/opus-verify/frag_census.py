"""Opus: for every wrist fixture frame, list the real-pixel pieces that the 15 px bridge joins into one blob.
Used to choose a fragment rule for pick_any.blobs() (F-1 of P2-B-TESTS) from data, not by guess."""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/intel/intel-junction-2026/app/tools")
import grasp_auto as ga  # noqa: E402
import pick_any as pa  # noqa: E402

FIX = "/home/intel/intel-junction-2026/app/tools/tests/fixtures"
files = sorted(glob.glob(f"{FIX}/servo/*.png") + glob.glob(f"{FIX}/split/*.png") + glob.glob(f"{FIX}/hover/*.png")
               + glob.glob(f"{FIX}/servo/*.jpg") + glob.glob(f"{FIX}/split/*.jpg") + glob.glob(f"{FIX}/hover/*.jpg"))
print("frames", len(files))
for f in files:
    img = cv2.imread(f)
    if img is None:
        continue
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    jaw = ga.JAW_MASK if ga.JAW_MASK is not None and ga.JAW_MASK.shape == H.shape else np.zeros(H.shape, bool)
    for cname, fn in pa.COLOUR_CLASSES.items():
        fg8 = cv2.morphologyEx((fn(H, S, V) & ~jaw).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        fg8 = cv2.morphologyEx(fg8, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n0, lab0, st0, _ = cv2.connectedComponentsWithStats(fg8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(fg8, np.ones((15, 15), np.uint8)))
        for i in range(1, n):
            m = (lab == i) & (fg8 > 0)
            if int(m.sum()) < 1500:
                continue
            pieces = sorted({int(v) for v in np.unique(lab0[m]) if v > 0}, key=lambda j: -st0[j, 4])
            if len(pieces) < 2:
                continue
            areas = [int(st0[j, 4]) for j in pieces]
            print(f"{os.path.relpath(f, FIX):40s} {cname:7s} pieces {areas} ratio2 {areas[1] / areas[0]:.3f}")
