"""V1 live-frame component experiment (analysis only). usage: python3 -I -B cores_v1.py APP_DIR
Compares fg opening 3x3 vs 5x5, colour-class cores (white versus chroma) on the live top and close stills."""
import sys

import cv2
import numpy as np

sys.path.insert(0, sys.argv[1])
import inspector as I  # noqa: E402

LIVE = "/home/intel/intel-junction-2026/logs/detlive/"
top = cv2.imread(LIVE + "top_153426.jpg")
close = cv2.imread(LIVE + "close_153426.jpg")


def comps(mask, min_a, planes, tag, W=640, H=480):
    for m, a, st in I._components(mask, min_a):
        x, y, w, hh = (int(t) for t in st[:4])
        touch = x <= 1 or y <= 1 or x + w >= W - 1 or y + hh >= H - 1
        med = [round(float(np.median(q[m])), 1) for q in planes]
        print("  ", tag, "area", a, "bbox", [x, y, w, hh], "touch", touch, "medHSV", med)


h, s, v = I._split(top)
roi = I._roi(h.shape, (I.TOP_X0, 0, 640, 364))
robot = np.zeros(h.shape, np.uint8)
cv2.fillPoly(robot, [I.ROBOT_POLY], 1)
fg = roi & (v > 70) & (robot == 0)
for k in (3, 5):
    f = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8)).astype(bool)
    print("TOP fg open", k)
    comps(f, 200, (h, s, v), "T")
print("TOP white core s<25 v>150")
comps(roi & (s < 25) & (v > 150) & (robot == 0), 200, (h, s, v), "Wt")
print("TOP chroma core s>=45 v>70")
comps(roi & (s >= 45) & (v > 70) & (robot == 0), 200, (h, s, v), "Ch")

hc, sc, vc = I._split(close)
rc = I._roi(hc.shape, I.CLOSE_ROI)
fgc = rc & (((sc > 85) & (vc > 90)) | ((vc > 170) & (sc < 70)))
for k in (3, 5):
    f = cv2.morphologyEx(fgc.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8)).astype(bool)
    print("CLOSE fg open", k)
    comps(f, 300, (hc, sc, vc), "C")
print("CLOSE white core s<25 v>150")
comps(rc & (sc < 25) & (vc > 150), 300, (hc, sc, vc), "Cw")
print("CLOSE chroma core s>=45 v>90")
comps(rc & (sc >= 45) & (vc > 90), 300, (hc, sc, vc), "Cc")
