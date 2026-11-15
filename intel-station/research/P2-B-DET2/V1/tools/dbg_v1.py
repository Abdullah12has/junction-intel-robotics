"""V1 neck-split debugger (analysis only). usage: python3 -I -B dbg_v1.py APP_DIR"""
import sys

import cv2
import numpy as np

sys.path.insert(0, sys.argv[1])
import inspector as I  # noqa: E402

top = cv2.imread("/home/intel/intel-junction-2026/logs/detlive/top_153426.jpg")
h, s, v = I._split(top)
y1 = I.top_rail_row(v)
print("rail row", y1)
roi = I._roi(h.shape, (I.TOP_X0, 0, h.shape[1], y1))
robot = np.zeros(h.shape, np.uint8)
cv2.fillPoly(robot, [I.ROBOT_POLY], 1)
fg = roi & (v > 70) & (robot == 0)
fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, I.K3).astype(bool)
for m, a, st in I._components(fg, I.MIN_AREA_TOP):
    if a < 4000:
        continue
    print("blob", a, st[:4].tolist())
    dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    print("dmax", round(float(dist.max()), 2))
    for name, core in (("white", m & (s < 25) & (v > 150)), ("chroma", m & (s >= 45))):
        for cm, ca, cst in I._components(core, 0.4 * I.A2X2_TOP):
            ys, xs = np.nonzero(cm)
            dv = dist[ys, xs]
            picked = []
            for i in np.argsort(-dv):
                if dv[i] < 0.6 * dv.max():
                    break
                x0, y0 = int(xs[i]), int(ys[i])
                if all((x0 - px) ** 2 + (y0 - py) ** 2 >= 144 for px, py in picked):
                    picked.append((x0, y0))
                if len(picked) >= (3 if ca >= 1.5 * I.A2X2_TOP else 1):
                    break
            print(" ", name, "core", ca, cst[:4].tolist(), "picked", picked, "dvmax", round(float(dv.max()), 2))
    pcs = I.neck_split(h, s, v, m, I.A2X2_TOP)
    print("neck_split pieces", None if pcs is None else [(int(pa), pst[:4].tolist()) for _, pa, pst in pcs])
