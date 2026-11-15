"""Calibration diagnostics for the E2 segmentation (prints only; writes nothing).

usage: python3 -I calib.py
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inspector as I  # noqa: E402
from common import CLOSE, TOP, WORK  # noqa: E402

samples = json.load(open(os.path.join(WORK, "samples", "samples.json")))
areas_22, areas_all = [], []
for sid in sorted(samples):
    rec = samples[sid]
    if rec["ds8"] != "6d6bdce3":
        continue
    top = cv2.imread(rec["views"][TOP]["png"])
    close = cv2.imread(rec["views"][CLOSE]["png"])
    h, s, v = I._split(top)
    roi = I._roi(h.shape, I.TOP_ROI)
    mat = roi & (v < 70)
    print(sid, "mat_V_pct50/90", int(np.percentile(v[mat], 50)), int(np.percentile(v[mat], 90)),
          "mat_frac", round(float(mat.sum()) / roi.sum(), 3))
    seg = I.segment_top(top)
    for b in seg["bricks"]:
        areas_all.append(b["area"])
        if b["aspect"] <= 1.25:
            areas_22.append(b["area"])
    # close view: count raw components before rejection
    hc, sc, vc = I._split(close)
    fg = cv2.morphologyEx((I._roi(hc.shape, I.CLOSE_ROI) & (vc > 70)).astype(np.uint8), cv2.MORPH_OPEN,
                          I.K3).astype(bool)
    raw = [(int(st[0]), int(st[1]), int(st[2]), int(st[3]), a) for _, a, st in I._components(fg, 300)]
    print("   close raw comps (x,y,w,h,area):", sorted(raw, key=lambda t: -t[4])[:12])
    print("   top shape counts:", {k: sum(1 for b in seg["bricks"] if b["shape"] == k)
                                  for k in ("2x2", "not_2x2", "unknown")},
          "aspects(2x2-like) sample:", [b["aspect"] for b in seg["bricks"][:8]])
a22 = np.array(areas_22)
print("TOP area of aspect<=1.25 comps: n", len(a22), "pct10/50/90", np.percentile(a22, [10, 50, 90]).round(0))
print("TOP area all comps: pct10/50/90", np.percentile(np.array(areas_all), [10, 50, 90]).round(0))
