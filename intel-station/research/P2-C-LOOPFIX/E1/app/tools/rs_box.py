#!/usr/bin/env python3
"""Opus I-59: height stats of everything above the mat inside a depth-pixel box (e.g. the left jaw beside the brick).
usage: rs_box.py DIR x0,y0,x1,y1 [--min-h 2.5]"""
import json
import os
import sys

import numpy as np

import rs_height as rh

d = sys.argv[1]
x0, y0, x1, y1 = map(int, sys.argv[2].split(","))
mh = float(sys.argv[4]) if len(sys.argv) > 4 else 2.5
D = np.load(os.path.join(d, "close_depth.npy"))
h, w = D.shape
fx, fy, cx, cy, _ = rh.intrinsics(w, h, 65.0, "")
P = rh.cloud(D, fx, fy, cx, cy)
valid = np.isfinite(D)
pts = P[valid]
pts = pts[np.random.default_rng(1).choice(len(pts), min(40000, len(pts)), replace=False)]
n, dd, _, _ = rh.fit_plane(pts)
H = np.where(valid, np.einsum("ijk,k->ij", np.nan_to_num(P), n) + dd, np.nan)
B = H[y0:y1, x0:x1]
m = np.isfinite(B) & (B > mh)
ys, xs = np.nonzero(m)
out = {"n": int(m.sum())}
if m.sum():
    hv = B[m]
    out.update({"h_p1": round(float(np.percentile(hv, 1)), 1), "h_p5": round(float(np.percentile(hv, 5)), 1),
                "h_min": round(float(hv.min()), 1), "bottom_row": int(ys.max() + y0)})
    rows = []
    for yy in range(max(0, ys.max() - 12), ys.max() + 1, 2):
        r = B[yy][m[yy]]
        if r.size:
            rows.append([int(yy + y0), round(float(np.median(r)), 1), int(r.size)])
    out["rows_y_hmed_n"] = rows
print(json.dumps(out))
