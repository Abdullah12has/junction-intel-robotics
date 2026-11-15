#!/usr/bin/env python3
"""Opus I-59: brick footprint and jaw-plate footprints in the mat frame from a close-camera depth grab.
Same mat frame as rs_jaws.py (origin = brick top centre on the mat, x = image-right on mat, y = away from camera)."""
import argparse
import json
import os

import cv2
import numpy as np

import rs_height as rh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--brick-box", default="190,95,265,165")
    ap.add_argument("--jaw-box", default="140,0,270,135")
    ap.add_argument("--split-x", type=int, default=215)
    ap.add_argument("--band", type=float, default=8.0, help="jaw slice height above its tip (mm)")
    a = ap.parse_args()
    D = np.load(os.path.join(a.dir, "close_depth.npy"))
    h, w = D.shape
    fx, fy, cx, cy, _ = rh.intrinsics(w, h, 65.0, "")
    P = rh.cloud(D, fx, fy, cx, cy)
    valid = np.isfinite(D)
    pts = P[valid]
    pts = pts[np.random.default_rng(1).choice(len(pts), min(40000, len(pts)), replace=False)]
    n, d, _, _ = rh.fit_plane(pts)
    H = np.where(valid, np.einsum("ijk,k->ij", np.nan_to_num(P), n) + d, np.nan)
    ex = np.array([1.0, 0, 0]) - n * n[0]
    ex /= np.linalg.norm(ex)
    ey = np.cross(n, ex)
    bx0, by0, bx1, by1 = map(int, a.brick_box.split(","))
    Hb, Pb = H[by0:by1, bx0:bx1], P[by0:by1, bx0:bx1]
    top = np.isfinite(Hb) & (Hb > 15) & (Hb < 26)
    btop = Pb[top].mean(0)
    origin = btop - n * (btop @ n + d)

    def xy(p):
        q = p - origin
        return np.stack([q @ ex, q @ ey], -1)

    res = {}
    allb = np.isfinite(Hb) & (Hb > 3) & (Hb < 26)
    for nm, m in (("brick_top", top), ("brick_all", allb)):
        q = xy(Pb[m]).astype(np.float32)
        (cxm, cym), (sw, sh), ang = cv2.minAreaRect(q * 10)
        res[nm] = {"n": int(m.sum()), "centre": [round(cxm / 10, 1), round(cym / 10, 1)],
                   "size": [round(sw / 10, 1), round(sh / 10, 1)], "angle_deg": round(ang, 1),
                   "x_range": [round(float(q[:, 0].min()), 1), round(float(q[:, 0].max()), 1)],
                   "y_range": [round(float(q[:, 1].min()), 1), round(float(q[:, 1].max()), 1)]}
    jx0, jy0, jx1, jy1 = map(int, a.jaw_box.split(","))
    for nm, (x0, x1) in (("left_jaw", (jx0, a.split_x)), ("right_jaw", (a.split_x, jx1))):
        Hj, Pj = H[jy0:jy1, x0:x1], P[jy0:jy1, x0:x1]
        m = np.isfinite(Hj) & (Hj > 24)
        if m.sum() < 20:
            continue
        hmin = float(np.percentile(Hj[m], 1))
        sl = m & (Hj < hmin + a.band)
        q = xy(Pj[sl])
        c = q.mean(0)
        u, s, vt = np.linalg.svd(q - c, full_matrices=False)
        res[nm] = {"hmin": round(hmin, 1), "slice_n": int(sl.sum()), "slice_centre": [round(float(c[0]), 1), round(float(c[1]), 1)],
                   "plate_axis": [round(float(vt[0][0]), 2), round(float(vt[0][1]), 2)],
                   "plate_len_mm": round(float(np.ptp(q @ vt[0])), 1), "plate_thick_mm": round(float(np.ptp(q @ vt[1])), 1)}
    print(json.dumps(res))


if __name__ == "__main__":
    main()
