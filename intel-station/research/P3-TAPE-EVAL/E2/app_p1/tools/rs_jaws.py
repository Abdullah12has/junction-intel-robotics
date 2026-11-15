#!/usr/bin/env python3
"""Opus I-59: measure the two jaw tips and the brick in the close D415 depth (mat frame, mm).

Jaws = the two largest components of (height > --jaw-min-h) that touch the top image edge (the arm enters from
the top); L = smaller image x, R = larger. Tip = mean of the component's bottom 4 rows; hmin = 1st percentile height.
Each jaw also reports a reference point at a fixed height band (--ref-h +-2 mm) so rigid motion can be tracked
after the tips drop below the threshold.
Mat frame: z = mat normal (up), origin = brick top-face centre on the mat, x = image-right on the mat,
y = z cross x (away from the close camera, toward the robot base).
"""
import argparse
import json
import os

import cv2
import numpy as np

import rs_height as rh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--cam", default="close")
    ap.add_argument("--hfov", type=float, default=65.0)
    ap.add_argument("--brick-box", default="190,95,265,165")
    ap.add_argument("--jaw-min-h", type=float, default=24.0)
    ap.add_argument("--ref-h", type=float, default=40.0)
    ap.add_argument("--jaw-box", default="")  # kept for CLI compatibility (unused)
    ap.add_argument("--split-x", default="")  # kept for CLI compatibility (unused)
    a = ap.parse_args()
    D = np.load(os.path.join(a.dir, f"{a.cam}_depth.npy"))
    h, w = D.shape
    fx, fy, cx, cy, src = rh.intrinsics(w, h, a.hfov, "")
    P = rh.cloud(D, fx, fy, cx, cy)
    valid = np.isfinite(D)
    pts = P[valid]
    pts = pts[np.random.default_rng(1).choice(len(pts), min(40000, len(pts)), replace=False)]
    n, d, inl, rms = rh.fit_plane(pts)
    H = np.where(valid, np.einsum("ijk,k->ij", np.nan_to_num(P), n) + d, np.nan)
    bx0, by0, bx1, by1 = map(int, a.brick_box.split(","))
    Hb, Pb = H[by0:by1, bx0:bx1], P[by0:by1, bx0:bx1]
    top = np.isfinite(Hb) & (Hb > 15) & (Hb < 26)
    btop = Pb[top].mean(0)
    ez = n
    ex = np.array([1.0, 0, 0]) - n * n[0]
    ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    origin = btop - ez * (btop @ ez + d)

    def mat(p):
        q = p - origin
        return [round(float(q @ ex), 1), round(float(q @ ey), 1), round(float(p @ ez + d), 1)]

    res = {"plane_rms_mm": round(rms, 2), "brick": {"top_h_p75": round(float(np.percentile(Hb[top], 75)), 1),
                                                   "top_px": int(top.sum())}}
    m0 = np.nan_to_num(H, nan=-99) > a.jaw_min_h
    m0[int(h * 0.85):, :] = False
    md = cv2.dilate(m0.astype(np.uint8), np.ones((5, 5), np.uint8))  # bridge depth holes on thin jaw edges
    k, lab, st, _ = cv2.connectedComponentsWithStats(md, connectivity=8)
    area0 = {i: int((m0 & (lab == i)).sum()) for i in range(1, k)}
    cands = [i for i in range(1, k) if st[i, 1] <= 15 and area0[i] >= 30]
    cands = sorted(cands, key=lambda i: -area0[i])[:2]
    cands = sorted(cands, key=lambda i: st[i, 0] + st[i, 2] / 2)
    names = ["left_jaw", "right_jaw"] if len(cands) == 2 else ["jaw"]
    for nm, i in zip(names, cands):
        comp = m0 & (lab == i)
        ys, xs = np.nonzero(comp)
        ybot = int(np.percentile(ys, 99))
        tip = comp & (np.arange(h)[:, None] >= ybot - 4)
        hv = H[comp]
        band = comp & (np.abs(H - a.ref_h) < 2)
        res[nm] = {"px_bbox": [int(st[i, 0]), int(st[i, 1]), int(st[i, 2]), int(st[i, 3])], "area": int(st[i, 4]),
                   "tip_h_min": round(float(np.percentile(hv, 1)), 1), "tip_px_y": ybot,
                   "tip_mat": mat(P[tip].mean(0)),
                   "ref_mat": mat(P[band].mean(0)) if band.sum() >= 5 else None}
    if len(cands) < 2:
        res["warning"] = f"found {len(cands)} jaw components"
    print(json.dumps(res))
    json.dump(res, open(os.path.join(a.dir, f"{a.cam}_jaws.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
