#!/usr/bin/env python3
"""V1 footprint check with the minimum-area rectangle (cv2.minAreaRect) instead of PCA axes.
PCA axes are arbitrary for a square, so the earlier PCA extents (footprint_scale.py) over-state a square by up to
about 41 percent and are not used as evidence. Same blobs as footprint_scale.py, same plane (depth_check.py).
Also tests the flying-pixel halo by eroding the half-height mask by one pixel.
Writes outputs/footprint_rect.json."""
import json
import math
import sys

import cv2
import numpy as np

TOOLS = "/home/intel/intel-junction-2026/work/P3-TWIN-SCENE/V1/tools"
sys.path.insert(0, TOOLS)
import depth_check as dc  # noqa: E402

OUT = "/home/intel/intel-junction-2026/work/P3-TWIN-SCENE/V1/outputs"
DEP = "/home/intel/intel-junction-2026/logs/depth/g1_162903"
CAMS = {
    "top": (DEP + "/top_depth.npy", [("top_blob_A_builder_45p1_132p8", (472, 306, 37, 37), "31.42 x 30.41"),
                                     ("top_blob_B_builder_107p5_194p6", (539, 242, 33, 31), "31.59 x 28.35")]),
    "close": (DEP + "/c424x240@15/close_depth.npy", [("close_blob_builder_85p7_186p8", (301, 79, 58, 38), "30.12 x 25.92")]),
}
THRS = [9.6, 12.0, 15.0]


def rect_mm(xy):
    (_, _), (w, h), _ = cv2.minAreaRect(np.float32(xy))
    return round(float(max(w, h)), 2), round(float(min(w, h)), 2)


def run(cam, path, targets):
    dep = np.load(path).astype(np.float64)
    P, valid, f, cx, cy = dc.points(dep)
    X = P[valid]
    (nv, d), _ = dc.lmeds(X, np.random.default_rng(7))
    nv, d = dc.refine(X, nv, d, 2.5)
    ex, ey = dc.basis(nv)
    H = np.where(valid, P @ nv + d, -99.0)
    m = ((H > 9.6) & (H < 30.0)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    blobs = []
    for name, (bx, by, bw, bh), builder in targets:
        best, bo = 0, 0
        for i in range(1, n):
            x, y, w_, h_, area = [int(v) for v in st[i]]
            ox = max(0, min(x + w_, bx + bw) - max(x, bx))
            oy = max(0, min(y + h_, by + bh) - max(y, by))
            if ox * oy > bo:
                best, bo = i, ox * oy
        sel = lab == best
        rec = {"name": name, "builder_long_x_short_mm": builder, "bbox_xywh_mine": [int(v) for v in st[best][:4]]}
        for t in THRS:
            s2 = sel & (H > t)
            xy = np.stack([P[s2] @ ex, P[s2] @ ey], 1)
            L, S = rect_mm(xy)
            rec["minrect_H>%g" % t] = {"n": int(s2.sum()), "long_mm": L, "short_mm": S, "median_H_mm": round(float(np.median(H[s2])), 2)}
        er = cv2.erode(sel.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        s3 = er & (H > 9.6)
        xy3 = np.stack([P[s3] @ ex, P[s3] @ ey], 1)
        L3, S3 = rect_mm(xy3)
        rec["minrect_H>9.6_eroded1px"] = {"n": int(s3.sum()), "long_mm": L3, "short_mm": S3}
        blobs.append(rec)
    return {"camera": cam, "plane_cam_to_mat_mm": round(d, 2), "tilt_deg": round(math.degrees(math.acos(min(1.0, abs(float(nv[2]))))), 2),
            "f_px_nominal": round(f, 2), "blobs": blobs}


def main():
    out = {"stage": "twin-scene-v1", "worker": "V1", "nominal_2x2_mm": 31.7,
           "method": "cv2.minAreaRect on in-plane (P.ex, P.ey) coordinates of the blob pixels, per height threshold",
           "cams": [run(c, p, t) for c, (p, t) in CAMS.items()]}
    with open(OUT + "/footprint_rect.json", "w") as fh:
        json.dump(out, fh, indent=1)
    for c in out["cams"]:
        print(c["camera"], "plane", c["plane_cam_to_mat_mm"], "tilt", c["tilt_deg"], "f_nominal", c["f_px_nominal"])
        for b in c["blobs"]:
            print(" ", json.dumps(b))


if __name__ == "__main__":
    main()
