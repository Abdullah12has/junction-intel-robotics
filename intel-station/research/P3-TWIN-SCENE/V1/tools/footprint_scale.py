#!/usr/bin/env python3
"""V1 brick footprint and top-height check at several height thresholds (top and close depth maps).
Uses the plane fit and helpers from depth_check.py in the same folder (imported, not copied).
Footprint = PCA extents of the blob pixels in the mat plane; top = median H of pixels above each threshold.
Writes outputs/footprint_scale.json."""
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
    "top": (DEP + "/top_depth.npy", [("top_blob_A_builder_45p1_132p8", (472, 306, 37, 37)),
                                     ("top_blob_B_builder_107p5_194p6", (539, 242, 33, 31))]),
    "close": (DEP + "/c424x240@15/close_depth.npy", [("close_blob_builder_85p7_186p8", (301, 79, 58, 38))]),
}
THRS = [9.6, 12.0, 15.0, 17.0]
NOMINAL_FOOT_MM = 31.7


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
    for name, (bx, by, bw, bh) in targets:
        best, bo = 0, 0
        for i in range(1, n):
            x, y, w_, h_, area = [int(v) for v in st[i]]
            ox = max(0, min(x + w_, bx + bw) - max(x, bx))
            oy = max(0, min(y + h_, by + bh) - max(y, by))
            if ox * oy > bo:
                best, bo = i, ox * oy
        if best == 0:
            blobs.append({"name": name, "error": "no component overlaps the builder bbox"})
            continue
        sel = lab == best
        rec = {"name": name, "bbox_xywh_mine": [int(v) for v in st[best][:4]], "area_px_halfheight": int(st[best][4])}
        for t in THRS:
            s2 = sel & (H > t)
            if int(s2.sum()) < 30:
                rec["H>%g" % t] = {"n": int(s2.sum())}
                continue
            A = np.stack([P[s2] @ ex, P[s2] @ ey], 1)
            _, vecs = np.linalg.eigh(np.cov(A.T))
            a = A @ vecs
            ext = np.percentile(a, 99.5, axis=0) - np.percentile(a, 0.5, axis=0)
            full = a.max(axis=0) - a.min(axis=0)
            rec["H>%g" % t] = {"n": int(s2.sum()),
                               "long_p99_mm": round(float(ext.max()), 2), "short_p99_mm": round(float(ext.min()), 2),
                               "long_full_mm": round(float(full.max()), 2), "short_full_mm": round(float(full.min()), 2),
                               "median_H_mm": round(float(np.median(H[s2])), 2),
                               "long_p99_over_nominal": round(float(ext.max()) / NOMINAL_FOOT_MM, 3)}
        blobs.append(rec)
    return {"camera": cam, "file": path, "plane_cam_to_mat_mm": round(d, 2),
            "tilt_deg": round(math.degrees(math.acos(min(1.0, abs(float(nv[2]))))), 2),
            "intrinsics_f_cx_cy": [round(f, 2), cx, cy], "blobs": blobs}


def main():
    out = {"stage": "twin-scene-v1", "worker": "V1", "nominal_footprint_mm": NOMINAL_FOOT_MM,
           "note": "PCA extents in the mat plane; p99 = 0.5-99.5 percentile range; full = min-max.",
           "cams": [run(c, p, t) for c, (p, t) in CAMS.items()]}
    with open(OUT + "/footprint_scale.json", "w") as fh:
        json.dump(out, fh, indent=1)
    for c in out["cams"]:
        print(c["camera"], "plane", c["plane_cam_to_mat_mm"], c["tilt_deg"])
        for b in c["blobs"]:
            print(" ", json.dumps(b))


if __name__ == "__main__":
    main()
