#!/usr/bin/env python3
"""P3-TWIN-CAMS E1 step 9 diagnostic 6 (read-only): joint zero convention x sign pattern, on our own data.

Two zero conventions for the five body joints, both with the calibration scale k = width/200 * 2pi/4095:
  mid    : rad = sign * k * norm                       (zero at the calibrated range midpoint; the kin convention)
  homing : rad = sign * (k * norm + (mid - 2048) * 2pi/4095)   (zero at tick 2048, the Feetech centre)
For each of the 64 combinations, OpenCV PnP-RANSAC over f = 400..900 (step 100) at a 12 px threshold on the
fixed-jaw tips. The best combinations by inlier count are reported; a real convention should give a clear jump.
Run: envs/anomalib-env/bin/python -I tools/diag_zero.py <top|close>
"""
import itertools
import sys

import cv2
import numpy as np

E1 = "/home/intel/intel-junction-2026/work/P3-TWIN-CAMS/E1"
sys.path.insert(0, f"{E1}/tools")
import s9_fit as S  # noqa: E402

BODY = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def pnp_best(obj, img, thr=12.0):
    best = (0, None, None, None)
    for f in np.arange(400.0, 901.0, 100.0):
        K = np.array([[f, 0.0, 320.0], [0.0, f, 240.0], [0.0, 0.0, 1.0]])
        ok, rv, tv, inl = cv2.solvePnPRansac(obj, img, K, None, iterationsCount=1500, reprojectionError=thr,
                                             confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < 8:
            continue
        idx = inl.ravel()
        rv, tv = cv2.solvePnPRefineLM(obj[idx], img[idx], K, None, rv, tv)
        proj, _ = cv2.projectPoints(obj[idx], rv, tv, K, None)
        err = np.linalg.norm(proj.reshape(-1, 2) - img[idx], axis=1)
        if len(idx) > best[0]:
            best = (len(idx), int(f), round(float(np.median(err)), 2),
                    np.round((-cv2.Rodrigues(rv)[0].T @ tv).ravel(), 3).tolist())
    return best


def run(cam):
    D = S.load(cam)
    two = D["two"]
    img = np.vstack([D["A"], D["B"][two]]).astype(np.float64)
    k = S.CAL.k[:5]
    mid = np.array([(S.CAL.joints[j].range_min + S.CAL.joints[j].range_max) / 2.0 for j in BODY])
    homing = (mid - 2048.0) * 2.0 * np.pi / 4095.0
    print(cam, "homing offsets deg", np.round(np.degrees(homing), 2).tolist(), flush=True)
    res = []
    for zname, zero in (("mid", np.zeros(5)), ("homing", homing)):
        for s in itertools.product((1.0, -1.0), repeat=5):
            sign = np.array(s)
            Q = sign * (k * D["pose"][:, :5] + zero)
            Tbg = S.fk_batch(Q)
            Pf, _ = S.features(Tbg, np.zeros(len(Q)), 0.0)
            obj = np.vstack([Pf, Pf[two]]).astype(np.float64)
            n, f, med, C = pnp_best(obj, img)
            res.append((n, zname, s, f, med, C))
    res.sort(key=lambda r: -r[0])
    print(cam, "observations", len(img), "| best 8 by inliers (inliers, zero, signs pan/lift/elbow/wflex/roll, f, median px, C):", flush=True)
    for r in res[:8]:
        print("  ", r, flush=True)
    print(cam, "median inliers over all 64", int(np.median([r[0] for r in res])), flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
