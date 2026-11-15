#!/usr/bin/env python3
"""P3-TWIN-CAMS E1 step 9 diagnostic 3 (read-only): joint-sign hypothesis test on our own still-window data.

For each of the 32 sign patterns over (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll), the
fixed-jaw tip is computed with rad = sign * CAL.k * norm (mid-range zero) and OpenCV PnP-RANSAC is run over focal
guesses 400..900 at a 12 px threshold. A convention that matches the image should give a clear jump in inliers.
Run: envs/anomalib-env/bin/python -I tools/diag_sign.py <top|close>
"""
import itertools
import sys

import cv2
import numpy as np

E1 = "/home/intel/intel-junction-2026/work/P3-TWIN-CAMS/E1"
sys.path.insert(0, f"{E1}/tools")
import s9_fit as S  # noqa: E402


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
            best = (len(idx), int(f), float(np.median(err)), np.round((-cv2.Rodrigues(rv)[0].T @ tv).ravel(), 3).tolist())
    return best


def run(cam):
    D = S.load(cam)
    two = D["two"]
    img = np.vstack([D["A"], D["B"][two]]).astype(np.float64)
    res = []
    for s in itertools.product((1.0, -1.0), repeat=5):
        sign = np.array(s)
        Q = sign * S.CAL.k[:5] * D["pose"][:, :5]
        Tbg = S.fk_batch(Q)
        Pf, _ = S.features(Tbg, np.zeros(len(Q)), 0.0)
        obj = np.vstack([Pf, Pf[two]]).astype(np.float64)
        n, f, med, C = pnp_best(obj, img)
        res.append((n, s, f, med, C))
        print(cam, "signs(pan,lift,elbow,wflex,roll)", s, "inliers", n, "of", len(obj), "f", f,
              "median px", round(med, 2) if f else None, "C", C, flush=True)
    res.sort(key=lambda r: -r[0])
    print(cam, "TOP 5 by inliers:", [(r[0], r[1], r[2], round(r[3], 2) if r[2] else None) for r in res[:5]], flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
