#!/usr/bin/env python3
"""P3-TWIN-CAMS E1 step 9 diagnostic 2 (read-only): PnP-RANSAC inlier counts at looser reprojection thresholds,
for the fixed-jaw tip (A) and the moving-jaw tip (B) candidates, with a small grid of tip offsets along gripper z.
Run: envs/anomalib-env/bin/python -I tools/diag_pnp2.py <top|close>
"""
import sys

import cv2
import numpy as np

E1 = "/home/intel/intel-junction-2026/work/P3-TWIN-CAMS/E1"
sys.path.insert(0, f"{E1}/tools")
import s9_fit as S  # noqa: E402


def best_pnp(obj, img, thr):
    best = None
    for f in np.arange(400.0, 901.0, 50.0):
        K = np.array([[f, 0.0, 320.0], [0.0, f, 240.0], [0.0, 0.0, 1.0]])
        ok, rv, tv, inl = cv2.solvePnPRansac(obj, img, K, None, iterationsCount=3000, reprojectionError=thr,
                                             confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < 8:
            continue
        idx = inl.ravel()
        rv, tv = cv2.solvePnPRefineLM(obj[idx], img[idx], K, None, rv, tv)
        proj, _ = cv2.projectPoints(obj[idx], rv, tv, K, None)
        err = np.linalg.norm(proj.reshape(-1, 2) - img[idx], axis=1)
        C = (-cv2.Rodrigues(rv)[0].T @ tv).ravel()
        if best is None or len(idx) > best[0]:
            best = (len(idx), int(f), float(np.median(err)), np.round(C, 3).tolist())
    return best


def run(cam):
    D = S.load(cam)
    two = D["two"]
    n_obs = len(D["A"]) + int(two.sum())
    print(cam, "frames", len(D["A"]), "two-jaw", int(two.sum()), "observations", n_obs, flush=True)
    for dz in (0.0, -0.006, 0.006, -0.012):
        Pf, _ = S.features(D["Tbg0"], np.zeros(len(D["g"])), dz)
        obj = np.vstack([Pf, Pf[two]]).astype(np.float64)
        img = np.vstack([D["A"], D["B"][two]]).astype(np.float64)
        for thr in (6.0, 12.0, 25.0):
            r = best_pnp(obj, img, thr)
            print("delta_z_mm", round(dz * 1000, 1), "thr_px", thr, "best (inliers, f, median px, C m):", r, flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
