#!/usr/bin/env python3
"""P3-TWIN-CAMS E1 step 9 diagnostic (read-only on recordings): does the kinematic jaw model admit a camera?

For each focal guess f, OpenCV PnP-RANSAC on all (3D fixed-jaw tip, 2D red-blob tip) correspondences of the window
set (every observed blob is a candidate; wrong pairs are outliers). If the geometry and joint convention are right,
a large inlier set with a few-px reprojection error must exist for some f.
Run: envs/anomalib-env/bin/python -I tools/diag_pnp.py <top|close>
"""
import sys

import cv2
import numpy as np

E1 = "/home/intel/intel-junction-2026/work/P3-TWIN-CAMS/E1"
sys.path.insert(0, f"{E1}/tools")
import s9_fit as S  # noqa: E402


def run(cam):
    D = S.load(cam)
    Pf, _ = S.features(D["Tbg0"], np.zeros(len(D["g"])), 0.0)
    two = D["two"]
    obj = np.vstack([Pf, Pf[two]]).astype(np.float64)
    img = np.vstack([D["A"], D["B"][two]]).astype(np.float64)
    print(cam, "frames", len(Pf), "correspondences", len(obj), flush=True)
    rows = []
    for f in np.arange(350.0, 1001.0, 50.0):
        K = np.array([[f, 0.0, 320.0], [0.0, f, 240.0], [0.0, 0.0, 1.0]])
        ok, rv, tv, inl = cv2.solvePnPRansac(obj, img, K, None, iterationsCount=4000, reprojectionError=6.0,
                                             confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < 8:
            print("f", int(f), "no consistent pose", flush=True)
            continue
        idx = inl.ravel()
        rv, tv = cv2.solvePnPRefineLM(obj[idx], img[idx], K, None, rv, tv)
        proj, _ = cv2.projectPoints(obj[idx], rv, tv, K, None)
        err = np.linalg.norm(proj.reshape(-1, 2) - img[idx], axis=1)
        Rcv = cv2.Rodrigues(rv)[0]
        C = (-Rcv.T @ tv).ravel()
        axis = Rcv.T[:, 2]
        rows.append((len(idx), f, float(np.median(err)), C, axis))
        print("f", int(f), "inliers", len(idx), "of", len(obj), "median px", round(float(np.median(err)), 2),
              "p90 px", round(float(np.percentile(err, 90)), 2), "C m", np.round(C, 3).tolist(),
              "optical axis", np.round(axis, 3).tolist(), flush=True)
    if rows:
        best = max(rows, key=lambda r: r[0])
        print("best by inliers: f", int(best[1]), "inliers", best[0], "median px", round(best[2], 2),
              "C", np.round(best[3], 3).tolist(), flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
