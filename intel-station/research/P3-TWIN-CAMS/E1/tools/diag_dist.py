#!/usr/bin/env python3
"""P3-TWIN-CAMS E1 step 9 diagnostic 7 (read-only): is the lack of a PnP solution caused by lens distortion?

For radial k1 in a grid and focal lengths 300..900, the observed tips are undistorted with a plumb-bob (k1 only)
model, then OpenCV PnP-RANSAC (pinhole) runs on the fixed-jaw tip correspondences (kin convention, mid zero, all
signs +1; and homing-2048 zero). Reports the best inlier count per camera.
Run: envs/anomalib-env/bin/python -I tools/diag_dist.py <top|close>
"""
import sys

import cv2
import numpy as np

E1 = "/home/intel/intel-junction-2026/work/P3-TWIN-CAMS/E1"
sys.path.insert(0, f"{E1}/tools")
import s9_fit as S  # noqa: E402

BODY = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def undistort_px(uv, f, k1):
    K = np.array([[f, 0.0, 320.0], [0.0, f, 240.0], [0.0, 0.0, 1.0]])
    und = cv2.undistortPoints(uv.reshape(-1, 1, 2).astype(np.float64), K, np.array([k1, 0.0, 0.0, 0.0]))
    return und.reshape(-1, 2) * f + np.array([320.0, 240.0])


def run(cam):
    D = S.load(cam)
    two = D["two"]
    k = S.CAL.k[:5]
    mid = np.array([(S.CAL.joints[j].range_min + S.CAL.joints[j].range_max) / 2.0 for j in BODY])
    homing = (mid - 2048.0) * 2.0 * np.pi / 4095.0
    best_all = []
    for zname, zero in (("mid", np.zeros(5)), ("homing", homing)):
        Q = k * D["pose"][:, :5] + zero
        Tbg = S.fk_batch(Q)
        Pf, _ = S.features(Tbg, np.zeros(len(Q)), 0.0)
        obj0 = np.vstack([Pf, Pf[two]]).astype(np.float64)
        img0 = np.vstack([D["A"], D["B"][two]]).astype(np.float64)
        for k1 in (-0.6, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2):
            for f in (300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0):
                img = undistort_px(img0, f, k1)
                K = np.array([[f, 0.0, 320.0], [0.0, f, 240.0], [0.0, 0.0, 1.0]])
                ok, rv, tv, inl = cv2.solvePnPRansac(obj0, img, K, None, iterationsCount=1500,
                                                     reprojectionError=6.0, confidence=0.999,
                                                     flags=cv2.SOLVEPNP_EPNP)
                if not ok or inl is None or len(inl) < 8:
                    continue
                idx = inl.ravel()
                rv, tv = cv2.solvePnPRefineLM(obj0[idx], img[idx], K, None, rv, tv)
                proj, _ = cv2.projectPoints(obj0[idx], rv, tv, K, None)
                err = np.linalg.norm(proj.reshape(-1, 2) - img[idx], axis=1)
                C = (-cv2.Rodrigues(rv)[0].T @ tv).ravel()
                best_all.append((len(idx), zname, k1, f, round(float(np.median(err)), 2), np.round(C, 3).tolist()))
    best_all.sort(key=lambda r: -r[0])
    print(cam, "observations", len(img0), "| best 8 (inliers, zero, k1, f, median px, C m):", flush=True)
    for r in best_all[:8]:
        print("  ", r, flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
