#!/usr/bin/env python3
"""P3-TWIN-CAMS E1 step 9 diagnostic 5 (read-only): per-episode PnP-RANSAC (fixed-jaw tip correspondences, kin
convention). If a single camera explains one episode well but the episodes disagree, the camera (or the timing)
changed between episodes; if no episode is consistent, the feature or convention is wrong.
Run: envs/anomalib-env/bin/python -I tools/diag_episode.py <top|close>
"""
import sys

import cv2
import numpy as np

E1 = "/home/intel/intel-junction-2026/work/P3-TWIN-CAMS/E1"
sys.path.insert(0, f"{E1}/tools")
import s9_fit as S  # noqa: E402


def pnp_best(obj, img, thr):
    best = None
    for f in np.arange(400.0, 901.0, 50.0):
        K = np.array([[f, 0.0, 320.0], [0.0, f, 240.0], [0.0, 0.0, 1.0]])
        ok, rv, tv, inl = cv2.solvePnPRansac(obj, img, K, None, iterationsCount=1500, reprojectionError=thr,
                                             confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < 6:
            continue
        idx = inl.ravel()
        rv, tv = cv2.solvePnPRefineLM(obj[idx], img[idx], K, None, rv, tv)
        proj, _ = cv2.projectPoints(obj[idx], rv, tv, K, None)
        err = np.linalg.norm(proj.reshape(-1, 2) - img[idx], axis=1)
        if best is None or len(idx) > best[0]:
            best = (len(idx), int(f), float(np.median(err)), np.round((-cv2.Rodrigues(rv)[0].T @ tv).ravel(), 3).tolist())
    return best


def run(cam):
    D = S.load(cam)
    Pf_all, _ = S.features(D["Tbg0"], np.zeros(len(D["g"])), 0.0)
    for e in sorted(set(D["ep"].tolist())):
        m = D["ep"] == e
        two = D["two"][m]
        Pf = Pf_all[m]
        obj = np.vstack([Pf, Pf[two]]).astype(np.float64)
        img = np.vstack([D["A"][m], D["B"][m][two]]).astype(np.float64)
        n_obs = len(obj)
        if n_obs < 12:
            print(cam, e, "windows", int(m.sum()), "observations", n_obs, "skipped (too few)", flush=True)
            continue
        r6 = pnp_best(obj, img, 6.0)
        r12 = pnp_best(obj, img, 12.0)
        print(cam, e, "windows", int(m.sum()), "obs", n_obs, "| thr6 best", r6, "| thr12 best", r12, flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
