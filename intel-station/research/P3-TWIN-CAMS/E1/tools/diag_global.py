#!/usr/bin/env python3
"""P3-TWIN-CAMS E1 step 9e (diagnostic, read-only): global PnP-RANSAC on fixed-jaw tip candidates.

Each window contributes its fixed-jaw tip (FK with the saved geometry, delta 0) paired with EVERY observed jaw blob of
that window (both blobs for two-jaw windows). With the right camera and joint convention, about one inlier per window
should appear. This replaces the earlier diag scripts, which paired the second blob with the fixed tip and so mislabelled
half of the correspondences. Sweeps: joint zero (mid-range or homing-2048), 32 sign patterns, focal 300..1000 px
(cx, cy at the 640x480 centre, no distortion). Read-only on the recordings.
Run: envs/anomalib-env/bin/python -I tools/diag_global.py <top|close>   (writes out/diag_global_<cam>.log via stdout)
"""
import itertools
import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/intel/intel-junction-2026/work/P3-TWIN-CAMS/E1/tools")
import s9_fit as S  # noqa: E402

BODY = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def run(cam):
    D = S.load(cam)
    A, B, two = D["A"], D["B"], D["two"]
    N = len(A)
    win_idx = np.concatenate([np.arange(N), np.where(two)[0]])
    uv = np.vstack([A, B[two]]).astype(np.float64)
    P5 = D["pose"][:, :5].astype(float)
    base = np.broadcast_to(np.asarray(S.CAL.k, float), (5,)).copy()
    mid = np.array([(S.CAL.joints[j].range_min + S.CAL.joints[j].range_max) / 2.0 for j in BODY])
    zeros = {"mid": np.zeros(5), "homing2048": (mid - 2048.0) * 2.0 * np.pi / 4095.0}
    results = []
    for zname, z in zeros.items():
        for sg in itertools.product([1.0, -1.0], repeat=5):
            sg = np.array(sg)
            Q = sg[None, :] * (base[None, :] * P5 + z[None, :])
            Tbg = S.fk_batch(Q)
            Pf = np.einsum("nij,j->ni", Tbg[:, :3, :3], S.PF) + Tbg[:, :3, 3]
            obj = Pf[win_idx].astype(np.float64)
            for f in (300.0, 400.0, 500.0, 600.0, 800.0, 1000.0):
                K = np.array([[f, 0.0, 320.0], [0.0, f, 240.0], [0.0, 0.0, 1.0]])
                ok, rv, tv, inl = cv2.solvePnPRansac(obj, uv, K, None, iterationsCount=800,
                                                     reprojectionError=8.0, confidence=0.999,
                                                     flags=cv2.SOLVEPNP_EPNP)
                if not ok or inl is None or len(inl) < 6:
                    continue
                idx = inl.ravel()
                rv, tv = cv2.solvePnPRefineLM(obj[idx], uv[idx], K, None, rv, tv)
                proj, _ = cv2.projectPoints(obj, rv, tv, K, None)
                err = np.linalg.norm(proj.reshape(-1, 2) - uv, axis=1)
                inl_mask = err <= 8.0
                C = (-cv2.Rodrigues(rv)[0].T @ tv).ravel()
                results.append((int(inl_mask.sum()), zname, tuple(int(s) for s in sg), f,
                                round(float(np.median(err[inl_mask])), 2) if inl_mask.any() else None,
                                np.round(C, 3).tolist()))
    results.sort(key=lambda r: -r[0])
    print(cam, "windows", N, "candidates", len(uv), "(inlier = within 8 px of the refined model)", flush=True)
    print("top 12 (inliers, zero, sign pattern, f, median px of inliers, camera centre in base frame m):", flush=True)
    for r in results[:12]:
        print("  ", r, flush=True)
    hist = {}
    for r in results:
        hist[r[0] // 50 * 50] = hist.get(r[0] // 50 * 50, 0) + 1
    print("histogram of inlier counts (bin floor: number of combinations):", dict(sorted(hist.items())), flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
