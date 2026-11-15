#!/usr/bin/env python3
"""P3-TWIN-CAMS E1 step 9f (diagnostic, read-only): global PnP-RANSAC with the validated kinematics of app/tools/kin.py.

Joint conversion is kin.CAL.rad_from_norm(q_norm + DQ), the same path as kin.fk. DQ is zero, or the wrist-camera
joint offsets (app/eyeinhand/wrist_cam_model.json joint_offsets_norm, as wristcam._DQ). The fixed-jaw tip (saved
geometry, gripper_link frame) is paired with every observed jaw blob of each still window. OpenCV EPnP RANSAC over those
candidates; focal 300..1000 px, cx, cy at the 640x480 centre, no distortion. Reports the camera centre's height above the
mat plane (kin.MAT_Z) for comparison with the depth-measured distance (top 589.46 mm).
Run: envs/anomalib-env/bin/python -I tools/diag_kin.py <top|close> <zero|wrist>
"""
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/intel/intel-junction-2026/app/tools")
import kin  # noqa: E402

sys.path.insert(0, "/home/intel/intel-junction-2026/work/P3-TWIN-CAMS/E1/tools")
import s9_fit as S  # noqa: E402


def q_from(pose, dq):
    return np.array([np.asarray(kin.CAL.rad_from_norm(list(np.asarray(p[:5], float) + dq)), float)[:5]
                     for p in pose])


def run(cam, mode):
    D = S.load(cam)
    A, B, two = D["A"], D["B"], D["two"]
    N = len(A)
    win_idx = np.concatenate([np.arange(N), np.where(two)[0]])
    uv = np.vstack([A, B[two]]).astype(np.float64)
    dq = np.zeros(5)
    if mode == "wrist":
        M = json.load(open("/home/intel/intel-junction-2026/app/eyeinhand/wrist_cam_model.json"))
        dq = np.asarray(M["joint_offsets_norm"], float)[:5]
    Q = q_from(D["pose"], dq)
    chk = kin.CH.fk(list(Q[0]))
    chk2 = S.CH.fk(list(Q[0]))
    print("sanity: kin.CH.fk(q0) position (m or mm as returned)", np.round(np.asarray(chk)[:3, 3], 5).tolist(),
          "| S.CH.fk(q0)", np.round(np.asarray(chk2)[:3, 3], 5).tolist(), flush=True)
    Tbg = S.fk_batch(Q)
    Pf = np.einsum("nij,j->ni", Tbg[:, :3, :3], S.PF) + Tbg[:, :3, 3]
    obj = Pf[win_idx].astype(np.float64)
    results = []
    for f in (300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 1000.0):
        K = np.array([[f, 0.0, 320.0], [0.0, f, 240.0], [0.0, 0.0, 1.0]])
        ok, rv, tv, inl = cv2.solvePnPRansac(obj, uv, K, None, iterationsCount=1500,
                                             reprojectionError=8.0, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < 6:
            results.append((0, f, None, None, None))
            continue
        idx = inl.ravel()
        rv, tv = cv2.solvePnPRefineLM(obj[idx], uv[idx], K, None, rv, tv)
        proj, _ = cv2.projectPoints(obj, rv, tv, K, None)
        err = np.linalg.norm(proj.reshape(-1, 2) - uv, axis=1)
        mask = err <= 8.0
        Cm = (-cv2.Rodrigues(rv)[0].T @ tv).ravel()
        results.append((int(mask.sum()), f, round(float(np.median(err[mask])), 2) if mask.any() else None,
                        np.round(Cm, 4).tolist(), round(float(Cm[2] * 1000.0 - kin.MAT_Z), 1)))
    results.sort(key=lambda r: -r[0])
    print(cam, mode, "windows", N, "candidates", len(uv), "| inlier = reprojection <= 8 px after LM refinement",
          flush=True)
    print("(inliers, f px, median px of inliers, camera centre base frame m, camera height above mat mm):", flush=True)
    for r in results:
        print("  ", r, flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
