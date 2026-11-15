#!/usr/bin/env python3
"""P3-TWIN-CAMS E2 step 22: top camera from held-brick correspondences (TCP-based), RANSAC, LOEO, G3.

Each kept frame (out/s21_held_top.json, held brick in the gripper): joint time t_j = t_video - OFF; q = pos0..4 interpolated
at t_j from joints.csv; TCP = kin.fk(q + DQ) (validated convention; DQ zero or the wrist-camera joint offsets); the brick
centre is the TCP plus DZ = 1.6 mm in z (TIP_FINAL 8.0 mm above the mat; brick centre 9.6 mm above the mat).
Candidates: (brick centre, every colour blob of that frame). Global PnP-RANSAC (OpenCV EPnP, 8 px) over the candidates for
f in {500, 615, 730}; Huber least squares on the inliers with fx = fy, principal point within 40 px of the centre and
|k1| < 0.3. Leave-one-episode-out (LOEO): a held-out frame's residual = distance from its projected centre to the nearest
blob within a 30 px gate. Also sweeps the clock offset (OFF_TEST) to check the measured top offset.
Run: envs/anomalib-env/bin/python -I tools/s22_held_fit.py <zero|wrist>
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/home/intel/intel-junction-2026/app/tools")
import kin  # noqa: E402

R = "/home/intel/intel-junction-2026"
E2 = f"{R}/work/P3-TWIN-CAMS/E2"
DZ = 1.6
OFF = 0.0333
GATE = 30.0


def load_joints():
    J = {}
    for f in glob.glob(f"{R}/logs/sort/any_*/ep_*/joints.csv"):
        rel = os.path.relpath(os.path.dirname(f), f"{R}/logs/sort")
        a = np.genfromtxt(f, delimiter=",", names=True)
        J[rel] = (a["t"], np.stack([a[f"pos{k}"] for k in range(6)], axis=1))
    return J


def build(frames, J, dq, off):
    rows = []
    for fr in frames:
        t, pos = J[fr["ep"]]
        tj = fr["t_video"] - off
        if tj < t[0] or tj > t[-1] or not fr["blobs"]:
            continue
        q = np.array([np.interp(tj, t, pos[:, k]) for k in range(5)]) + dq
        p, _ = kin.fk(list(q))
        P = np.asarray(p, float) + np.array([0.0, 0.0, DZ])
        rows.append((fr["i"], fr["ep"], P, fr["blobs"]))
    return rows


def candidates(rows):
    obj, img, fid, ep = [], [], [], []
    for k, (i, e, P, blobs) in enumerate(rows):
        for b in blobs:
            obj.append(P / 1000.0)
            img.append((b["u"], b["v"]))
            fid.append(k)
            ep.append(e)
    return np.array(obj, float), np.array(img, float), np.array(fid), np.array(ep)


def project(x, P):
    Rm = Rotation.from_rotvec(x[0:3]).as_matrix()
    Pc = P @ Rm.T + x[3:6]
    z = Pc[:, 2]
    zs = np.where(z > 1e-6, z, 1e-6)
    xn = Pc[:, 0] / zs
    yn = Pc[:, 1] / zs
    r2 = xn * xn + yn * yn
    d = 1.0 + x[9] * r2
    return np.stack([x[6] * xn * d + x[7], x[6] * yn * d + x[8]], axis=1)


def ransac(obj, img, f):
    K = np.array([[f, 0.0, 320.0], [0.0, f, 240.0], [0.0, 0.0, 1.0]])
    ok, rv, tv, inl = cv2.solvePnPRansac(obj, img, K, None, iterationsCount=3000, reprojectionError=8.0,
                                         confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < 6:
        return None
    idx = inl.ravel()
    rv, tv = cv2.solvePnPRefineLM(obj[idx], img[idx], K, None, rv, tv)
    return rv.ravel(), tv.ravel(), idx, f


def fit_all(rows):
    obj, img, _, _ = candidates(rows)
    best = None
    for f in (500.0, 615.0, 730.0):
        r = ransac(obj, img, f)
        if r is None:
            continue
        rv, tv, idx, f0 = r
        x0 = np.r_[rv, tv, f0, 320.0, 240.0, 0.0]
        lb = np.r_[[-np.pi] * 3, [-3, -3, -3], 400.0, 280.0, 200.0, -0.3]
        ub = np.r_[[np.pi] * 3, [3, 3, 3], 900.0, 360.0, 280.0, 0.3]
        oi, ii = obj[idx], img[idx]
        x0 = np.clip(x0, lb + 1e-9, ub - 1e-9)
        sol = least_squares(lambda x: (project(x, oi) - ii).ravel(), x0, bounds=(lb, ub), loss="huber",
                            f_scale=3.0, x_scale="jac", max_nfev=500)
        x = sol.x
        pr = project(x, obj)
        e = np.linalg.norm(pr - img, axis=1)
        inl_n = int((e <= 8.0).sum())
        if best is None or inl_n > best[0]:
            best = (inl_n, x, f0)
    return None if best is None else best[1]


def per_frame_residuals(x, rows):
    res, keep = [], []
    for i, e, P, blobs in rows:
        uv = project(x, (P / 1000.0)[None, :])[0]
        bu = np.array([[b["u"], b["v"]] for b in blobs], float)
        d = np.linalg.norm(bu - uv, axis=1)
        k = int(np.argmin(d))
        if d[k] <= GATE:
            res.append(float(d[k]))
            keep.append((i, e))
    return np.array(res), keep


def stats(v):
    v = np.asarray(v, float)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "median_px": round(float(np.median(v)), 3), "p90_px": round(float(np.percentile(v, 90)), 3),
            "rms_px": round(float(np.sqrt(np.mean(v * v))), 3)}


def describe(x):
    Rm = Rotation.from_rotvec(x[0:3]).as_matrix()
    C = -Rm.T @ x[3:6]
    axis = Rm[2, :]
    tilt = float(np.degrees(np.arccos(np.clip(-axis[2], -1, 1))))
    C_mm = C * 1000.0
    perp = float(C_mm[2] - kin.MAT_Z)
    depth = float(perp / max(-axis[2], 1e-6))
    return {"rvec": np.round(x[0:3], 6).tolist(), "t_m": np.round(x[3:6], 6).tolist(),
            "C_base_mm": np.round(C_mm, 1).tolist(), "fx=fy": round(float(x[6]), 2), "cx": round(float(x[7]), 2),
            "cy": round(float(x[8]), 2), "k1": round(float(x[9]), 5), "optical_axis_tilt_deg": round(tilt, 2),
            "height_above_mat_mm": round(perp, 1), "depth_at_centre_mm": round(depth, 1)}


def main(dq_mode):
    dq = np.zeros(5)
    if dq_mode == "wrist":
        M = json.load(open(f"{R}/app/eyeinhand/wrist_cam_model.json"))
        dq = np.asarray(M["joint_offsets_norm"], float)[:5]
    frames = json.load(open(f"{E2}/out/s21_held_top.json"))["frames"]
    J = load_joints()
    rows = build(frames, J, dq, OFF)
    eps = sorted(set(r[1] for r in rows))
    print("dq", dq_mode, "frames with blobs", len(rows), "episodes", len(eps), flush=True)
    sweep = {}
    for off in (-0.0667, -0.0333, 0.0, 0.0333, 0.0667, 0.1):
        rr = build(frames, J, dq, off)
        x = fit_all(rr)
        if x is None:
            sweep[str(off)] = None
            continue
        res, _ = per_frame_residuals(x, rr)
        sweep[str(off)] = stats(res)
        print("OFF_TEST", off, sweep[str(off)], flush=True)
    x_full = fit_all(rows)
    res_in, _ = per_frame_residuals(x_full, rows)
    out = {"dq": dq_mode, "DZ_mm": DZ, "OFF_s": OFF, "frames": len(rows), "episodes": len(eps),
           "full_fit": describe(x_full), "in_sample": stats(res_in), "offset_sweep": sweep, "loeo": {}}
    pooled = []
    for e in eps:
        tr = [r for r in rows if r[1] != e]
        te = [r for r in rows if r[1] == e]
        xe = fit_all(tr)
        if xe is None:
            out["loeo"][e] = None
            continue
        res, _ = per_frame_residuals(xe, te)
        out["loeo"][e] = {"test": stats(res), "camera": describe(xe)}
        pooled.append(res)
        print("LOEO", e, stats(res), flush=True)
    allr = np.concatenate(pooled) if pooled else np.array([])
    out["loeo_pooled"] = stats(allr)
    out["G3"] = {"median_le_3px": bool(out["loeo_pooled"].get("median_px", 99) <= 3.0),
                 "p90_le_6px": bool(out["loeo_pooled"].get("p90_px", 99) <= 6.0)}
    out["G3"]["pass"] = bool(out["G3"]["median_le_3px"] and out["G3"]["p90_le_6px"])
    json.dump(out, open(f"{E2}/out/s22_top_{dq_mode}.json", "w"), indent=1)
    np.save(f"{E2}/out/s22_top_{dq_mode}_x.npy", x_full)
    print("FULL", out["full_fit"], flush=True)
    print("IN-SAMPLE", out["in_sample"], flush=True)
    print("LOEO POOLED", out["loeo_pooled"], "G3", out["G3"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "zero")
