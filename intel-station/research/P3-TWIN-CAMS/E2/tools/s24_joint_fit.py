#!/usr/bin/env python3
"""P3-TWIN-CAMS E2 step 24 (final): held-brick fit with physical intrinsics priors, free arm joint offsets (prior and sigma
from the wrist-camera fit), camera pose free, Huber loss, LOEO by episode, G3.
Joint offsets enter through a linearised Jacobian of the held-brick TCP (six kin passes at the prior, +EPS per joint).
Run: envs/anomalib-env/bin/python -I tools/s24_joint_fit.py
"""
import json
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/home/intel/intel-junction-2026/app/tools")
sys.path.insert(0, "/home/intel/intel-junction-2026/work/P3-TWIN-CAMS/E2/tools")
import kin  # noqa: E402
import s22_held_fit as F22  # noqa: E402
import s23_bootstrap_icp as S23  # noqa: E402

E2 = F22.E2
DQ0 = np.array([0.013158, -5.641743, -2.486231, -2.757761, 0.001345])
SIG = 3.0
EPS = 0.5
MAT_Z = float(kin.MAT_Z)
LB = np.r_[[-np.pi] * 3, [-3000.0, -3000.0, 0.0], 575.0, 290.0, 210.0, -0.15, DQ0 - 20.0]
UB = np.r_[[np.pi] * 3, [3000.0, 3000.0, 3000.0], 655.0, 350.0, 270.0, 0.15, DQ0 + 20.0]

frames = json.load(open(f"{E2}/out/s21_held_top.json"))["frames"]
J = F22.load_joints()
rows0 = F22.build(frames, J, DQ0, F22.OFF)
N = len(rows0)
P0 = np.array([r[2] for r in rows0], float)
EP = np.array([r[1] for r in rows0])
BL = [np.array([[b["u"], b["v"]] for b in r[3]], float).reshape(-1, 2) for r in rows0]
Jac = np.zeros((N, 3, 5))
for k in range(5):
    dq = DQ0.copy()
    dq[k] += EPS
    rk = F22.build(frames, J, dq, F22.OFF)
    assert len(rk) == N
    Jac[:, :, k] = (np.array([r[2] for r in rk], float) - P0) / EPS
CAND = S23.bootstrap_candidates(554.0) + S23.bootstrap_candidates(589.0)


def points(x, idx):
    return P0[idx] + np.einsum("nik,k->ni", Jac[idx], x[10:15] - DQ0)


def nearest(uv, idx, gate):
    sel, suv, dd = [], [], []
    for n, k in enumerate(idx):
        bu = BL[k]
        if bu.shape[0] == 0:
            continue
        d = np.linalg.norm(bu - uv[n], axis=1)
        m = int(np.argmin(d))
        if d[m] <= gate:
            sel.append(n)
            suv.append(bu[m])
            dd.append(d[m])
    return np.array(sel, int), np.array(suv, float).reshape(-1, 2), np.array(dd, float)


def resid(x, idx, suv):
    uv = S23.proj_x(x[:10], points(x, idx))
    return np.r_[(uv - suv).ravel(), (x[10:15] - DQ0) / SIG]


def fit(idx, x):
    for gate in (30.0, 15.0, 10.0):
        uv = S23.proj_x(x[:10], points(x, idx))
        sn, suv, _ = nearest(uv, idx, gate)
        if len(sn) < 30:
            break
        sol = least_squares(resid, x, args=(idx[sn], suv), bounds=(LB, UB), loss="huber", f_scale=3.0,
                            x_scale="jac", max_nfev=200)
        x = sol.x
    return x


def clipx(x):
    return np.clip(x, LB + 1e-6, UB - 1e-6)


def best_fit(idx):
    scored = []
    for c in CAND:
        x = clipx(np.r_[S23.to_x(c), DQ0])
        sn, _, _ = nearest(S23.proj_x(x[:10], points(x, idx)), idx, 30.0)
        scored.append((len(sn), x))
    scored.sort(key=lambda t: -t[0])
    best = None
    for _, x0 in scored[:4]:
        x = fit(idx, x0)
        _, _, dd = nearest(S23.proj_x(x[:10], points(x, idx)), idx, 30.0)
        if len(dd) < 30:
            continue
        key = (int((dd <= 10.0).sum()), -float(np.median(dd)))
        if best is None or key > best[0]:
            best = (key, x)
    return None if best is None else best[1]


def heldout(x, idx):
    _, _, dd = nearest(S23.proj_x(x[:10], points(x, idx)), idx, 30.0)
    return dd


def stats(v):
    v = np.asarray(v, float)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "median_px": round(float(np.median(v)), 3),
            "p90_px": round(float(np.percentile(v, 90)), 3), "rms_px": round(float(np.sqrt(np.mean(v * v))), 3)}


def describe(x):
    Rm = Rotation.from_rotvec(x[0:3]).as_matrix()
    C = x[3:6]
    axis = Rm[2, :]
    tilt = float(np.degrees(np.arccos(np.clip(-axis[2], -1.0, 1.0))))
    height = float(C[2] - MAT_Z)
    return {"C_base_mm": np.round(C, 1).tolist(), "height_above_mat_mm": round(height, 1),
            "tilt_deg": round(tilt, 2), "depth_along_axis_mm": round(height / max(-axis[2], 1e-6), 1),
            "fx=fy": round(float(x[6]), 2), "cx": round(float(x[7]), 2), "cy": round(float(x[8]), 2),
            "k1": round(float(x[9]), 4), "joint_offsets_fit": np.round(x[10:15], 4).tolist(),
            "joint_offsets_minus_prior_sigma": np.round((x[10:15] - DQ0) / SIG, 3).tolist()}


def main():
    idx_all = np.arange(N)
    eps = sorted(set(EP.tolist()))
    out = {"name": "s24", "frames": N, "episodes": eps,
           "priors": {"fx_fy_mm_px": "615 +- 40", "principal_point": "within 30 px of centre", "abs_k1_max": 0.15,
                      "dq_prior": DQ0.tolist(), "dq_sigma": SIG, "depth_prior_mm": 589, "tilt_prior_deg": 20},
           "loeo": {}}
    pooled = []
    for e in eps:
        tr = idx_all[EP != e]
        te = idx_all[EP == e]
        x = best_fit(tr)
        if x is None:
            out["loeo"][e] = None
            print("LOEO", e, "no fit", flush=True)
            continue
        dd = heldout(x, te)
        pooled.append(dd)
        out["loeo"][e] = {"test": stats(dd), "test_frames": int(te.size), "fit": describe(x)}
        print("LOEO", e, out["loeo"][e]["test"], out["loeo"][e]["fit"]["height_above_mat_mm"],
              out["loeo"][e]["fit"]["tilt_deg"], flush=True)
    allr = np.concatenate(pooled) if pooled else np.array([])
    out["loeo_pooled"] = stats(allr)
    out["loeo_in_gate_fraction"] = round(float(allr.size) / float(N), 3)
    xf = best_fit(idx_all)
    out["full_fit"] = describe(xf) if xf is not None else None
    out["in_sample"] = stats(heldout(xf, idx_all)) if xf is not None else None
    g3 = {"median_le_3px": bool(out["loeo_pooled"].get("median_px", 99.0) <= 3.0),
          "p90_le_6px": bool(out["loeo_pooled"].get("p90_px", 99.0) <= 6.0)}
    g3["pass"] = bool(g3["median_le_3px"] and g3["p90_le_6px"])
    out["G3"] = g3
    json.dump(out, open(f"{E2}/out/s24_top.json", "w"), indent=1)
    print("FULL", out["full_fit"], flush=True)
    print("IN-SAMPLE", out["in_sample"], flush=True)
    print("LOEO POOLED", out["loeo_pooled"], "in_gate", out["loeo_in_gate_fraction"], "G3", g3, flush=True)


if __name__ == "__main__":
    main()
