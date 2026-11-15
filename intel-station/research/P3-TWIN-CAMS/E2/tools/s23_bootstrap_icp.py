#!/usr/bin/env python3
"""P3-TWIN-CAMS E2 step 23: top camera from plate centres (bootstrap) and held-brick TCP points (ICP refine), LOEO, G3.

1. Bootstrap (no colour blobs used): camera parametrised by azimuth phi, roll psi and the mat point T the optical axis
   hits (Tx, Ty) at fixed tilt 20 deg, depth D in {554, 589} mm (the P3-TWIN-SCENE 589.46 mm depth at the centre, or its
   perpendicular equivalent 589.46 cos 20 = 554 mm), f = 615 px, principal point at the centre. Plate centres
   (PLATES_XY at MAT_Z) against the ellipse centres measured in the first pre_top frame give the 4 equations.
2. Score each bootstrap camera on the held-brick frames (out/s21_held_top.json; TCP from kin.fk plus DZ): the fraction of
   frames whose predicted centre has a colour blob within 10 px. Per-episode hit counts are stored once, so LOEO
   selection uses only training episodes.
3. ICP refine (Huber least squares, fx = fy, principal point within 40 px, |k1| < 0.3): nearest blob within a shrinking
   gate, 25, 15, 10, 8 px. LOEO by episode; G3 on held-out residuals (nearest blob within 30 px).
Run: envs/anomalib-env/bin/python -I tools/s23_bootstrap_icp.py
"""
import json
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/home/intel/intel-junction-2026/app/tools")
sys.path.insert(0, "/home/intel/intel-junction-2026/work/P3-TWIN-CAMS/E2/tools")
import kin  # noqa: E402
import s22_held_fit as F22  # noqa: E402

E2 = F22.E2
MAT_Z = float(kin.MAT_Z)
PL = np.array([[187.1, -86.1, MAT_Z], [194.6, 218.3, MAT_Z]])
PL_OBS = np.array([[240.4, 147.8], [576.3, 176.5]])
F0 = 615.0
TAU = np.radians(20.0)


def cam_from(phi, psi, tx, ty, D, tau=TAU):
    a = np.array([np.sin(tau) * np.cos(phi), np.sin(tau) * np.sin(phi), -np.cos(tau)])
    C = np.array([tx, ty, MAT_Z]) - D * a
    er = np.cross(a, [0.0, 0.0, 1.0])
    er /= np.linalg.norm(er)
    ed = np.cross(a, er)
    xc = np.cos(psi) * er + np.sin(psi) * ed
    yc = -np.sin(psi) * er + np.cos(psi) * ed
    return np.vstack([xc, yc, a]), C


def proj_R(R, C, f, cx, cy, P, k1=0.0):
    Pc = (P - C) @ R.T
    z = np.where(Pc[:, 2] > 1e-6, Pc[:, 2], 1e-6)
    x = Pc[:, 0] / z
    y = Pc[:, 1] / z
    d = 1.0 + k1 * (x * x + y * y)
    return np.stack([f * x * d + cx, f * y * d + cy], axis=1)


def proj_x(x, P):
    Rm = Rotation.from_rotvec(x[0:3]).as_matrix()
    return proj_R(Rm, x[3:6], x[6], x[7], x[8], P, x[9])


def bootstrap_candidates(D):
    sols = []
    for phi in np.radians(np.arange(0, 360, 15)):
        for psi in np.radians([0, 90, 180, 270]):
            def res(p):
                R, C = cam_from(p[0], p[1], p[2], p[3], D)
                return (proj_R(R, C, F0, 320.0, 240.0, PL) - PL_OBS).ravel()
            r = least_squares(res, [phi, psi, 190.0, 60.0], max_nfev=400)
            rms = float(np.sqrt(np.mean(r.fun ** 2)))
            if rms < 25.0:
                sols.append({"D": D, "phi": float(r.x[0]), "psi": float(r.x[1]), "tx": float(r.x[2]),
                             "ty": float(r.x[3]), "plate_rms_px": round(rms, 2)})
    uniq = []
    for s in sols:
        key = (round(s["phi"] % (2 * np.pi), 2), round(s["psi"] % (2 * np.pi), 2), round(s["tx"]), round(s["ty"]))
        if all(key != k for k, _ in uniq):
            uniq.append((key, s))
    return [s for _, s in uniq]


def to_x(c):
    R, C = cam_from(c["phi"], c["psi"], c["tx"], c["ty"], c["D"])
    rv = Rotation.from_matrix(R).as_rotvec()
    return np.r_[rv, C, F0, 320.0, 240.0, 0.0]


def hits_for(x, rows, gate=10.0):
    P = np.array([r[2] for r in rows], float)
    uv = proj_x(x, P)
    out = {}
    for k, (i, e, Pk, blobs) in enumerate(rows):
        bu = np.array([[b["u"], b["v"]] for b in blobs], float)
        if bu.size == 0:
            continue
        d = np.linalg.norm(bu - uv[k], axis=1)
        m = int(np.argmin(d))
        h, t = out.get(e, (0, 0))
        out[e] = (h + int(d[m] <= gate), t + 1)
    return out


def icp(x, rows, gates=(25.0, 15.0, 10.0, 8.0, 8.0)):
    lb = np.r_[[-np.pi] * 3, [-1500, -1500, 0], 450.0, 280.0, 200.0, -0.3]
    ub = np.r_[[np.pi] * 3, [1500, 1500, 2000], 800.0, 360.0, 280.0, 0.3]
    x = np.clip(np.asarray(x, float), lb + 1e-9, ub - 1e-9)
    for g in gates:
        uv = proj_x(x, np.array([r[2] for r in rows], float))
        P_pairs, uv_pairs = [], []
        for k, (i, e, Pk, blobs) in enumerate(rows):
            bu = np.array([[b["u"], b["v"]] for b in blobs], float)
            if bu.size == 0:
                continue
            d = np.linalg.norm(bu - uv[k], axis=1)
            m = int(np.argmin(d))
            if d[m] <= g:
                P_pairs.append(Pk)
                uv_pairs.append(bu[m])
        if len(P_pairs) < 20:
            break
        P_pairs = np.array(P_pairs, float)
        uv_pairs = np.array(uv_pairs, float)
        sol = least_squares(lambda y: (proj_x(y, P_pairs) - uv_pairs).ravel(), x, bounds=(lb, ub),
                            loss="huber", f_scale=3.0, x_scale="jac", max_nfev=300)
        x = sol.x
    return x


def residuals(x, rows, gate=30.0):
    P = np.array([r[2] for r in rows], float)
    uv = proj_x(x, P)
    res = []
    for k, (i, e, Pk, blobs) in enumerate(rows):
        bu = np.array([[b["u"], b["v"]] for b in blobs], float)
        if bu.size == 0:
            continue
        d = np.linalg.norm(bu - uv[k], axis=1)
        m = int(np.argmin(d))
        if d[m] <= gate:
            res.append((e, float(d[m])))
    return res


def stats(v):
    v = np.asarray(v, float)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "median_px": round(float(np.median(v)), 3), "p90_px": round(float(np.percentile(v, 90)), 3),
            "rms_px": round(float(np.sqrt(np.mean(v * v))), 3)}


def describe(x):
    Rm = Rotation.from_rotvec(x[0:3]).as_matrix()
    C = x[3:6]
    axis = Rm[2, :]
    tilt = float(np.degrees(np.arccos(np.clip(-axis[2], -1, 1))))
    perp = float(C[2] - MAT_Z)
    depth = float(perp / max(-axis[2], 1e-6))
    return {"rvec": np.round(x[0:3], 6).tolist(), "C_base_mm": np.round(C, 1).tolist(), "f": round(float(x[6]), 2),
            "cx": round(float(x[7]), 2), "cy": round(float(x[8]), 2), "k1": round(float(x[9]), 5),
            "optical_axis_tilt_deg": round(tilt, 2), "height_above_mat_mm": round(perp, 1),
            "depth_at_centre_mm": round(depth, 1)}


def main():
    frames = json.load(open(f"{E2}/out/s21_held_top.json"))["frames"]
    J = F22.load_joints()
    rows = F22.build(frames, J, np.zeros(5), F22.OFF)
    eps = sorted(set(r[1] for r in rows))
    print("rows", len(rows), "episodes", len(eps), flush=True)
    cands = bootstrap_candidates(554.0) + bootstrap_candidates(589.0)
    print("bootstrap candidates", len(cands), flush=True)
    table = []
    for c in cands:
        x0 = to_x(c)
        h = hits_for(x0, rows)
        tot_h = sum(v[0] for v in h.values())
        tot_t = sum(v[1] for v in h.values())
        table.append({"cand": c, "x0": x0, "hits": h, "score": tot_h / max(tot_t, 1)})
    table.sort(key=lambda t: -t["score"])
    print("top bootstrap scores (hit fraction at 10 px, plates rms):", flush=True)
    for t in table[:6]:
        print("  D", t["cand"]["D"], "score", round(t["score"], 4), "plate_rms", t["cand"]["plate_rms_px"],
              "init C", np.round(t["x0"][3:6], 1).tolist(), flush=True)
    best = table[0]
    x_full = icp(best["x0"], rows)
    res_in = residuals(x_full, rows)
    out = {"method": "plate bootstrap + held-brick ICP", "DZ_mm": F22.DZ, "OFF_s": F22.OFF,
           "bootstrap_top": [{"D": t["cand"]["D"], "score": round(t["score"], 4),
                              "plate_rms_px": t["cand"]["plate_rms_px"]} for t in table[:6]],
           "full_fit": describe(x_full), "in_sample": stats([r for _, r in res_in]), "loeo": {}}
    print("FULL", out["full_fit"], "IN-SAMPLE", out["in_sample"], flush=True)
    pooled = []
    for e in eps:
        tr = [r for r in rows if r[1] != e]
        te = [r for r in rows if r[1] == e]
        best_e = max(table, key=lambda t: sum(v[0] for k, v in t["hits"].items() if k != e) /
                     max(sum(v[1] for k, v in t["hits"].items() if k != e), 1))
        xe = icp(best_e["x0"], tr)
        rr = [r for _, r in residuals(xe, te)]
        out["loeo"][e] = {"test": stats(rr), "camera": describe(xe), "start_D": best_e["cand"]["D"]}
        pooled += rr
        print("LOEO", e, out["loeo"][e]["test"], flush=True)
    out["loeo_pooled"] = stats(pooled)
    out["G3"] = {"median_le_3px": bool(out["loeo_pooled"].get("median_px", 99) <= 3.0),
                 "p90_le_6px": bool(out["loeo_pooled"].get("p90_px", 99) <= 6.0)}
    out["G3"]["pass"] = bool(out["G3"]["median_le_3px"] and out["G3"]["p90_le_6px"])
    json.dump(out, open(f"{E2}/out/s23_top.json", "w"), indent=1)
    np.save(f"{E2}/out/s23_top_x.npy", x_full)
    print("LOEO POOLED", out["loeo_pooled"], "G3", out["G3"], flush=True)


if __name__ == "__main__":
    main()
