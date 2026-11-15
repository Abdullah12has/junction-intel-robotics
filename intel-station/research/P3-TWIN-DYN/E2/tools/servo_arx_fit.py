#!/usr/bin/env python3
"""P3-TWIN-DYN E2: fast per-joint ARX servo model, goal -> present, with gravity term.

y[k] = a1*y[k-1] + a2*y[k-2] + b0*u[k-d] + b1*u[k-d-1] + c*g_j(q[k-1]) + e   (normalized units, frames at 30 Hz)
g_j = gravity torque (N m) from MuJoCo mj_rne (zero velocity) on the no-geometry SO-101 URDF.
Delay d (0..6 frames) chosen per joint by training one-step SSE. Leave-one-episode-out (LOEO) free-run scoring.
"""
import datetime
import json
import os
import sys

import mujoco
import numpy as np

R = "/home/intel/intel-junction-2026"
E1 = R + "/work/P3-TWIN-DYN/E1"
E2 = R + "/work/P3-TWIN-DYN/E2"
DYN = R + "/sim/twin/dynamics"
sys.path.insert(0, E1 + "/tools")
import twin_lib as T  # noqa: E402
import plotlib as P  # noqa: E402
from PIL import Image  # noqa: E402,F401

DMAX = 6
LAG_SH = np.arange(-3.0, 3.0001, 0.05)
E1_RMSE = [0.2716, 0.8435, 0.8373, 0.326, 0.1841, 0.2433]
E1_LAG = [1.05, 1.35, 1.35, 1.35, 0.6, 0.3]
GATE, GATE_LAG = 0.5, 1.0

RUN_SRC = '''def run_servo_arx(cfg, goal, grav, y0):
    """Open-loop servo model. cfg: per-joint dicts with a1, a2, b0, b1, c, e, d (frames).
    goal: (n, 6) normalized goal per frame. grav(q_norm) -> (6,) gravity torque, N m.
    y0: (2, 6) first two normalized present frames. Returns (n, 6) predicted normalized present."""
    import numpy as np
    n = goal.shape[0]
    y = np.zeros((n, 6))
    y[:2] = y0[:2]
    for k in range(2, n):
        g = grav(y[k - 1])
        for j, p in enumerate(cfg):
            d = p["d"]
            u0 = goal[max(k - d, 0), j]
            u1 = goal[max(k - d - 1, 0), j]
            y[k, j] = p["a1"] * y[k - 1, j] + p["a2"] * y[k - 2, j] + p["b0"] * u0 + p["b1"] * u1 + p["c"] * g[j] + p["e"]
    return y
'''

GRAV_SRC = '''

def make_grav(urdf_path, k):
    """k: rad per normalized unit (6,). Returns grav(q_norm) -> gravity torque (6,) N m via MuJoCo mj_rne."""
    import mujoco
    import numpy as np
    m = mujoco.MjModel.from_xml_path(urdf_path)
    d = mujoco.MjData(m)
    out = np.zeros(6)

    def grav(qn):
        d.qpos[:] = np.asarray(qn) * np.asarray(k)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        mujoco.mj_rne(m, d, 0, out)
        return out.copy()
    return grav
'''


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fit_joint(eps, grav_rows, j, dmax=DMAX):
    best = None
    for d in range(dmax + 1):
        xs, ys = [], []
        for ep, G in zip(eps, grav_rows):
            y = ep["pos"][:, j]
            u = ep["goal"][:, j]
            g = G[:, j]
            n = len(y)
            ks = np.arange(max(2, d + 1), n)
            ks = ks[T.mask_for(ep, j)[ks]]
            xs.append(np.column_stack([y[ks - 1], y[ks - 2], u[ks - d], u[ks - d - 1], g[ks], np.ones(len(ks))]))
            ys.append(y[ks])
        X = np.vstack(xs)
        Y = np.concatenate(ys)
        coef = np.linalg.lstsq(X, Y, rcond=None)[0]
        sse = float(np.sum((X @ coef - Y) ** 2))
        if best is None or sse < best[0]:
            best = (sse, d, coef, len(Y))
    sse, d, coef, nrow = best
    return {"d": int(d), "a1": float(coef[0]), "a2": float(coef[1]), "b0": float(coef[2]),
            "b1": float(coef[3]), "c": float(coef[4]), "e": float(coef[5]),
            "train_one_step_rmse": float(np.sqrt(sse / nrow))}


def free_run(ep, coefs, grav):
    u = ep["goal"]
    n = len(u)
    yp = np.zeros((n, 6))
    yp[:2] = ep["pos"][:2]
    for k in range(2, n):
        g = grav(yp[k - 1])
        for j, c in enumerate(coefs):
            d = c["d"]
            yp[k, j] = (c["a1"] * yp[k - 1, j] + c["a2"] * yp[k - 2, j] + c["b0"] * u[max(k - d, 0), j]
                        + c["b1"] * u[max(k - d - 1, 0), j] + c["c"] * g[j] + c["e"])
    return yp


def score(ep, yp, sq, cnt, lagsq):
    idx = np.arange(len(yp), dtype=float)
    for j in range(6):
        m = T.mask_for(ep, j)
        err = yp[:, j] - ep["pos"][:, j]
        sq[j] += np.sum(err[m] ** 2)
        cnt[j] += m.sum()
        for si, s in enumerate(LAG_SH):
            q = np.interp(idx - s, idx, yp[:, j])
            lagsq[si, j] += np.sum((q - ep["pos"][:, j])[m] ** 2)


def main():
    os.makedirs(E2, exist_ok=True)
    k = T.calib_k()
    grav = None
    ns = {}
    exec(GRAV_SRC, ns)
    grav = ns["make_grav"](T.NOGEOM_URDF, k)
    exec(RUN_SRC, ns)
    eps = T.load_episodes()
    Gs = [np.array([grav(p) for p in ep["pos"]]) for ep in eps]
    print(now(), "episodes", len(eps), "rows", sum(len(e["t"]) for e in eps), flush=True)

    sq = np.zeros(6)
    cnt = np.zeros(6)
    lagsq = np.zeros((len(LAG_SH), 6))
    per_fold = []
    fold2 = None
    for e in range(len(eps)):
        tr = [i for i in range(len(eps)) if i != e]
        coefs = [fit_joint([eps[i] for i in tr], [Gs[i] for i in tr], j) for j in range(6)]
        yp = free_run(eps[e], coefs, grav)
        score(eps[e], yp, sq, cnt, lagsq)
        ef = [float(np.sqrt(np.mean((yp[T.mask_for(eps[e], j), j] - eps[e]["pos"][T.mask_for(eps[e], j), j]) ** 2)))
              for j in range(6)]
        per_fold.append({"episode": eps[e]["name"], "rmse_norm": np.round(ef, 4).tolist()})
        if e == 2:
            fold2 = yp.copy()
    rmse = np.sqrt(sq / cnt)
    best_sh = LAG_SH[np.argmin(lagsq, axis=0)]
    lag = np.abs(best_sh)
    loeo_coefs = None

    full = [fit_joint(eps, Gs, j) for j in range(6)]
    yfull = free_run(eps[2], full, grav)
    ns2 = {}
    exec(RUN_SRC, ns2)
    yfun = ns2["run_servo_arx"]([dict(c) for c in full], eps[2]["goal"], grav, eps[2]["pos"][:2])
    selfcheck = float(np.max(np.abs(yfun - yfull)))

    names = T.JOINTS
    print("joint, delay_frames, c, LOEO_rmse_norm, E1_rmse, LOEO_lag_frames, E1_lag", flush=True)
    for j in range(6):
        print(names[j], full[j]["d"], round(full[j]["c"], 5), round(float(rmse[j]), 4), E1_RMSE[j],
              round(float(lag[j]), 2), E1_LAG[j], flush=True)
    print("selfcheck max |run_servo_arx - free_run| =", selfcheck, flush=True)

    res = {"stage": "servo-arx-v2", "utc": now(), "rate_hz_nominal": 30.0,
           "model": "y[k]=a1*y[k-1]+a2*y[k-2]+b0*u[k-d]+b1*u[k-d-1]+c*g_j(q[k-1])+e, normalized units, frames",
           "gravity": "MuJoCo mj_rne at zero velocity on work/P3-TWIN-DYN/E1/urdf/so101_nogeom.urdf, q = normalized * k",
           "k_rad_per_norm_unit": dict(zip(names, [float(x) for x in k])),
           "per_joint_fit_on_all_17": {names[j]: full[j] for j in range(6)},
           "loeo": {"rmse_norm": [float(x) for x in rmse], "lag_frames": [float(x) for x in lag],
                    "e1_rmse_norm": E1_RMSE, "e1_lag_frames": E1_LAG,
                    "g2_rmse_pass": [bool(x <= GATE) for x in rmse], "g2_lag_pass": [bool(x <= GATE_LAG) for x in lag],
                    "per_fold_rmse_norm": per_fold},
           "selfcheck_max_abs_diff": selfcheck,
           "run_function_source": RUN_SRC}
    with open(E2 + "/servo_arx_fit.json", "w") as fh:
        json.dump(res, fh, indent=1)
    pub = {k2: res[k2] for k2 in ("stage", "utc", "rate_hz_nominal", "model", "gravity",
                                    "k_rad_per_norm_unit", "per_joint_fit_on_all_17", "run_function_source")}
    pub["loeo_summary"] = {"rmse_norm": res["loeo"]["rmse_norm"], "lag_frames": res["loeo"]["lag_frames"],
                           "g2_rmse_pass": res["loeo"]["g2_rmse_pass"], "g2_lag_pass": res["loeo"]["g2_lag_pass"]}
    with open(DYN + "/servo_arx_v2.json", "w") as fh:
        json.dump(pub, fh, indent=1)
    with open(DYN + "/servo_arx_v2.py", "w") as fh:
        fh.write('"""P3-TWIN-DYN E2 servo model: run with coefficients in servo_arx_v2.json (per_joint_fit_on_all_17)."""\n')
        fh.write(RUN_SRC + GRAV_SRC)

    # plots
    img, draw = P.new_canvas(1000, 1200)
    draw.text((20, 8), "E2 held-out fold 2 (any_1740/ep_0001) free run: logged goal (orange), logged present (blue), "
              "ARX free run (red), normalized units", fill=(0, 0, 0), font=P.FONT)
    ep = eps[2]
    t = ep["t"]
    for j in range(6):
        box = (90, 40 + j * 190, 900, 40 + j * 190 + 150)
        vals = np.concatenate([ep["goal"][:, j], ep["pos"][:, j], fold2[:, j]])
        lo_, hi_ = float(np.nanmin(vals)), float(np.nanmax(vals))
        pad = 0.05 * (hi_ - lo_ + 1e-6)
        pn = P.Panel(draw, box, (0, t[-1]), (lo_ - pad, hi_ + pad), title=names[j], xlabel="time s")
        pn.line(t, ep["goal"][:, j], "o", 1)
        pn.line(t, ep["pos"][:, j], "b", 1)
        pn.line(t, fold2[:, j], "r", 1)
    img.save(E2 + "/e2_fold2_overlay.png")

    img, draw = P.new_canvas(900, 560)
    labels = ["pan", "lift", "elbow", "wrist_flex", "roll", "gripper"]
    P.bar_chart(draw, (70, 40, 860, 250), labels, E1_RMSE, GATE, "E1 LOEO RMSE (normalized)", "units")
    P.bar_chart(draw, (70, 300, 860, 510), labels, [float(x) for x in rmse], GATE,
                "E2 ARX LOEO RMSE (normalized), gate 0.5", "units")
    img.save(E2 + "/e2_loeo_rmse.png")

    with open(E2 + "/e2_done.txt", "w") as fh:
        fh.write("done " + now() + "\n")
    print("done", now(), flush=True)


if __name__ == "__main__":
    main()
