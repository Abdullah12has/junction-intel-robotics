#!/usr/bin/env python3
"""P3-TWIN-CAMS E1 step 9d: s9c model plus free per-joint scale factors on the arm joints (0.4 to 1.6).

s9b and s9c (free tip offsets, free tip-frame rotation, joint zero offsets of about 11 degrees) left the in-sample
medians at 39 px (top, 702 obs) and 61 px (close, 233 obs). This variant tests whether the normalised-to-radian scale
(CAL.k, from project_robots.json) is wrong for some joints: each arm joint gets a multiplier on its scale. If the
in-sample median stays above 3 px, the fit is reported as failed; LOEO runs only when it passes.
Imports s9_fit only (nothing from R/twin).
Run: envs/anomalib-env/bin/python -I tools/s9d_scale.py <top|close> <A|B>   (A: no joint offsets; B: + 5 offsets)
"""
import json
import math
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/home/intel/intel-junction-2026/work/P3-TWIN-CAMS/E1/tools")
import s9_fit as S  # noqa: E402

NP = 34
LB2 = np.concatenate([S.LB, [-0.10] * 6, [-0.6] * 3, [0.4] * 5])
UB2 = np.concatenate([S.UB, [0.10] * 6, [0.6] * 3, [1.6] * 5])
FREE1 = [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
FREE2A = list(range(15)) + list(range(20, 34))
FREE2B = list(range(34))
BASE = np.broadcast_to(np.asarray(S.CAL.sign * S.CAL.k, float), (5,)).copy()
ZERO = np.broadcast_to(np.asarray(S.CAL.zero, float), (5,)).copy()


def predict_uv(x, D):
    Q = D["pose"][:, :5] * (BASE * x[29:34])[None, :] + ZERO[None, :] + x[15:20][None, :]
    Tbg = S.fk_batch(Q)
    theta = x[13] + x[14] * D["g"]
    Rb, tb = Tbg[:, :3, :3], Tbg[:, :3, 3]
    Rc = Rotation.from_rotvec(x[26:29]).as_matrix()
    pf = Rc @ (S.PF + x[20:23] + x[12] * S.EZ)
    pml = S.PM_LINK + S.T_GJ[:3, :3].T @ x[23:26]
    v = np.einsum("nij,j->ni", S.rot_axis(S.AX, theta), pml)
    pm = (S.T_GJ[:3, :3] @ v.T).T + S.T_GJ[:3, 3] + x[12] * S.EZ
    pm = pm @ Rc.T
    Pf = np.einsum("nij,j->ni", Rb, pf) + tb
    Pm = np.einsum("nij,nj->ni", Rb, pm) + tb
    return S.project(Pf, x), S.project(Pm, x)


def residuals2(x, D):
    uf, um = predict_uv(x, D)
    e_two, e_one = S.pair(uf, um, D)
    prior_off = 3.0 * x[15:20] / math.radians(3.0)
    prior_g = np.array([3.0 * ((x[14] / S.K_G) - 1.0) / 0.5])
    return np.concatenate([e_two.ravel(), e_one.ravel(), prior_off, prior_g])


def distances2(x, D):
    uf, um = predict_uv(x, D)
    e_two, e_one = S.pair(uf, um, D)
    dt = np.stack([np.hypot(e_two[:, 0], e_two[:, 1]), np.hypot(e_two[:, 2], e_two[:, 3])], axis=1)
    do = np.hypot(e_one[:, 0], e_one[:, 1])
    return dt, do


def run_ls2(x0, free, D):
    x0 = np.clip(np.asarray(x0, float), LB2 + 1e-9, UB2 - 1e-9)
    idx = np.asarray(free)

    def fun(p):
        x = x0.copy()
        x[idx] = p
        return residuals2(x, D)

    r = least_squares(fun, x0[idx], bounds=(LB2[idx], UB2[idx]), loss="soft_l1", f_scale=3.0, x_scale="jac",
                      max_nfev=300)
    x = x0.copy()
    x[idx] = r.x
    dt, do = distances2(x, D)
    return x, float(np.median(np.concatenate([dt.ravel(), do])))


def fit2(D, n_cam, seed, offsets, top_k=3):
    rng = np.random.default_rng(seed)
    Pf0, _ = S.features(D["Tbg0"], np.zeros(len(D["g"])), 0.0)
    c0 = Pf0.mean(axis=0)
    cands = []
    for _ in range(n_cam):
        pose = S.start_pose(rng, c0)
        f0 = rng.uniform(450.0, 800.0)
        for a in S.A_GRID:
            for b in S.B_GRID:
                x0 = np.zeros(NP)
                x0[0:6] = pose
                x0[6] = x0[7] = f0
                x0[8], x0[9] = 320.0, 240.0
                x0[13], x0[14] = a, b
                x0[29:34] = 1.0
                x1, m1 = run_ls2(x0, FREE1, D)
                cands.append((m1, x1))
    cands.sort(key=lambda t: t[0])
    res = []
    for m1, x1 in cands[:top_k]:
        x2, m2 = run_ls2(x1, FREE2B if offsets else FREE2A, D)
        res.append((m2, x2, m1))
    res.sort(key=lambda t: t[0])
    starts = [{"stage1_train_median_px": round(m1, 3), "stage2_train_median_px": round(m2, 3)} for m2, _, m1 in res]
    return res[0][1], res[0][0], starts


def describe2(x):
    d = S.describe(x)
    d["fixed_tip_offset_gripper_mm"] = np.round(x[20:23] * 1000, 2).tolist()
    d["moving_tip_rest_offset_gripper_mm"] = np.round(x[23:26] * 1000, 2).tolist()
    d["tip_frame_rotation_deg"] = round(float(np.degrees(np.linalg.norm(x[26:29]))), 3)
    d["joint_scale_multipliers"] = np.round(x[29:34], 4).tolist()
    return d


def main(cam, tag):
    offsets = tag == "B"
    D = S.load(cam)
    t0 = time.time()
    print(cam, tag, "frames", len(D["g"]), "two-jaw", int(D["two"].sum()),
          "episodes", len(set(D["ep"].tolist())), "base k (rad/unit)", np.round(BASE, 6).tolist(), flush=True)
    x, med, starts = fit2(D, n_cam=12, seed=11, offsets=offsets, top_k=3)
    dt, do = distances2(x, D)
    ins = S.stats(np.concatenate([dt.ravel(), do]))
    out = {"camera": cam, "variant": tag,
           "model": "s9c model plus free per-joint scale multipliers (0.4 to 1.6) on the arm joints",
           "frames": int(len(D["g"])), "two_jaw_frames": int(D["two"].sum()),
           "observations": int(dt.size + do.size),
           "full_fit": {"params": describe2(x), "train_median_px": round(med, 3), "in_sample": ins,
                        "starts": starts}}
    print(cam, tag, "full fit in-sample", ins, "starts", starts, "t", round(time.time() - t0), flush=True)
    print(cam, tag, "scales", np.round(x[29:34], 4).tolist(), "offsets rad", np.round(x[15:20], 4).tolist(),
          flush=True)
    if ins["median_px"] <= 3.0:
        eps = sorted(set(D["ep"].tolist()))
        per, dt_all, do_all = {}, [], []
        for k, e in enumerate(eps):
            te = D["ep"] == e
            Dtr, Dte = S.subset(D, ~te), S.subset(D, te)
            xk, medk, _ = fit2(Dtr, n_cam=6, seed=100 + k, offsets=offsets)
            dtk, dok = distances2(xk, Dte)
            per[e] = {"obs": int(dtk.size + dok.size), "train_median_px": round(medk, 3),
                      "test": S.stats(np.concatenate([dtk.ravel(), dok]))}
            dt_all.append(dtk.ravel())
            do_all.append(dok)
            print(cam, tag, "fold", e, per[e], "t", round(time.time() - t0), flush=True)
        pooled = S.stats(np.concatenate(dt_all + do_all))
        g3 = {"median_le_3px": bool(pooled["median_px"] <= 3.0), "p90_le_6px": bool(pooled["p90_px"] <= 6.0)}
        g3["pass"] = bool(g3["median_le_3px"] and g3["p90_le_6px"])
        out["loeo_pooled"] = pooled
        out["loeo_per_episode"] = per
        out["G3"] = g3
    else:
        out["G3"] = {"ran": False, "reason": "in-sample median above 3 px; LOEO not run"}
    with open(f"{S.E1}/out/feat3_{cam}_{tag}.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(cam, tag, "G3", out["G3"], "done in", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
