#!/usr/bin/env python3
"""P3-TWIN-DYN E3 grey-box fit (one background job).

Per joint, fitted: kp, kv, frictionloss, armature, transport delay. Forcerange fixed at 3 N.m (STS3215 stall).
One global link mass/inertia scale s (bodies 1..n) fitted. kp for shoulder_lift and elbow_flex is initialised
from still windows (goal and position range <= 0.5 normalized units over 31 frames) by regressing the
steady-state offset (ctrl - q) on the MuJoCo gravity bias torque (qfrc_bias at zero velocity).
Multi-start (8, or 2 in test), least_squares trf with ftol 1e-4 and max_nfev 500; LOEO folds warm-started from
the E1 fold fits with 40 nfev; G2 per joint (pooled LOEO RMSE <= 0.5 and |lag| <= 1 frame); PNG outputs.
Not fitted in E3: deadband, COM scale per link. Lag is the best shift of the simulated trajectory on a 5 ms grid.
Env: E3_TEST=1 small test run; E3_OUT output directory.
"""
import concurrent.futures as cf
import json
import os
import sys
import time

import mujoco
import numpy as np
from mujoco import rollout as mjro
from PIL import Image, ImageDraw
from scipy.optimize import least_squares

R = "/home/intel/intel-junction-2026"
E1_TOOLS = f"{R}/work/P3-TWIN-DYN/E1/tools"
E1_LOEO = f"{R}/work/P3-TWIN-DYN/E1/loeo"
sys.path.insert(0, E1_TOOLS)
import twin_lib as T  # noqa: E402

TEST = os.environ.get("E3_TEST") == "1"
OUT = os.environ.get("E3_OUT", "/tmp/e3test" if TEST else f"{R}/work/P3-TWIN-DYN/E3")
FR = 3.0
FREE5 = [0, 1, 2, 3, 5]
LO5 = np.array([[0.5, 1e-4, 1e-5, 1e-6, 0.002]] * 6)
HI5 = np.array([[40.0, 5.0, 3.0, 0.02, 0.3]] * 6)
S_LO, S_HI = 0.5, 2.0
START_BUDGET = float(os.environ.get("E3_START_BUDGET_S", "5400"))  # wall-clock cap per start (best point kept)
FOLD_BUDGET = float(os.environ.get("E3_FOLD_BUDGET_S", "1800"))  # wall-clock cap per LOEO fold
KV_STALL_LIMIT = 3.0 / (1.0 / 0.22 * 60.0 * np.pi / 180.0)  # stall torque / no-load speed ~ 0.63
FRAME = 1.0 / 30.0
SHIFTS = np.arange(-0.05, 0.2001, 0.005)
JOINTS = T.JOINTS
LO = np.concatenate([np.log(LO5).ravel(), [np.log(S_LO)]])
HI = np.concatenate([np.log(HI5).ravel(), [np.log(S_HI)]])


def unpack(z):
    z = np.asarray(z, float)
    p = np.zeros((6, 6))
    p[:, 4] = FR
    p[:, FREE5] = np.exp(z[:30]).reshape(6, 5)
    return p, float(np.exp(z[30]))


def pack(p, s):
    return np.concatenate([np.log(p[:, FREE5]).ravel(), [np.log(s)]])


def clip_z(z):
    return np.clip(z, LO + 1e-9, HI - 1e-9)


def simulate(eps, p, s, ro, nthread):
    m = mujoco.MjModel.from_xml_string(T.make_xml(p))
    m.body_mass[1:] *= s
    m.body_inertia[1:] *= s
    datas = [mujoco.MjData(m) for _ in range(nthread)]
    out = [None] * len(eps)
    for g in T.groups_by_length(eps):
        sub = [eps[i] for i in g]
        ns = T.nstep_for(sub)
        init, ctrl = T.build_inputs(sub, p, ns)
        st, _ = ro.rollout(m, datas, init, ctrl, nstep=ns)
        for loc, i in enumerate(g):
            out[i] = st[loc]
    return out


def residual(eps, z, ro, nthread):
    p, s = unpack(z)
    r = T.residual_vector(eps, simulate(eps, p, s, ro, nthread))
    return np.nan_to_num(r, nan=1e3, posinf=1e3, neginf=-1e3)


class _Budget(Exception):
    pass


class _Res:
    pass


def fit(eps, z0, max_nfev, nthread, budget_s):
    """least_squares trf (ftol 1e-4, max_nfev); stops at budget_s wall-clock and returns the best point seen."""
    ro = mjro.Rollout(nthread=nthread)
    t_end = time.time() + budget_s
    best = {"c": np.inf, "z": np.array(clip_z(z0), float)}

    def fun(z):
        if time.time() > t_end:
            raise _Budget()
        r = residual(eps, z, ro, nthread)
        c = 0.5 * float(r @ r)
        if c < best["c"]:
            best["c"] = c
            best["z"] = np.array(z, float).copy()
        return r

    try:
        return least_squares(fun, clip_z(z0), bounds=(LO, HI), method="trf", x_scale="jac", diff_step=1e-3,
                             max_nfev=max_nfev, ftol=1e-4, xtol=1e-10, gtol=1e-10)
    except _Budget:
        out = _Res()
        out.x = best["z"]
        out.cost = best["c"]
        out.status = -1
        out.nfev = -1
        out.message = "time budget reached; best point so far"
        return out


def rmse_by_joint(eps, states):
    sq = np.zeros(6)
    cnt = np.zeros(6)
    errs = T.errors_per_episode(eps, states)
    for e, ep in enumerate(eps):
        for j in range(6):
            x = errs[e][j][T.mask_for(ep, j)]
            sq[j] += float(np.sum(x ** 2))
            cnt[j] += x.size
    return sq, cnt


def still_windows(eps, win=31, step=15, tol=0.5):
    rows = []
    for e, ep in enumerate(eps):
        for a in range(0, len(ep["t"]) - win + 1, step):
            ok = (np.ptp(ep["goal"][a:a + win], axis=0) <= tol) & (np.ptp(ep["pos"][a:a + win], axis=0) <= tol)
            for j in range(6):
                if ok[j]:
                    rows.append((e, j, a))
    return rows


def stage0(eps):
    k = T.calib_k()
    m = mujoco.MjModel.from_xml_string(T.make_xml(T.INIT))
    d = mujoco.MjData(m)
    acc = {j: ([], []) for j in range(6)}
    for e, j, a in still_windows(eps):
        ep = eps[e]
        sl = slice(a, a + 31)
        q = k * ep["pos"][sl].mean(axis=0)
        off = float(k[j] * (ep["goal"][sl, j].mean() - ep["pos"][sl, j].mean()))
        d.qpos[:6] = q
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        acc[j][0].append(float(d.qfrc_bias[j]))
        acc[j][1].append(off)
    info, kp_init = {}, {}
    for j in range(6):
        g = np.array(acc[j][0])
        o = np.array(acc[j][1])
        rec = {"n_windows": int(g.size)}
        if g.size >= 20 and np.sum(g * g) > 0:
            slope = float(np.sum(g * o) / np.sum(g * g))
            ss_res = float(np.sum((o - slope * g) ** 2))
            ss_tot = float(np.sum((o - o.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            rec.update({"slope_rad_per_Nm": slope, "r2": r2, "max_abs_gravity_Nm": float(np.max(np.abs(g)))})
            if slope > 0 and r2 > 0.05 and j in (1, 2):
                kp_init[j] = 1.0 / slope
        info[JOINTS[j]] = rec
    return info, kp_init


def start_job(args):
    idx, z, eps, nfev, path = args
    if os.path.exists(path):
        return json.load(open(path))
    t0 = time.time()
    try:
        res = fit(eps, z, nfev, 1, START_BUDGET)
        p, s = unpack(res.x)
        sq, cnt = rmse_by_joint(eps, simulate(eps, p, s, mjro.Rollout(nthread=1), 1))
        rec = {"idx": idx, "status": int(res.status), "message": str(res.message), "cost": float(res.cost),
               "nfev": int(res.nfev), "s": s, "p_6x6": p.tolist(), "z": res.x.tolist(),
               "insample_rmse_norm": np.sqrt(sq / cnt).tolist(), "seconds": time.time() - t0}
    except Exception as ex:  # record and continue with the other starts
        rec = {"idx": idx, "status": "error", "message": repr(ex), "seconds": time.time() - t0}
    json.dump(rec, open(path, "w"), indent=1)
    return rec


def loeo_job(args):
    i, eps, z_best, nfev, path = args
    if os.path.exists(path):
        return json.load(open(path))
    t0 = time.time()
    try:
        init = "full_best"
        z_init = np.asarray(z_best, float)
        e1 = f"{E1_LOEO}/fold_{i:02d}.json"
        if os.path.exists(e1):
            p1 = np.array(json.load(open(e1))["p_6x6"], float)
            z_init = pack(p1, 1.0)
            init = "E1_fold"
        train = [ep for kk, ep in enumerate(eps) if kk != i]
        test_ep = [eps[i]]
        res = fit(train, z_init, nfev, 1, FOLD_BUDGET)
        p, s = unpack(res.x)
        st = simulate(test_ep, p, s, mjro.Rollout(nthread=1), 1)[0]
        ep = test_ep[0]
        k = T.calib_k()
        ts = st[:, 0]
        q = st[:, 1:7]
        sq, cnt = rmse_by_joint(test_ep, [st])
        mk = [T.mask_for(ep, j) for j in range(6)]
        lag_sq = np.zeros((len(SHIFTS), 6))
        for si, sh in enumerate(SHIFTS):
            qs = np.column_stack([np.interp(ep["t"] - sh, ts, q[:, j]) for j in range(6)])
            e_ = (qs - ep["pos"] * k) / k
            for j in range(6):
                lag_sq[si, j] = float(np.sum(e_[mk[j], j] ** 2))
        lag_best = SHIFTS[np.argmin(lag_sq, axis=0)]
        sim_at = np.column_stack([np.interp(ep["t"], ts, q[:, j]) for j in range(6)]) / k
        rec = {"fold": i, "episode": ep["name"], "rows": int(len(ep["t"])), "init": init,
               "sq": sq.tolist(), "cnt": cnt.tolist(), "rmse_norm": np.sqrt(sq / cnt).tolist(),
               "lag_sq": lag_sq.tolist(), "shifts": SHIFTS.tolist(), "lag_best_s": lag_best.tolist(),
               "nfev": int(res.nfev), "status": int(res.status), "cost": float(res.cost), "s": s,
               "p_6x6": p.tolist(), "t": np.round(ep["t"], 4).tolist(),
               "meas_norm": np.round(ep["pos"], 5).tolist(), "sim_norm": np.round(sim_at, 5).tolist(),
               "seconds": time.time() - t0}
    except Exception as ex:
        rec = {"fold": i, "status": "error", "message": repr(ex), "seconds": time.time() - t0}
    json.dump(rec, open(path, "w"))
    return rec


def bar_png(path, e3, e1, gate, title):
    W, H = 900, 520
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = 70, 60, W - 30, H - 60
    top = max(max(e3), max(e1), gate) * 1.15
    d.line([(x0, y1), (x1, y1)], fill="black")
    d.line([(x0, y0), (x0, y1)], fill="black")
    gy = y1 - gate / top * (y1 - y0)
    d.line([(x0, gy), (x1, gy)], fill="red")
    d.text((x1 - 120, gy - 14), "gate 0.5", fill="red")
    slot = (x1 - x0) / 6.0
    for j in range(6):
        bx = x0 + slot * j + slot * 0.15
        w = slot * 0.3
        for off, val, col in ((0.0, e1[j], (160, 160, 160)), (w + 3.0, e3[j], (30, 90, 200))):
            h = val / top * (y1 - y0)
            d.rectangle([int(bx + off), int(y1 - h), int(bx + off + w), int(y1)], fill=col)
            d.text((int(bx + off), int(y1 - h - 12)), f"{val:.2f}", fill="black")
        d.text((int(x0 + slot * j + 4), y1 + 8), JOINTS[j], fill="black")
    d.text((x0, 12), title, fill="black")
    d.text((x0, 32), "grey: E1   blue: E3   (pooled LOEO RMSE, normalized units)", fill="black")
    im.save(path)


def overlay_png(path, t, meas, sim, title):
    W, H = 1000, 1080
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    t = np.asarray(t, float)
    tmax = float(t[-1]) if t[-1] > 0 else 1.0
    meas = np.asarray(meas, float)
    sim = np.asarray(sim, float)
    ph = 170
    left, right = 80, W - 20
    for j in range(6):
        top = 40 + j * ph
        bot = top + ph - 30
        a = meas[:, j]
        b = sim[:, j]
        lo = min(a.min(), b.min())
        hi = max(a.max(), b.max())
        span = (hi - lo) or 1.0

        def xy(tt, vv):
            return (left + tt / tmax * (right - left), bot - (vv - lo) / span * (bot - top))

        d.rectangle([left, top, right, bot], outline=(200, 200, 200))
        d.line([xy(tt, vv) for tt, vv in zip(t[::2], a[::2])], fill=(120, 120, 120), width=1)
        d.line([xy(tt, vv) for tt, vv in zip(t[::2], b[::2])], fill=(30, 90, 200), width=1)
        d.text((left + 4, top + 2), f"{JOINTS[j]} (grey: logged pos, blue: E3 sim)", fill="black")
    d.text((80, 12), title, fill="black")
    im.save(path)


def main():
    os.makedirs(f"{OUT}/starts", exist_ok=True)
    os.makedirs(f"{OUT}/loeo", exist_ok=True)
    eps = T.load_episodes()
    print(f"episodes {len(eps)} test {TEST} out {OUT} utc {time.strftime('%FT%TZ', time.gmtime())}", flush=True)

    info, kp_init = stage0(eps)
    p0 = T.INIT.copy()
    p0[:, 4] = FR
    for j, kp in kp_init.items():
        p0[j, 0] = float(np.clip(kp, LO5[j, 0], HI5[j, 0]))
    p0[:, FREE5] = np.clip(p0[:, FREE5], LO5, HI5)
    json.dump({"windows": info, "kp_init_lift_elbow": {JOINTS[j]: v for j, v in kp_init.items()},
               "p0_6x6": p0.tolist()}, open(f"{OUT}/e3_stage0.json", "w"), indent=1)
    print("stage0 kp init", {JOINTS[j]: round(v, 3) for j, v in kp_init.items()}, flush=True)
    z0 = pack(p0, 1.0)

    n_starts = 2 if TEST else 8
    nfev = 3 if TEST else 500
    workers = 2 if TEST else 4
    rng = np.random.default_rng(20261114)
    sd = np.concatenate([np.full(30, 0.25), [0.1]])
    zs = [z0] + [clip_z(z0 + rng.normal(0.0, 1.0, 31) * sd) for _ in range(n_starts - 1)]
    jobs = [(i, zs[i], eps, nfev, f"{OUT}/starts/start_{i:02d}.json") for i in range(n_starts)]
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        starts = list(ex.map(start_job, jobs))
    for r in starts:
        print("start", r.get("idx"), r.get("status"), r.get("cost"), r.get("nfev"), flush=True)
    ok = [r for r in starts if "cost" in r]
    if not ok:
        raise RuntimeError("all starts failed")
    best = min(ok, key=lambda r: r["cost"])
    z_best = pack(np.array(best["p_6x6"]), best["s"])
    p_best, s_best = unpack(z_best)
    sq_full, cnt_full = rmse_by_joint(eps, simulate(eps, p_best, s_best, mjro.Rollout(nthread=workers), workers))
    json.dump({"best_start": best["idx"], "cost": best["cost"], "s": s_best, "p_6x6": p_best.tolist(),
               "insample_rmse_norm": np.sqrt(sq_full / cnt_full).tolist(),
               "starts": [{k: r.get(k) for k in ("idx", "status", "cost", "nfev", "s", "seconds")} for r in starts]},
              open(f"{OUT}/e3_full.json", "w"), indent=1)
    print("full best start", best["idx"], "s", round(s_best, 4), flush=True)

    folds = [0, 2] if TEST else list(range(len(eps)))
    fjobs = [(i, eps, z_best, 3 if TEST else 40, f"{OUT}/loeo/fold_{i:02d}.json") for i in folds]
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        fold_recs = list(ex.map(loeo_job, fjobs))
    fold_ok = [r for r in fold_recs if "sq" in r]
    print("folds done", len(fold_ok), "of", len(folds), flush=True)

    sq = sum(np.array(r["sq"]) for r in fold_ok)
    cnt = sum(np.array(r["cnt"]) for r in fold_ok)
    lag = sum(np.array(r["lag_sq"]) for r in fold_ok)
    rmse = np.sqrt(sq / cnt)
    lag_best = SHIFTS[np.argmin(lag, axis=0)]
    e1_sq = np.zeros(6)
    e1_cnt = np.zeros(6)
    for i in folds:
        f1 = json.load(open(f"{E1_LOEO}/fold_{i:02d}.json"))
        e1_sq += np.array(f1["sq"])
        e1_cnt += np.array(f1["cnt"])
    e1_rmse = np.sqrt(e1_sq / e1_cnt)
    p_list = np.array(best["p_6x6"])
    kv = p_best[:, 1]
    sanity = {
        "kv_above_stall_over_noload_limit": [JOINTS[j] for j in range(6) if kv[j] > KV_STALL_LIMIT],
        "explicit_step_risk_kp_dt2_over_arm_gt1": [JOINTS[j] for j in range(6)
                                                   if p_best[j, 0] * DT_SQ() / p_best[j, 3] > 1.0],
        "params_at_bounds": [],
        "gravity_bias_above_forcerange": [JOINTS[j] for j in range(6)
                                          if info[JOINTS[j]].get("max_abs_gravity_Nm", 0.0) > FR],
    }
    for j in range(6):
        for c, name in enumerate(["kp", "kv", "frictionloss", "armature", "delay"]):
            v = np.log(p_best[j, FREE5[c]])
            if v <= np.log(LO5[j, c]) + 0.01 or v >= np.log(HI5[j, c]) - 0.01:
                sanity["params_at_bounds"].append(f"{JOINTS[j]}.{name}")
    gate = {}
    for j in range(6):
        gate[JOINTS[j]] = {
            "loeo_rmse_norm": float(rmse[j]), "e1_rmse_norm_same_folds": float(e1_rmse[j]),
            "delta_vs_e1": float(rmse[j] - e1_rmse[j]), "pass_rmse": bool(rmse[j] <= 0.5),
            "lag_best_s": float(lag_best[j]), "lag_frames": float(lag_best[j] / FRAME),
            "pass_lag": bool(abs(lag_best[j]) <= FRAME + 1e-9)}
    g2 = all(gate[n]["pass_rmse"] and gate[n]["pass_lag"] for n in JOINTS)
    result = {"folds": folds, "n_folds_ok": len(fold_ok), "test_run": TEST, "G2_pass": bool(g2),
              "per_joint": gate, "s_full": s_best, "p_full_6x6": p_best.tolist(), "sanity": sanity,
              "fold_init": {str(r["fold"]): r.get("init") for r in fold_ok},
              "shift_convention": "sim delayed by shift (s): compare q_sim(t - shift) with logged pos(t)",
              "not_fitted": ["deadband", "per-link COM scale (only global mass/inertia scale s)"]}
    json.dump(result, open(f"{OUT}/e3_result.json", "w"), indent=1)
    print("G2", g2, flush=True)
    for n in JOINTS:
        print(n, {k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in gate[n].items()}, flush=True)

    bar_png(f"{OUT}/e3_bar.png", rmse.tolist(), e1_rmse.tolist(), 0.5,
            "P3-TWIN-DYN E3 grey-box LOEO RMSE vs E1")
    ov = next((r for r in fold_ok if r["fold"] == 2), fold_ok[0])
    overlay_png(f"{OUT}/e3_fold2_overlay.png", ov["t"], ov["meas_norm"], ov["sim_norm"],
                f"E3 LOEO fold {ov['fold']} ({ov['episode']})")
    with open(f"{OUT}/e3_done.txt", "w") as fh:
        fh.write(f"done {time.strftime('%FT%TZ', time.gmtime())} G2_pass={g2}\n")
    print("done", flush=True)


def DT_SQ():
    return T.DT * T.DT


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        with open(f"{OUT}/e3_failed.txt", "w") as fh:
            fh.write(repr(ex) + "\n")
        raise
