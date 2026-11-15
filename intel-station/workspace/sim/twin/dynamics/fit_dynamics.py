#!/usr/bin/env python3
"""P3-TWIN-DYN E1 driver.

  selftest            rollout vs plain mj_step loop (exactness), nominal INIT errors
  dtcheck             nominal INIT errors and wall time at DT 0.002 and 0.004
  full [nfev]         fit all 17 episodes from the fixed INIT; writes sim/twin/dynamics params JSON, MJCF, scripts
  fold <e> [nfev]     leave-one-episode-out fold e: fit on the other 16 from the same fixed INIT, score episode e;
                      writes work/P3-TWIN-DYN/E1/loeo/fold_<e>.json (skipped if present)
  aggregate           pools the 17 folds, applies the G2 gate, writes loeo_result.json (report_build.py writes g2_report.json and metrics.json)

G2 gate: per-joint held-out RMSE (normalized units, pooled over held-out rows) <= 0.5 AND held-out lag error <= 1 frame
(33.3 ms). Lag = best time shift of the simulated present against the logged present, pooled over held-out rows.
"""
import datetime
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import twin_lib as T  # noqa: E402

WORK = f"{T.R}/work/P3-TWIN-DYN/E1"
DYN = f"{T.R}/sim/twin/dynamics"
LAG_GRID = np.arange(-0.15, 0.1501, 0.005)
FRAME = 1.0 / 30.0


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg):
    print(now() + " " + msg, flush=True)


def counts_and_sq(eps, states):
    """Per episode: sum of squared normalized errors per joint (masked rows) and masked row counts."""
    errs = T.errors_per_episode(eps, states)
    sq = np.zeros((len(eps), 6))
    cnt = np.zeros((len(eps), 6))
    for e, ep in enumerate(eps):
        for j in range(6):
            m = T.mask_for(ep, j)
            sq[e, j] = np.sum(errs[e][j][m] ** 2)
            cnt[e, j] = m.sum()
    return sq, cnt


def lag_sq(eps, states):
    """Per episode, per shift delta (sim evaluated at t - delta), per joint: sum of squared normalized errors."""
    k = T.calib_k()
    out = np.zeros((len(eps), len(LAG_GRID), 6))
    for e, ep in enumerate(eps):
        st = states[e]
        ts = st[:, 0]
        for di, d in enumerate(LAG_GRID):
            for j in range(6):
                q = np.interp(ep["t"] - d, ts, st[:, 1 + j])
                err = (q - ep["pos"][:, j] * k[j]) / k[j]
                out[e, di, j] = np.sum(err[T.mask_for(ep, j)] ** 2)
    return out


def selftest(eps, ro):
    sub = eps[:3]
    t0 = time.time()
    states = T.simulate(sub, T.INIT.copy(), ro)
    log(f"selftest rollout {len(sub)} eps, lengths {[len(s) for s in states]} wall {time.time() - t0:.2f}s")
    log(f"selftest time column first {np.round(states[0][:3, 0], 6).tolist()} "
        f"expected {[round(T.DT * s, 6) for s in (1, 2, 3)]}")
    ns0 = len(states[0])
    init, ctrl = T.build_inputs(sub[:1], T.INIT.copy(), ns0)
    m = T.mujoco.MjModel.from_xml_string(T.make_xml(T.INIT.copy()))
    d = T.mujoco.MjData(m)
    d.qpos[:] = init[0, 1:7]
    d.qvel[:] = 0.0
    rec = np.zeros((ns0, 6))
    for s in range(ns0):
        d.ctrl[:] = ctrl[0, s]
        T.mujoco.mj_step(m, d)
        rec[s] = d.qpos
    maxdiff = float(np.max(np.abs(rec - states[0][:, 1:7])))
    log(f"selftest rollout vs plain mj_step loop, episode 0 ({ns0} steps): max abs qpos diff {maxdiff:.3e} rad")
    sq, cnt = counts_and_sq(sub, states)
    log("selftest nominal INIT normalized rmse per joint (3 eps) " +
        str(np.round(np.sqrt(sq.sum(0) / cnt.sum(0)), 3).tolist()))


def dtcheck(eps, ro):
    for dt in (0.002, 0.004):
        T.DT = dt
        t0 = time.time()
        states = T.simulate(eps, T.INIT.copy(), ro)
        wall = time.time() - t0
        sq, cnt = counts_and_sq(eps, states)
        log(f"dtcheck DT={dt} nominal INIT rmse {np.round(np.sqrt(sq.sum(0) / cnt.sum(0)), 3).tolist()} "
            f"simulate wall {wall:.2f}s")
    T.DT = 0.004


def run_fit(eps, label, nfev, verbose=0):
    t0 = time.time()
    p, res = T.fit(eps, max_nfev=nfev, verbose=verbose)
    log(f"{label}: episodes {len(eps)} nfev {res.nfev} njev {res.njev} status {res.status} "
        f"cost {res.cost:.6g} wall {time.time() - t0:.1f}s")
    return p, res


def write_full_outputs(p_full, res, eps, insample_rmse):
    os.makedirs(DYN, exist_ok=True)
    k = T.calib_k().tolist()
    params = {}
    for j, name in enumerate(T.JOINTS):
        params[name] = {pn: float(p_full[j, i]) for i, pn in enumerate(T.PARAM_NAMES)}
    doc = {
        "name": "so101_twin_dynamics", "stage": "twin-dyn-v1", "utc": now(),
        "model": "MuJoCo hinge joints (no joint limits) with one <position> actuator per joint; gravity and link "
                 "inertias from so101.urdf (no-geometry copy)",
        "control_law": "force = clip(kp*(ctrl-q) - kv*qdot, -forcerange, forcerange); ctrl = k*goal(t - delay)",
        "timestep_s": T.DT, "integrator": "implicitfast",
        "units": {"kp": "N.m/rad", "kv": "N.m.s/rad", "frictionloss": "N.m", "armature": "kg.m^2",
                  "forcerange": "N.m (symmetric, fixed at 10 = URDF effort limit, not identified)", "delay": "s",
                  "k_rad_per_norm_unit": "rad per normalized unit (width/200 or /100 * 2pi/4095)"},
        "k_rad_per_norm_unit": dict(zip(T.JOINTS, k)),
        "parameters": params,
        "fit": {"data": "logs/sort/any_17*/ep_*/joints.csv (17 files)", "episodes": len(eps),
                "nfev": int(res.nfev), "njev": int(res.njev), "cost": float(res.cost), "status": int(res.status),
                "start": "fixed INIT (tools/twin_lib.py)", "gripper_brick_contact_rows_masked": True},
        "insample_rmse_norm": dict(zip(T.JOINTS, [float(x) for x in insample_rmse])),
        "files": {"model_xml": "so101_dyn_v1.xml", "fit_scripts": ["twin_lib.py", "fit_dynamics.py"]},
    }
    with open(f"{DYN}/twin_dynamics_v1.json", "w") as fh:
        json.dump(doc, fh, indent=1)
    with open(f"{DYN}/so101_dyn_v1.xml", "w") as fh:
        fh.write(T.make_xml(p_full))
    for fn in ("twin_lib.py", "fit_dynamics.py"):
        shutil.copy(f"{HERE}/{fn}", f"{DYN}/{fn}")
    log(f"wrote {DYN}/twin_dynamics_v1.json, so101_dyn_v1.xml and fit scripts")


def do_full(eps, ro, nfev):
    p_full, res = run_fit(eps, "full fit (all 17 episodes)", nfev, verbose=2)
    states = T.simulate(eps, p_full, ro)
    sq, cnt = counts_and_sq(eps, states)
    insample = np.sqrt(sq.sum(0) / cnt.sum(0))
    log("in-sample rmse normalized " + str(np.round(insample, 3).tolist()))
    np.save(f"{WORK}/p_full.npy", p_full)
    with open(f"{WORK}/full_result.json", "w") as fh:
        json.dump({"utc": now(), "nfev": int(res.nfev), "njev": int(res.njev), "cost": float(res.cost),
                   "status": int(res.status), "insample_rmse_norm": insample.tolist(),
                   "insample_rows": cnt.sum(0).tolist(), "params_6x6": p_full.tolist()}, fh, indent=1)
    write_full_outputs(p_full, res, eps, insample)


def do_fold(e, eps, ro, nfev):
    os.makedirs(f"{WORK}/loeo", exist_ok=True)
    path = f"{WORK}/loeo/fold_{e:02d}.json"
    if os.path.exists(path):
        log(f"fold {e} exists, skipped")
        return
    train = [ep for i, ep in enumerate(eps) if i != e]
    held = [eps[e]]
    p_e, res_e = run_fit(train, f"LOEO fold {e} holdout {eps[e]['name']}", nfev)
    states = T.simulate(held, p_e, ro)
    sq, cnt = counts_and_sq(held, states)
    ls = lag_sq(held, states)[0]
    best = [float(LAG_GRID[int(np.argmin(ls[:, j]))]) for j in range(6)]
    rmse_e = np.sqrt(sq[0] / np.maximum(cnt[0], 1))
    rec = {"fold": e, "episode": eps[e]["name"], "rows": int(len(eps[e]["t"])), "utc": now(),
           "sq": sq[0].tolist(), "cnt": cnt[0].tolist(), "lag_sq": ls.tolist(), "lag_best_s": best,
           "rmse_norm": rmse_e.tolist(), "nfev": int(res_e.nfev), "njev": int(res_e.njev),
           "cost": float(res_e.cost), "status": int(res_e.status), "p_6x6": p_e.tolist()}
    with open(path, "w") as fh:
        json.dump(rec, fh)
    log(f"fold {e} rmse {np.round(rmse_e, 3).tolist()} lag_best_s {np.round(best, 3).tolist()}")


def do_aggregate(eps):
    files = sorted(glob.glob(f"{WORK}/loeo/fold_*.json"))
    folds = [json.load(open(f)) for f in files]
    if len(folds) != len(eps):
        log(f"WARNING: {len(folds)} of {len(eps)} folds present")
    sq_tot = np.sum([np.array(f["sq"]) for f in folds], axis=0)
    cnt_tot = np.sum([np.array(f["cnt"]) for f in folds], axis=0)
    lag_tot = np.sum([np.array(f["lag_sq"]) for f in folds], axis=0)
    loeo_rmse = np.sqrt(sq_tot / np.maximum(cnt_tot, 1))
    best_pooled = np.array([LAG_GRID[int(np.argmin(lag_tot[:, j]))] for j in range(6)])
    lag_frames = np.abs(best_pooled) / FRAME
    g2_rmse = bool(np.all(loeo_rmse <= 0.5))
    g2_lag = bool(np.all(lag_frames <= 1.0))
    per_ep = [{"fold": f["fold"], "episode": f["episode"], "rows": f["rows"],
               "rmse_norm": [round(x, 4) for x in f["rmse_norm"]],
               "lag_s_best_per_joint": [round(x, 3) for x in f["lag_best_s"]],
               "nfev": f["nfev"], "njev": f["njev"], "status": f["status"]} for f in folds]
    full = json.load(open(f"{WORK}/full_result.json")) if os.path.exists(f"{WORK}/full_result.json") else None
    res = {"utc": now(), "folds": len(folds), "loeo_rmse_norm": loeo_rmse.tolist(),
           "loeo_rows_scored": cnt_tot.tolist(), "loeo_lag_s_pooled": best_pooled.tolist(),
           "loeo_lag_frames": lag_frames.tolist(), "g2_rmse_pass": g2_rmse, "g2_lag_pass": g2_lag,
           "g2_pass": g2_rmse and g2_lag, "per_episode": per_ep,
           "insample_rmse_norm": full["insample_rmse_norm"] if full else None}
    with open(f"{WORK}/loeo_result.json", "w") as fh:
        json.dump(res, fh, indent=1)
    log("LOEO rmse normalized " + str(np.round(loeo_rmse, 3).tolist()))
    log("LOEO lag frames " + str(np.round(lag_frames, 2).tolist()) + " g2_rmse " + str(g2_rmse) +
        " g2_lag " + str(g2_lag))
    return res


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    eps = T.load_episodes()
    ro = T.mjro.Rollout(nthread=T.NTHREAD)
    log(f"loaded {len(eps)} episodes, {sum(len(e['t']) for e in eps)} rows; mode {mode}; threads {T.NTHREAD}")
    if mode == "selftest":
        selftest(eps, ro)
    elif mode == "dtcheck":
        dtcheck(eps, ro)
    elif mode == "full":
        do_full(eps, ro, int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    elif mode == "fold":
        do_fold(int(sys.argv[2]), eps, ro, int(sys.argv[3]) if len(sys.argv) > 3 else 20)
    elif mode == "aggregate":
        do_aggregate(eps)
    else:
        raise SystemExit(f"unknown mode {mode}")
    log("done")


if __name__ == "__main__":
    main()
