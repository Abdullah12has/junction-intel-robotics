#!/usr/bin/env python3
"""P3-TWIN-DYN E1: diagnostics of the 17 joints.csv logs (sag vs gravity, lag, ramp speeds, gripper readings)."""
import csv
import glob
import json
import math
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plotlib  # noqa: E402

R = "/home/intel/intel-junction-2026"
OUT = f"{R}/work/P3-TWIN-DYN/E1"
URDF = f"{OUT}/urdf/so101_nogeom.urdf"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TICKS_PER_REV = 4095.0


def calib_k():
    entries = json.load(open(f"{R}/app/motion/project_robots.json"))
    e = next(x for x in entries if x["type"] == "SO101_Follower")
    cal = e["payload"]["calibration"]
    k = []
    for name in JOINTS:
        lo, hi = float(cal[name]["range_min"]), float(cal[name]["range_max"])
        span = 100.0 if name == "gripper" else 200.0
        k.append((hi - lo) / span * 2.0 * math.pi / TICKS_PER_REV)
    return np.array(k)


def load_episodes():
    out = []
    for f in sorted(glob.glob(f"{R}/logs/sort/any_17*/ep_*/joints.csv")):
        with open(f, newline="") as fh:
            rd = csv.reader(fh)
            header = next(rd)
            rows = [r for r in rd if r and len(r) == len(header)]
        a = np.array(rows, dtype=float)
        out.append({"name": "/".join(f.split("/")[-3:-1]), "t": a[:, 0], "goal": a[:, 1:7], "pos": a[:, 7:13]})
    return out


def stationary_mask(goal, n=30):
    same = np.all(np.abs(np.diff(goal, axis=0)) < 1e-9, axis=1)
    same = np.concatenate([[False], same])
    cs = np.concatenate([[0], np.cumsum(same.astype(int))])
    m = np.zeros(len(goal), dtype=bool)
    for i in range(n - 1, len(goal)):
        m[i] = (cs[i + 1] - cs[i + 1 - n]) == n
    return m


def zoh_goal(t, goal, query):
    idx = np.searchsorted(t, query, side="right") - 1
    return goal[np.clip(idx, 0, None)]


def main():
    k = calib_k()
    print("k rad per unit", np.round(k, 6).tolist())
    eps = load_episodes()
    model = mujoco.MjModel.from_xml_path(URDF)
    data = mujoco.MjData(model)

    # 1) sag on stationary rows (goal constant for 1 s) vs gravity torque at the measured pose
    xs, ys, names = [], [], []
    for ep in eps:
        m = stationary_mask(ep["goal"])
        for i in np.where(m)[0][::15]:
            q = ep["pos"][i] * k
            data.qpos[:] = q
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            xs.append(data.qfrc_bias[:5].copy())
            ys.append(((ep["goal"][i] - ep["pos"][i]) * k)[:5])
    X = np.array(xs)
    Y = np.array(ys)
    print("stationary samples", len(X))
    for j in range(5):
        A = np.vstack([X[:, j], np.ones(len(X))]).T
        coef, *_ = np.linalg.lstsq(A, Y[:, j], rcond=None)
        pred = A @ coef
        ss_res = float(np.sum((Y[:, j] - pred) ** 2))
        ss_tot = float(np.sum((Y[:, j] - Y[:, j].mean()) ** 2)) + 1e-30
        kp = 1.0 / coef[0] if coef[0] > 0 else float("nan")
        print(f"sag {JOINTS[j]}: slope {coef[0]:.4f} rad/(N.m) -> kp~{kp:.3f} N.m/rad, intercept {coef[1]:.5f} rad, "
              f"R2 {1 - ss_res / ss_tot:.3f}, grav range {X[:, j].min():.3f}..{X[:, j].max():.3f} N.m, "
              f"sag range {Y[:, j].min():.4f}..{Y[:, j].max():.4f} rad")

    # 2) lag: RMSE of pos(t) against goal_zoh(t - delta) on moving rows, pooled over episodes
    deltas = np.arange(0.0, 0.2001, 0.005)
    sse = np.zeros((len(deltas), 6))
    cnt = 0
    for ep in eps:
        mv = ~stationary_mask(ep["goal"])
        t = ep["t"]
        for di, dlt in enumerate(deltas):
            g = zoh_goal(t, ep["goal"], t - dlt)
            sse[di] += np.sum((g[mv] - ep["pos"][mv]) ** 2, axis=0)
        cnt += int(mv.sum())
    rmse = np.sqrt(sse / max(cnt, 1))
    for j in range(6):
        best = int(np.argmin(rmse[:, j]))
        print(f"lag {JOINTS[j]}: best delay {deltas[best]*1000:.0f} ms, rmse {rmse[best, j]:.3f} units "
              f"(zero delay {rmse[0, j]:.3f}), moving rows {cnt}")

    # 3) ramp speeds and accelerations achieved on the goal (from survey.json)
    sv = json.load(open(f"{OUT}/survey.json"))
    vmax = np.max([v["vmax_goal_units_s"] for v in sv.values()], axis=0)
    amax = np.max([v["amax_goal_units_s2"] for v in sv.values()], axis=0)
    print("max goal speed units/s", np.round(vmax, 2).tolist())
    print("max goal accel units/s2", np.round(amax, 1).tolist())

    # 4) gripper readings while stationary
    closed, opened = [], []
    for ep in eps:
        m = stationary_mask(ep["goal"])
        closed += ep["pos"][m & (ep["goal"][:, 5] <= 1.0), 5].tolist()
        opened += ep["pos"][m & (ep["goal"][:, 5] >= 5.0), 5].tolist()
    if closed:
        c = np.array(closed)
        print("gripper stationary goal<=1: n", len(c), "pos5 pct 5/25/50/75/95",
              np.round(np.percentile(c, [5, 25, 50, 75, 95]), 2).tolist())
        vals, cts = np.unique(np.round(c, 1), return_counts=True)
        top = np.argsort(-cts)[:8]
        print("gripper pos5 most common (rounded 0.1):", [(float(vals[i]), int(cts[i])) for i in top])
    if opened:
        o = np.array(opened)
        print("gripper stationary goal>=5: n", len(o), "pos5 pct 5/50/95", np.round(np.percentile(o, [5, 50, 95]), 2).tolist())

    # 5) overlay plot of one long episode (goal dashed, pos solid)
    ep = [e for e in eps if e["name"] == "any_1740/ep_0013"][0]
    img, draw = plotlib.new_canvas(1000, 1200)
    draw.text((20, 8), "any_1740/ep_0013 joints.csv: logged goal (orange) and present (blue), normalized units",
              fill=(0, 0, 0), font=plotlib.FONT)
    t0 = ep["t"] - ep["t"][0]
    for j in range(6):
        box = (90, 40 + j * 190, 900, 40 + j * 190 + 150)
        g = ep["goal"][:, j]
        p = ep["pos"][:, j]
        lo = min(g.min(), p.min())
        hi = max(g.max(), p.max())
        pad = 0.05 * (hi - lo + 1e-6)
        pn = plotlib.Panel(draw, box, (0, t0[-1]), (lo - pad, hi + pad), title=JOINTS[j], xlabel="time s")
        pn.line(t0, g, "o", 1)
        pn.line(t0, p, "b", 1)
    img.save(f"{OUT}/diag_any_1740_ep_0013.png")
    print("wrote", f"{OUT}/diag_any_1740_ep_0013.png")


if __name__ == "__main__":
    main()
