#!/usr/bin/env python3
"""P2-D-SIM2 W3: read-only evaluation of the team's MuJoCo episodes against real Studio data.
Reads only: data/sim/2026-11-14/dataset-150-raw, data/studio-datasets/6d6bdce3..., app/motion/project_robots.json.
Writes only: work/P2-D-SIM2/W3/numbers_sim.json."""
import glob
import json
import math

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = "/home/intel/intel-junction-2026"
SIM = f"{ROOT}/data/sim/2026-11-14/dataset-150-raw"
REAL = f"{ROOT}/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af"
OUT = f"{ROOT}/work/P2-D-SIM2/W3/numbers_sim.json"
K = 2 * math.pi / 4096.0          # Feetech STS3215: 4096 ticks per revolution (theory)
R2D = 180.0 / math.pi
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CAL = [r for r in json.load(open(f"{ROOT}/app/motion/project_robots.json"))
       if r["type"] == "SO101_Follower"][0]["payload"]["calibration"]
MN = {j: CAL[j]["range_min"] for j in J}
MX = {j: CAL[j]["range_max"] for j in J}
SPAN = {j: MX[j] - MN[j] for j in J}
RANGE = {j: (100.0 if j == "gripper" else 200.0) for j in J}
UPR = {j: RANGE[j] / (K * SPAN[j]) for j in J}      # normalized units per radian (drive_mode 0)
UPR_A = np.array([UPR[j] for j in J])


def tick2norm(j, t):
    v = RANGE[j] * (t - MN[j]) / SPAN[j]
    return v if j == "gripper" else v - 100.0


N0 = {j: tick2norm(j, 2047.5) for j in J}            # zero hypothesis: mid-range tick
N0_A = np.array([N0[j] for j in J])
# Real reference poses copied from shared/plan-s/calib.json (Opus-verified, normalized units)
BLUE_REAL = np.array([32.1, 5.47, 25.59, 61.85, -36.78])     # ep7
GREEN_REAL = np.array([-54.67, 24.51, 1.91, 54.22, -52.63])  # median of ep8-10
HOLD_REAL = np.array([-15.21, 1.59, 41.11, 55.15, -50.9])    # grasp end, n=11 median
HOME_REAL = np.array([-5.99, -98.32, 95.37, 60.32, -48.81])  # home, n=11 median
G_CLOSED_BRICK = 2.221
G_RELEASE_OPEN = 18.75
PLATE_XY = {"blue": np.array([-0.105, -0.13]), "green": np.array([0.235, -0.16])}  # station_scene.BOWL_CENTERS
SIM_HOME = np.array([0, -0.5, 0.8, 0.6, 0, 0.5])            # sim reset pose (model-only)


def stats(x):
    x = np.asarray(x, float).ravel()
    return {"p5": round(float(np.percentile(x, 5)), 3), "p50": round(float(np.percentile(x, 50)), 3),
            "p95": round(float(np.percentile(x, 95)), 3), "min": round(float(x.min()), 3),
            "max": round(float(x.max()), 3)}


def r3(a):
    return [round(float(v), 3) for v in np.ravel(a)]


out = {"theory": {"units_per_rad": {j: round(UPR[j], 3) for j in J},
                  "deg_per_unit": {j: round(1 / UPR[j] * R2D, 4) for j in J},
                  "N0_zero_tick_2047.5": {j: round(N0[j], 3) for j in J},
                  "span_ticks": SPAN}}

# ---------------- simulation episodes ----------------
summ = json.load(open(f"{SIM}/summary.json"))
rows, A_all, v_sim, a3_sim, ep_len = [], [], [], [], []
rel = {"blue": [], "green": []}
grasp_pose, g_grasp, g_rel = [], [], []
for e in summ:
    d = f"{SIM}/attempt-{e['attempt']:04d}"
    z = np.load(f"{d}/trajectory.npz")
    A = z["action"].astype(float)
    S = z["state"].astype(float)
    tr = json.load(open(f"{d}/trace.json"))
    ph = np.array([t["phase"] for t in tr])
    assert len(ph) == len(A) == len(S), d
    A_all.append(A)
    ep_len.append(len(A))
    v_sim.append(np.diff(A, axis=0) * 30.0 * R2D)
    a3_sim.append(np.diff(A, n=3, axis=0) * 30.0 ** 3 * R2D)
    if (ph == "release").any():
        rel[e["destination"]].append(np.median(A[ph == "release"], axis=0))
        g_rel.append(np.median(S[ph == "release", 5]))
    if (ph == "grasp").any():
        grasp_pose.append(np.median(A[ph == "grasp"], axis=0))
        g_grasp.append(np.median(S[ph == "grasp", 5]))
    fx = np.array(e["final_object_xyz"][:2])
    dd = {p: float(np.linalg.norm(fx - PLATE_XY[p])) for p in PLATE_XY}
    rows.append({"colour": e["target_color"], "dest": e["destination"], "brick": e["brick_type"],
                 "accepted": e["accepted"], "success": e["success"], "grasp_ok": e["physical_grasp_verified"],
                 "err": e["error"], "frames": e["frames"], "d_dest": dd[e["destination"]],
                 "nearest": min(dd, key=dd.get)})

out["sim_counts"] = {
    "episodes": len(rows), "accepted": sum(r["accepted"] for r in rows), "success": sum(r["success"] for r in rows),
    "grasp_verified": sum(r["grasp_ok"] for r in rows), "errors_nonnull": sum(r["err"] is not None for r in rows),
    "colour": {c: sum(r["colour"] == c for r in rows) for c in sorted({r["colour"] for r in rows})},
    "dest": {c: sum(r["dest"] == c for r in rows) for c in ("blue", "green")},
    "brick": {c: sum(r["brick"] == c for r in rows) for c in ("2x2", "2x4")},
    "duration_s": stats(np.array(ep_len) / 30.0),
    "frames_total": int(sum(ep_len))}
xt = {f"{b}->{d}": sum(1 for r in rows if r["brick"] == b and r["dest"] == d) for b in ("2x2", "2x4") for d in ("blue", "green")}
out["sim_routing"] = {"crosstab": xt,
                      "OD10_contradictions(2x2->green + 2x4->blue)": xt["2x2->green"] + xt["2x4->blue"],
                      "relabel_check": {"nearest_plate_equals_dest": sum(r["nearest"] == r["dest"] for r in rows),
                                        "d_dest_to_plate_m": stats([r["d_dest"] for r in rows])}}

A_sim = np.vstack(A_all)
out["sim_positions_units"] = {}
for name, sgn in (("s+1", 1), ("s-1", -1)):
    U = N0_A + sgn * UPR_A * A_sim
    out["sim_positions_units"][name] = {j: stats(U[:, i]) for i, j in enumerate(J)}
out["sim_positions_rad"] = {j: stats(A_sim[:, i]) for i, j in enumerate(J)}
out["sim_home_reset_units"] = {"s+1": r3(N0_A + UPR_A * SIM_HOME), "s-1": r3(N0_A - UPR_A * SIM_HOME)}

# plate drop poses: sim expert release phase vs real release (calib.json)
blue_s = np.median(np.array(rel["blue"]), axis=0)
green_s = np.median(np.array(rel["green"]), axis=0)
out["plates"] = {"n_blue": len(rel["blue"]), "n_green": len(rel["green"]),
                 "sim_blue_rad": r3(blue_s), "sim_green_rad": r3(green_s),
                 "sim_pan_blue_minus_green_units": round(float(UPR["shoulder_pan"] * (blue_s[0] - green_s[0])), 2),
                 "real_pan_blue_minus_green_units": round(float(BLUE_REAL[0] - GREEN_REAL[0]), 2)}
for sgn in (1, -1):
    sb = N0_A[:5] + sgn * UPR_A[:5] * blue_s[:5]
    sg = N0_A[:5] + sgn * UPR_A[:5] * green_s[:5]
    rb, rg = BLUE_REAL - sb, GREEN_REAL - sg
    fit_n0 = ((BLUE_REAL - sgn * UPR_A[:5] * blue_s[:5]) + (GREEN_REAL - sgn * UPR_A[:5] * green_s[:5])) / 2
    out["plates"][f"sign{sgn:+d}"] = {
        "sim_blue_units": r3(sb), "real_blue": r3(BLUE_REAL), "resid_blue(real-sim)": r3(rb),
        "sim_green_units": r3(sg), "real_green": r3(GREEN_REAL), "resid_green(real-sim)": r3(rg),
        "rms_resid_all_units": round(float(np.sqrt(np.mean(np.r_[rb, rg] ** 2))), 2),
        "fitted_zero_units": r3(fit_n0), "theory_zero_units": r3(N0_A[:5]),
        "fitted_minus_theory_units": r3(fit_n0 - N0_A[:5])}

# grasp (hold) pose and home reference, descriptive only (brick cells differ from real)
gp = np.median(np.array(grasp_pose), axis=0)
out["grasp_pose"] = {"sim_rad": r3(gp), "sim_units_s+1": r3(N0_A[:5] + UPR_A[:5] * gp[:5]),
                     "sim_units_s-1": r3(N0_A[:5] - UPR_A[:5] * gp[:5]), "real_hold_median": r3(HOLD_REAL)}
out["home_reference"] = {"real_home_median": r3(HOME_REAL),
                         "sim_reset_home_units_s+1": out["sim_home_reset_units"]["s+1"][:5],
                         "sim_reset_home_units_s-1": out["sim_home_reset_units"]["s-1"][:5]}

# gripper: sim joint state vs real gripper.pos (0-100 normalized)
gg = float(np.median(g_grasp))
gr = float(np.median(g_rel))
gn = lambda rad: float(N0["gripper"] + UPR["gripper"] * rad)  # noqa: E731
slope_fit = (G_RELEASE_OPEN - G_CLOSED_BRICK) / (gr - gg)
out["gripper"] = {"sim_state_grasp_rad": round(gg, 4), "sim_state_release_rad": round(gr, 4),
                  "sim_grasp_units_theory": round(gn(gg), 3), "real_closed_on_brick_p50": G_CLOSED_BRICK,
                  "sim_release_units_theory": round(gn(gr), 3), "real_release_open_p50": G_RELEASE_OPEN,
                  "two_point_slope_units_per_rad": round(slope_fit, 3),
                  "theory_slope_units_per_rad": round(UPR["gripper"], 3)}

# ---------------- real dataset (6d6bdce3) ----------------
files = sorted(glob.glob(f"{REAL}/data/chunk-000/*.parquet"))
dfr = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
A_r = np.stack(dfr["action"].to_numpy()).astype(float)
ep_r = dfr["episode_index"].to_numpy()
v_real, a3_real, lens_r = [], [], []
for e in np.unique(ep_r):
    Ae = A_r[ep_r == e]
    lens_r.append(len(Ae))
    v_real.append(np.diff(Ae, axis=0) * 30.0 / UPR_A * R2D)
    a3_real.append(np.diff(Ae, n=3, axis=0) * 30.0 ** 3 / UPR_A * R2D)
out["real_positions_units"] = {j: stats(A_r[:, i]) for i, j in enumerate(J)}
out["real_episodes"] = {"n": len(lens_r), "frames": int(sum(lens_r)), "duration_s": stats(np.array(lens_r) / 30.0)}

# speed and smoothness: degrees/s and RMS jerk (deg/s^3), per joint, pooled
V_s, V_r = np.vstack(v_sim), np.vstack(v_real)
X_s, X_r = np.vstack(a3_sim), np.vstack(a3_real)
sp = {}
for i, j in enumerate(J):
    sp[j] = {"sim_abs_vel_deg_s_p50": round(float(np.percentile(np.abs(V_s[:, i]), 50)), 2),
             "real_abs_vel_deg_s_p50": round(float(np.percentile(np.abs(V_r[:, i]), 50)), 2),
             "sim_abs_vel_deg_s_p95": round(float(np.percentile(np.abs(V_s[:, i]), 95)), 2),
             "real_abs_vel_deg_s_p95": round(float(np.percentile(np.abs(V_r[:, i]), 95)), 2),
             "sim_frac_still_lt1dps": round(float(np.mean(np.abs(V_s[:, i]) < 1)), 3),
             "real_frac_still_lt1dps": round(float(np.mean(np.abs(V_r[:, i]) < 1)), 3),
             "sim_jerk_rms_degs3": round(float(np.sqrt(np.mean(X_s[:, i] ** 2))), 1),
             "real_jerk_rms_degs3": round(float(np.sqrt(np.mean(X_r[:, i] ** 2))), 1)}
out["speed_jerk"] = sp
json.dump(out, open(OUT, "w"), indent=1)
print("WROTE", OUT)
print(json.dumps({k: out[k] for k in ("theory", "sim_counts", "sim_routing", "plates", "gripper")}))
