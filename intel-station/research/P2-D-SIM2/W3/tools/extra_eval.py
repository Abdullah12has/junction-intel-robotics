#!/usr/bin/env python3
"""P2-D-SIM2 W3: read-only extra checks: gripper open/closed peaks, smoothed acceleration, rest-pose coverage.
Writes work/P2-D-SIM2/W3/numbers_extra.json only."""
import glob
import json
import math

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = "/home/intel/intel-junction-2026"
SIM = f"{ROOT}/data/sim/2026-11-14/dataset-150-raw"
REAL = f"{ROOT}/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af"
OUT = f"{ROOT}/work/P2-D-SIM2/W3/numbers_extra.json"
K = 2 * math.pi / 4096.0
R2D = 180.0 / math.pi
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CAL = [r for r in json.load(open(f"{ROOT}/app/motion/project_robots.json"))
       if r["type"] == "SO101_Follower"][0]["payload"]["calibration"]
UPR = {j: (100.0 if j == "gripper" else 200.0) / (K * (CAL[j]["range_max"] - CAL[j]["range_min"])) for j in J}
UPR_A = np.array([UPR[j] for j in J])
MA = 9  # centred moving-average window (frames) before differentiating


def smooth(x):
    k = np.ones(MA) / MA
    return np.stack([np.convolve(np.pad(x[:, i], (MA // 2, MA // 2), mode="edge"), k, mode="valid")
                     for i in range(x.shape[1])], axis=1)


out = {}
summ = json.load(open(f"{SIM}/summary.json"))
g_peak, g_close, acc_sim, rest_sim, lift_units_sim = [], [], [], [], []
for e in summ:
    d = f"{SIM}/attempt-{e['attempt']:04d}"
    z = np.load(f"{d}/trajectory.npz")
    A = z["action"].astype(float)
    S = z["state"].astype(float)
    ph = np.array([t["phase"] for t in json.load(open(f"{d}/trace.json"))])
    g_peak.append(np.max(S[ph == "release", 5]))
    g_close.append(np.median(S[ph == "grasp", 5]))
    As = smooth(A)
    acc_sim.append(np.diff(As, n=2, axis=0) * 30.0 ** 2 * R2D)
    lift_units_sim.append(A[:, 1])
g_peak_u = float(np.median(g_peak))
g_close_u = float(np.median(g_close))
slope_peak = (18.75 - 2.221) / (g_peak_u - g_close_u)
out["gripper_peaks"] = {"sim_release_open_peak_rad_p50": round(g_peak_u, 4),
                        "sim_grasp_closed_state_rad_p50": round(g_close_u, 4),
                        "sim_open_peak_units_theory": round(float(UPR["gripper"] * g_peak_u + 0.098), 3),
                        "sim_closed_units_theory": round(float(UPR["gripper"] * g_close_u + 0.098), 3),
                        "two_point_slope_units_per_rad": round(slope_peak, 2),
                        "theory_slope_units_per_rad": round(UPR["gripper"], 2),
                        "real_open_release_p50": 18.75, "real_closed_on_brick_p50": 2.221,
                        "real_open_before_grasp_p50": 25.44, "real_closed_empty": 0.98,
                        "sim_closed_command_units_theory": round(float(UPR["gripper"] * -0.08 + 0.098), 3)}

A_sim_u = np.concatenate(lift_units_sim)
sim_lift_u = 0.0 + UPR["shoulder_lift"] * A_sim_u + 8.069
out["sim_lift_units_s+1"] = {"p5": round(float(np.percentile(sim_lift_u, 5)), 2),
                             "p50": round(float(np.percentile(sim_lift_u, 50)), 2),
                             "p95": round(float(np.percentile(sim_lift_u, 95)), 2),
                             "min": round(float(sim_lift_u.min()), 2),
                             "frac_frames_below_-90": round(float(np.mean(sim_lift_u < -90)), 4)}

acc_sim = np.vstack(acc_sim)
dfr = pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(f"{REAL}/data/chunk-000/*.parquet"))],
                ignore_index=True)
A_r = np.stack(dfr["action"].to_numpy()).astype(float)
ep_r = dfr["episode_index"].to_numpy()
acc_real = []
for e in np.unique(ep_r):
    Ae = A_r[ep_r == e]
    As = smooth(Ae)
    acc_real.append(np.diff(As, n=2, axis=0) * 30.0 ** 2 / UPR_A * R2D)
acc_real = np.vstack(acc_real)
sm = {}
for i, j in enumerate(J):
    sm[j] = {"sim_acc_rms_degs2_smoothed": round(float(np.sqrt(np.mean(acc_sim[:, i] ** 2))), 1),
             "real_acc_rms_degs2_smoothed": round(float(np.sqrt(np.mean(acc_real[:, i] ** 2))), 1),
             "sim_acc_p95": round(float(np.percentile(np.abs(acc_sim[:, i]), 95)), 1),
             "real_acc_p95": round(float(np.percentile(np.abs(acc_real[:, i]), 95)), 1)}
out["smoothed_acceleration_deg_s2(MA9)"] = sm
lift_real = A_r[:, 1]
out["real_rest_coverage"] = {"frac_frames_lift_below_-90": round(float(np.mean(lift_real < -90)), 4),
                             "frac_frames_elbow_above_90": round(float(np.mean(A_r[:, 2] > 90)), 4),
                             "n_frames": int(len(A_r))}
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out))
