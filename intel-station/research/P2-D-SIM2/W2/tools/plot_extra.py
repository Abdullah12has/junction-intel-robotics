"""Plots and extra counts for W2 (sim vs real). Reads out/rows.json and out/mapping_validation.json.

Writes plots/mapping_place_and_speed.png and out/extra_counts.json.
"""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

W = "/home/intel/intel-junction-2026/work/P2-D-SIM2/W2"
OUT = W + "/out"
D = json.load(open(OUT + "/rows.json"))
MV = json.load(open(OUT + "/mapping_validation.json"))
sim, real, K = D["sim"], [r for r in D["real"] if r["set"] == "raw6d6"], D["K"]
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
BLUE = {7, 11, 12}
GREEN = {9, 10, 14, 15}

# extra counts from raw npz (wrist_flex saturation near URDF upper limit 1.65806 rad)
SIM = "/home/intel/intel-junction-2026/data/sim/2026-11-14/dataset-150-raw"
frac_sat, frac_lim, total = 0, 0, 0
for f in sorted(glob.glob(SIM + "/attempt-*/trajectory.npz")):
    S = np.load(f)["state"].astype(float)
    frac_sat += int((S[:, 3] >= 1.65).sum())
    frac_lim += int((S[:, 3] >= 1.6579).sum())
    total += len(S)
extra = {"sim_frames_wrist_flex_ge_1.65rad": frac_sat, "sim_frames_total": total,
         "sim_frac_wrist_flex_ge_1.65rad": round(frac_sat / total, 4),
         "sim_frames_wrist_flex_ge_1.6579rad_urdf_limit": frac_lim,
         "sim_frac_wrist_flex_ge_1.6579rad_urdf_limit": round(frac_lim / total, 4)}
rfl = [r for r in real if r["set"] == "raw6d6"]
real_wf = np.concatenate([np.asarray(json.load(open(OUT + "/rows.json"))["real"][0]["home_n"])[:0]] + [np.asarray([r["home_n"][3]]) for r in rfl])
extra["real_eps_home_wrist_flex_n_min_max"] = [round(float(min(r["home_n"][3] for r in rfl)), 2), round(float(max(r["home_n"][3] for r in rfl)), 2)]
extra["real_grip_max_n_median_over_eps"] = round(float(np.median([r["grip_max"] for r in rfl])), 2)
json.dump(extra, open(OUT + "/extra_counts.json", "w"), indent=1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
x = np.arange(5)
for k, (dest, eps, colr) in enumerate((("blue", BLUE, "#1f77b4"), ("green", GREEN, "#2ca02c"))):
    med = np.asarray(MV[f"place_sign_1"][dest]["sim_median_n"])
    lo = np.asarray(MV[f"place_sign_1"][dest]["sim_iqr_n"]["p25"])
    hi = np.asarray(MV[f"place_sign_1"][dest]["sim_iqr_n"]["p75"])
    ax.errorbar(x + (k - 0.5) * 0.18, med, yerr=[med - lo, hi - med], fmt="o", color=colr, capsize=3,
                label=f"sim {dest} drops (median, IQR)")
    pts = [r["place_n"][:5] for r in rfl if r["ep"] in eps and r["place_n"] is not None]
    for p in pts:
        ax.scatter(x + (k - 0.5) * 0.18, p, marker="x", color=colr, s=40)
ax.scatter([], [], marker="x", color="k", label="real drops (each episode)")
ax.set_xticks(x)
ax.set_xticklabels(["pan", "lift", "elbow", "wrist_flex", "wrist_roll"])
ax.set_ylabel("Studio normalized units (K * q_sim, sign +1)")
ax.set_title("Plate-drop poses: sim (IQR) vs real episodes")
ax.axhline(0, color="grey", lw=0.5)
ax.legend(fontsize=8)

ax = axes[1]
sv = np.asarray(MV["sim_vmax_n_s_p90_over_eps"])
rv = np.asarray(MV["real_vmax_n_s_p90_over_eps"])
xx = np.arange(6)
ax.bar(xx - 0.2, sv, width=0.4, label="sim p90 over episodes")
ax.bar(xx + 0.2, rv, width=0.4, label="real p90 over episodes (raw set)")
ax.set_xticks(xx)
ax.set_xticklabels(["pan", "lift", "elbow", "wrist_flex", "wrist_roll", "gripper"], fontsize=8)
ax.set_ylabel("max joint speed, normalized units / s")
ax.set_title("Per-episode max joint speed (p90)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(W + "/plots/mapping_place_and_speed.png", dpi=110)
print("PLOT_OK", W + "/plots/mapping_place_and_speed.png")
print("EXTRA", json.dumps(extra))
