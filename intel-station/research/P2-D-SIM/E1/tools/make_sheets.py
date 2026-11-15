#!/usr/bin/env python3
"""Joint-range table, joint histograms and top-view contact sheet: simulation vs real (read-only).
Run: venv/bin/python -I -B make_sheets.py
Sim actions are absolute radians (converted to degrees for display); real actions are LeRobot
normalized values (no unit mapping is applied; shape-only overlay is scaled per joint)."""
import csv
import glob
import os
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

SIM = "/home/intel/intel-junction-2026/data/sim/2026-11-14/extract-lerobot/dataset-150"
RAW = "/home/intel/intel-junction-2026/data/sim/2026-11-14/dataset-150-raw"
REAL = "/home/intel/intel-junction-2026/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af"
OUT = "/home/intel/intel-junction-2026/work/P2-D-SIM/E1/out"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
REAL_TOP_KEY = "observation.images.realsense 243622061421"  # top view per FACTS (OPUS-verified 13:23Z)
os.makedirs(OUT, exist_ok=True)


def actions(root):
    out = []
    for f in sorted(glob.glob(os.path.join(root, "data", "chunk-*", "file-*.parquet"))):
        out.append(np.array(pq.read_table(f, columns=["action"]).column("action").to_pylist(), dtype=np.float64))
    return np.concatenate(out)


sim_rad = actions(SIM)
sim_deg = np.degrees(sim_rad)
real = actions(REAL)
assert sim_rad.shape[1] == 6 and real.shape[1] == 6

rows = []
for j, name in enumerate(JOINTS):
    def q(a):
        return [round(float(np.percentile(a[:, j], p)), 3) for p in (0, 1, 50, 99, 100)]
    rows.append({"joint": name, "sim_rad_min_p1_p50_p99_max": q(sim_rad), "sim_deg_min_p1_p50_p99_max": q(sim_deg),
                 "real_norm_min_p1_p50_p99_max": q(real)})
with open(os.path.join(OUT, "joint_ranges_sim_vs_real.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["joint", "sim_rad min,p1,p50,p99,max", "sim_deg min,p1,p50,p99,max", "real_norm min,p1,p50,p99,max"])
    for r in rows:
        w.writerow([r["joint"], r["sim_rad_min_p1_p50_p99_max"], r["sim_deg_min_p1_p50_p99_max"],
                    r["real_norm_min_p1_p50_p99_max"]])
print("RANGES")
for r in rows:
    print(r["joint"], "sim_deg", r["sim_deg_min_p1_p50_p99_max"], "real_norm", r["real_norm_min_p1_p50_p99_max"])
print("counts sim", sim_rad.shape[0], "real", real.shape[0])

# histograms: row 1 sim (deg), row 2 real (normalized)
fig, axes = plt.subplots(2, 6, figsize=(18, 6))
for j, name in enumerate(JOINTS):
    axes[0, j].hist(sim_deg[:, j], bins=60, color="#4477aa")
    axes[0, j].set_title(f"sim {name} (deg)", fontsize=9)
    axes[1, j].hist(real[:, j], bins=60, color="#aa4444")
    axes[1, j].set_title(f"real {name} (norm)", fontsize=9)
fig.suptitle("Action histograms: sim (absolute radians shown in degrees) vs real (LeRobot normalized); no unit mapping")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "joint_hist_sim_vs_real.png"), dpi=80)
plt.close(fig)

# shape-only overlay: each joint scaled by its own p1..p99
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for j, name in enumerate(JOINTS):
    ax = axes.flat[j]
    for arr, lab, col in ((sim_deg, "sim deg", "#4477aa"), (real, "real norm", "#aa4444")):
        lo, hi = np.percentile(arr[:, j], 1), np.percentile(arr[:, j], 99)
        ax.hist((arr[:, j] - lo) / (hi - lo + 1e-9), bins=60, range=(-0.5, 1.5), alpha=0.5, density=True,
                color=col, label=lab)
    ax.set_title(name, fontsize=9)
    ax.legend(fontsize=7)
fig.suptitle("Shape only: each series scaled to its own p1..p99 (not a unit mapping)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "joint_shape_overlay.png"), dpi=80)
plt.close(fig)

# contact sheet: sim overhead frames vs real top view (3 times each)
os.makedirs(os.path.join(OUT, "frames"), exist_ok=True)


def grab(src, t, dst):
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", src, "-frames:v", "1", dst],
                       capture_output=True, text=True)
    return r.returncode == 0 and os.path.exists(dst)


times = [0.5, 4.0, 8.0]
sim_src = os.path.join(RAW, "attempt-0000", "overhead.mp4")
real_src = glob.glob(os.path.join(REAL, "videos", REAL_TOP_KEY, "chunk-000", "file-000.mp4"))[0]
tiles = []
status = []
for label, src in (("sim_overhead_attempt0000", sim_src), ("real_top_realsense_243622061421", real_src)):
    row = []
    for t in times:
        dst = os.path.join(OUT, "frames", f"{label}_t{t:.1f}.png")
        ok = grab(src, t, dst)
        status.append((label, t, ok))
        row.append(Image.open(dst).convert("RGB").resize((320, 240)) if ok else Image.new("RGB", (320, 240), "gray"))
    tiles.append(row)
W, H = 320, 240
sheet = Image.new("RGB", (W * 3, (H + 22) * 2), "white")
for r, row in enumerate(tiles):
    for c, im in enumerate(row):
        sheet.paste(im, (c * W, r * (H + 22) + 22))
        d = ImageDraw.Draw(sheet)
        lab = ("SIM overhead (attempt-0000)" if r == 0 else "REAL top view (243622061421)") + f"  t={times[c]}s"
        d.text((c * W + 4, r * (H + 22) + 4), lab, fill="black")
sheet.save(os.path.join(OUT, "contact_sheet_sim_vs_real_top.png"))
print("FRAMES", status)
print("SHEET", os.path.join(OUT, "contact_sheet_sim_vs_real_top.png"))
