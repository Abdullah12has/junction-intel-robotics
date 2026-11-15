#!/usr/bin/env python3
"""P2-D-SIM2 W1 (read-only): per-episode inventory of a real Studio LeRobot v3 dataset (normalized units).

Usage: python3 -I real_inventory.py <dataset dir> <out dir>
Release onset = start of the largest 5-frame gripper rise after the closing minimum.
Physical plate by sign of the release shoulder_pan (OD-10 / Opus 13:52Z: positive pan = BLUE).
"""
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DS = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

info = json.loads((DS / "meta/info.json").read_text())
tasks = pd.read_parquet(DS / "meta/tasks.parquet")
texts = tasks["task"].tolist() if "task" in tasks.columns else tasks.index.tolist()
task_map = dict(zip(tasks["task_index"].tolist(), texts))

files = sorted((DS / "data").rglob("*.parquet"))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def plate_claim(text):
    t = text.lower()
    if "blue" in t or "left" in t:
        return "blue"
    if "green" in t or "right" in t:
        return "green"
    return "none"


rows = []
for ep, g in df.groupby("episode_index"):
    g = g.sort_values("frame_index")
    S = np.stack(g["observation.state"].to_numpy()).astype(float)
    N = len(S)
    grip = S[:, 5]
    t_close = int(np.argmin(grip[: max(2, int(0.7 * N))]))
    after = grip[t_close:]
    rise = after[5:] - after[:-5]
    t_rel = t_close + int(np.argmax(rise))
    v = np.diff(S, axis=0) * 30.0
    text = task_map[int(g["task_index"].iloc[0])]
    rel = S[t_rel]
    row = {
        "episode": int(ep), "task_index": int(g["task_index"].iloc[0]), "task": text, "frames": N,
        "t_close": t_close, "t_release": t_rel, "rise5_units": float(rise.max()),
        "plate_by_pan_sign": "blue" if rel[0] > 0 else "green", "task_claims": plate_claim(text),
    }
    for j, name in enumerate(JOINTS):
        row[f"rel_{name}"] = float(rel[j])
        row[f"grasp_{name}"] = float(S[t_close, j])
        row[f"vmax_{name}"] = float(np.abs(v[:, j]).max())
        row[f"vp99_{name}"] = float(np.percentile(np.abs(v[:, j]), 99))
        row[f"smin_{name}"] = float(S[:, j].min())
        row[f"smax_{name}"] = float(S[:, j].max())
    rows.append(row)

with open(OUT / "real_episodes.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

summary = {
    "dataset": DS.name, "fps": info.get("fps"), "codebase": info.get("codebase_version"),
    "n_episodes": len(rows), "n_rows": int(len(df)), "task_counts": dict(collections.Counter(r["task"] for r in rows)),
    "plate_by_pan_sign": dict(collections.Counter(r["plate_by_pan_sign"] for r in rows)),
    "task_claim_vs_pan_sign": {f"{a}|{b}": sum(1 for r in rows if r["task_claims"] == a and r["plate_by_pan_sign"] == b)
                               for a in ("blue", "green", "none") for b in ("blue", "green")},
    "release_pose_norm_by_plate": {},
    "vmax_norm_per_s": {j: max(r[f"vmax_{j}"] for r in rows) for j in JOINTS},
    "vp99_norm_per_s_median": {j: float(np.median([r[f"vp99_{j}"] for r in rows])) for j in JOINTS},
    "state_min_norm": {j: min(r[f"smin_{j}"] for r in rows) for j in JOINTS},
    "state_max_norm": {j: max(r[f"smax_{j}"] for r in rows) for j in JOINTS},
    "gripper_grasp_norm_median": float(np.median([r["grasp_gripper"] for r in rows])),
}
for plate in ("blue", "green"):
    sub = [r for r in rows if r["plate_by_pan_sign"] == plate]
    if sub:
        summary["release_pose_norm_by_plate"][plate] = {
            "n": len(sub), "episodes": [r["episode"] for r in sub],
            "mean": {j: float(np.mean([r[f"rel_{j}"] for r in sub])) for j in JOINTS},
            "std": {j: float(np.std([r[f"rel_{j}"] for r in sub])) for j in JOINTS},
            "per_episode": {r["episode"]: [round(r[f"rel_{j}"], 2) for j in JOINTS] for r in sub},
        }
(OUT / "real_summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
print(json.dumps(summary, indent=1, default=float))
