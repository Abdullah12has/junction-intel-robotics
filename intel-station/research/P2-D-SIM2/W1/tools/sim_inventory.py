#!/usr/bin/env python3
"""P2-D-SIM2 W1 (read-only): inventory of the simulated pick-and-place raw attempts.

Usage: python3 -I sim_inventory.py <dataset-150-raw dir> <out dir>
Re-derives the scene draws of BrickSortingEnv.reset with the same numpy call
order (no team code is executed) to recover the brick type of each target.
"""
import collections
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

RAW = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

COLOR_KEYS = ["red", "yellow", "orange", "purple", "white", "cyan"]  # insertion order of sorting_env.COLORS
BLOCK_TYPES = ["2x2", "2x2", "2x2", "2x4", "2x2", "2x2"]            # sorting_env.BLOCK_TYPES (by block index)
PLATES = {"blue": (-0.105, -0.13), "green": (0.235, -0.16)}          # station_scene.BOWL_CENTERS
CELLS = np.array([(0, -.10), (.075, -.09), (0, -.25), (.075, -.27), (0, -.34), (.075, -.35)])
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
PHASES = ["settle", "approach", "descend", "align", "close", "grasp", "lift", "lift_check",
          "carry", "lower", "release", "retreat", "settle_placement"]


def replay_scene(seed, target_color):
    """Same numpy draws, same order, as BrickSortingEnv.reset (count=None, target and destination given)."""
    rng = np.random.default_rng(seed)
    count = int(rng.integers(4, 7))
    block_colors = rng.permutation(COLOR_KEYS).tolist()
    active = sorted(rng.choice(6, count, replace=False).tolist())
    cells = CELLS.copy()
    rng.shuffle(cells)
    xy = []
    for i in range(6):
        xy.append(cells[i] + rng.uniform(-0.003, 0.003, 2))
        rng.uniform(-np.pi, np.pi)  # yaw draw of every block, active or not
    index = block_colors.index(target_color)
    return {"target_index": index, "block_type": BLOCK_TYPES[index], "init_xy": xy[index],
            "count": count, "active": active}


rows = []
for d in sorted(RAW.glob("attempt-*")):
    meta = json.loads((d / "metadata.json").read_text())
    trace = json.loads((d / "trace.json").read_text())
    with np.load(d / "trajectory.npz") as z:
        state = z["state"].astype(float)
        action = z["action"].astype(float)
    phases = [t["phase"] for t in trace]
    first = {}
    for i, p in enumerate(phases):
        first.setdefault(p, i)
    scene = replay_scene(int(meta["seed"]), meta["target_color"])
    init = meta["initial_object_xyz"]
    init_diff = math.hypot(init[0] - scene["init_xy"][0], init[1] - scene["init_xy"][1]) * 1000
    fin = meta["final_object_xyz"]
    px, py = PLATES[meta["destination"]]
    final_dist = math.hypot(fin[0] - px, fin[1] - py) * 1000
    v = np.diff(state, axis=0) * 30.0
    ja = np.diff(action, n=2, axis=0) * 900.0
    row = {
        "attempt": d.name, "seed": meta["seed"], "color": meta["target_color"],
        "destination": meta["destination"], "task": meta["task"], "accepted": meta["accepted"],
        "success": meta["success"], "grasp_verified": meta["physical_grasp_verified"],
        "error": meta["error"] or "", "frames": meta["frames"], "n_state": len(state),
        "n_trace": len(trace), "block_index": scene["target_index"], "block_type": scene["block_type"],
        "count_active": scene["count"], "init_xy_diff_mm": round(init_diff, 3),
        "final_dist_mm": round(final_dist, 3), "max_object_z": max(t["object_xyz"][2] for t in trace),
        "release_frame": first.get("release", -1), "grasp_frame": first.get("grasp", -1),
        "path_len_rad": float(np.abs(np.diff(state, axis=0)).sum()),
    }
    rel = state[first["release"]] if "release" in first else np.full(6, np.nan)
    gr = state[first["grasp"]] if "grasp" in first else np.full(6, np.nan)
    for j, name in enumerate(JOINTS):
        row[f"rel_{name}"] = float(rel[j])
        row[f"grasp_{name}"] = float(gr[j])
        row[f"vmax_{name}"] = float(np.abs(v[:, j]).max())
        row[f"vp99_{name}"] = float(np.percentile(np.abs(v[:, j]), 99))
        row[f"jrms_{name}"] = float(np.sqrt(np.mean(ja[:, j] ** 2)))
        row[f"smin_{name}"] = float(state[:, j].min())
        row[f"smax_{name}"] = float(state[:, j].max())
        row[f"amin_{name}"] = float(action[:, j].min())
        row[f"amax_{name}"] = float(action[:, j].max())
    for p in PHASES:
        row[f"n_{p}"] = phases.count(p)
    rows.append(row)

with open(OUT / "sim_episodes.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def med(key):
    return float(np.median([r[key] for r in rows]))


summary = {
    "n_attempts": len(rows),
    "accepted": sum(bool(r["accepted"]) for r in rows),
    "success": sum(bool(r["success"]) for r in rows),
    "grasp_verified": sum(bool(r["grasp_verified"]) for r in rows),
    "errors": dict(collections.Counter(r["error"] for r in rows)),
    "n_frames_state_eq_trace": sum(r["n_state"] == r["n_trace"] == r["frames"] for r in rows),
    "colors": dict(collections.Counter(r["color"] for r in rows)),
    "destinations": dict(collections.Counter(r["destination"] for r in rows)),
    "block_types": dict(collections.Counter(r["block_type"] for r in rows)),
    "type_by_destination": {f"{t}->{d}": sum(1 for r in rows if r["block_type"] == t and r["destination"] == d)
                            for t in ("2x2", "2x4") for d in ("blue", "green")},
    "od10_contradictions": sum(1 for r in rows if (r["block_type"] == "2x2" and r["destination"] == "green")
                               or (r["block_type"] == "2x4" and r["destination"] == "blue")),
    "init_xy_diff_mm_median": med("init_xy_diff_mm"),
    "init_xy_diff_mm_max": max(r["init_xy_diff_mm"] for r in rows),
    "init_xy_within_3mm": sum(r["init_xy_diff_mm"] <= 3.0 for r in rows),
    "final_dist_mm_median": med("final_dist_mm"),
    "final_dist_mm_max": max(r["final_dist_mm"] for r in rows),
    "final_within_15mm": sum(r["final_dist_mm"] < 15.0 for r in rows),
    "release_frame_missing": sum(r["release_frame"] < 0 for r in rows),
    "frames_min_median_max": [min(r["frames"] for r in rows), med("frames"), max(r["frames"] for r in rows)],
    "vmax_rad_s_over_episodes": {j: max(r[f"vmax_{j}"] for r in rows) for j in JOINTS},
    "vp99_rad_s_median": {j: float(np.median([r[f"vp99_{j}"] for r in rows])) for j in JOINTS},
    "jerk_rms_rad_s3_median": {j: float(np.median([r[f"jrms_{j}"] for r in rows])) for j in JOINTS},
    "action_min_rad": {j: min(r[f"amin_{j}"] for r in rows) for j in JOINTS},
    "action_max_rad": {j: max(r[f"amax_{j}"] for r in rows) for j in JOINTS},
    "state_min_rad": {j: min(r[f"smin_{j}"] for r in rows) for j in JOINTS},
    "state_max_rad": {j: max(r[f"smax_{j}"] for r in rows) for j in JOINTS},
    "path_len_rad_median": med("path_len_rad"),
    "phase_frames_median": {p: float(np.median([r[f"n_{p}"] for r in rows])) for p in PHASES},
    "release_pose_rad_by_destination": {},
    "grasp_pose_rad_by_destination": {},
}
for dest in ("blue", "green"):
    sub = [r for r in rows if r["destination"] == dest]
    summary["release_pose_rad_by_destination"][dest] = {
        "n": len(sub),
        "mean": {j: float(np.mean([r[f"rel_{j}"] for r in sub])) for j in JOINTS},
        "std": {j: float(np.std([r[f"rel_{j}"] for r in sub])) for j in JOINTS},
    }
    summary["grasp_pose_rad_by_destination"][dest] = {
        "n": len(sub),
        "mean": {j: float(np.mean([r[f"grasp_{j}"] for r in sub])) for j in JOINTS},
        "std": {j: float(np.std([r[f"grasp_{j}"] for r in sub])) for j in JOINTS},
    }
(OUT / "sim_summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
print(json.dumps(summary, indent=1, default=float))
