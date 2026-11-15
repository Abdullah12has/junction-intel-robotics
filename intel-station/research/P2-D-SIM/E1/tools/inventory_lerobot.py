#!/usr/bin/env python3
"""Read-only inventory of a LeRobot v3 dataset root. usage: inventory_lerobot.py ROOT OUT_JSON
Run as: venv/bin/python -I -B inventory_lerobot.py ROOT OUT_JSON"""
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq

root, out = sys.argv[1], sys.argv[2]


def jload(p):
    with open(p) as f:
        return json.load(f)


info = jload(os.path.join(root, "meta", "info.json"))
res = {
    "root": root,
    "codebase_version": info.get("codebase_version"),
    "fps": info.get("fps"),
    "robot_type": info.get("robot_type"),
    "total_episodes": info.get("total_episodes"),
    "total_frames": info.get("total_frames"),
    "total_tasks": info.get("total_tasks"),
    "data_path": info.get("data_path"),
    "video_path": info.get("video_path"),
    "features": {k: {"dtype": v.get("dtype"), "shape": v.get("shape"), "names": v.get("names"),
                     "codec": (v.get("info") or {}).get("video.codec")}
                 for k, v in info.get("features", {}).items()},
}

# tasks
tasks_files = glob.glob(os.path.join(root, "meta", "tasks*.parquet"))
tasks = []
if tasks_files:
    t = pq.read_table(tasks_files[0])
    res["tasks_columns"] = t.column_names
    tasks = t.to_pylist()
    res["tasks"] = tasks

# episodes metadata
ep_files = sorted(glob.glob(os.path.join(root, "meta", "episodes", "**", "*.parquet"), recursive=True))
ep_rows = []
for p in ep_files:
    ep_rows += pq.read_table(p).to_pylist()
if ep_rows:
    res["episodes_meta_columns"] = sorted(ep_rows[0].keys())
    res["episodes_meta_count"] = len(ep_rows)

# data frames
data_files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
res["data_files"] = [os.path.relpath(p, root) for p in data_files]
cols = None
state, action, ep_idx, ts, fidx, tidx = [], [], [], [], [], []
for p in data_files:
    tb = pq.read_table(p)
    cols = tb.column_names
    if "observation.state" in cols:
        state.append(np.array(tb.column("observation.state").to_pylist(), dtype=np.float64))
    if "action" in cols:
        action.append(np.array(tb.column("action").to_pylist(), dtype=np.float64))
    ep_idx.append(np.array(tb.column("episode_index").to_pylist()))
    if "timestamp" in cols:
        ts.append(np.array(tb.column("timestamp").to_pylist(), dtype=np.float64))
    if "task_index" in cols:
        tidx.append(np.array(tb.column("task_index").to_pylist()))
res["data_columns"] = cols
ep = np.concatenate(ep_idx) if ep_idx else np.array([])
res["data_rows"] = int(ep.size)
uniq, counts = np.unique(ep, return_counts=True)
res["episode_count_in_data"] = int(uniq.size)
res["episode_lengths_min_max_mean"] = [int(counts.min()), int(counts.max()), round(float(counts.mean()), 2)] if counts.size else None
res["episode_lengths"] = {int(u): int(c) for u, c in zip(uniq, counts)}
if tidx:
    tt = np.concatenate(tidx)
    tu, tc = np.unique(tt, return_counts=True)
    res["task_index_counts"] = {int(a): int(b) for a, b in zip(tu, tc)}


def stats(arr):
    a = np.concatenate(arr) if arr else None
    if a is None or a.size == 0:
        return None
    return {"n": int(a.shape[0]), "dim": int(a.shape[1]),
            "min": np.round(a.min(0), 4).tolist(), "max": np.round(a.max(0), 4).tolist(),
            "mean": np.round(a.mean(0), 4).tolist(), "std": np.round(a.std(0), 4).tolist(),
            "p1": np.round(np.percentile(a, 1, axis=0), 4).tolist(),
            "p99": np.round(np.percentile(a, 99, axis=0), 4).tolist()}


res["observation.state_stats"] = stats(state)
res["action_stats"] = stats(action)
if ts:
    tsa = np.concatenate(ts)
    res["timestamp_max"] = round(float(tsa.max()), 4)

# videos
res["videos"] = {}
for key_dir in sorted(glob.glob(os.path.join(root, "videos", "*"))):
    key = os.path.basename(key_dir)
    files = sorted(glob.glob(os.path.join(key_dir, "**", "*.mp4"), recursive=True))
    entries = []
    for f in files[:3]:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=codec_name,width,height,r_frame_rate,nb_frames,pix_fmt",
                            "-of", "json", f], capture_output=True, text=True)
        entries.append({"file": os.path.relpath(f, root), "ffprobe": json.loads(r.stdout or "{}").get("streams", [])})
    res["videos"][key] = {"file_count": len(files), "sample": entries}

with open(out, "w") as f:
    json.dump(res, f, indent=1, sort_keys=True)
print(json.dumps({"out": out, "episodes": res.get("episode_count_in_data"), "rows": res.get("data_rows"),
                  "videos": list(res["videos"].keys())}))
