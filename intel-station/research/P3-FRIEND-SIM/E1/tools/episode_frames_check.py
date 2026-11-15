"""Read-only check of per-attempt frame counts in the friend's sim (data only).

Compares trajectory.npz length with metadata.json 'frames' for every attempt
under the live mirror, and prints the distribution. Writes nothing.
Run with: envs/data-env/bin/python -I episode_frames_check.py
"""
import glob
import json
import os
import statistics

import numpy as np

ROOT = "/home/intel/intel-junction-2026/data/sim/live/2026-11-14"

npz_paths = sorted(glob.glob(os.path.join(ROOT, "**", "trajectory.npz"), recursive=True))
meta_paths = sorted(glob.glob(os.path.join(ROOT, "**", "metadata.json"), recursive=True))
meta_by_dir = {os.path.dirname(p): p for p in meta_paths}

rows = []
mismatch = []
for p in npz_paths:
    d = np.load(p)
    n_npz = int(d["state"].shape[0])
    a_npz = int(d["action"].shape[0])
    dirp = os.path.dirname(p)
    with open(meta_by_dir[dirp], "r", encoding="utf-8") as fh:
        m = json.load(fh)
    n_meta = int(m.get("frames", -1))
    rows.append(n_npz)
    if n_npz != n_meta or a_npz != n_npz:
        mismatch.append({"dir": os.path.basename(dirp), "npz_state": n_npz, "npz_action": a_npz, "meta": n_meta})

counts = {}
for n in rows:
    counts[n] = counts.get(n, 0) + 1

out = {
    "root": ROOT,
    "trajectory_files": len(npz_paths),
    "metadata_files": len(meta_paths),
    "frames_min": min(rows),
    "frames_max": max(rows),
    "frames_median": statistics.median(rows),
    "frames_mean": round(sum(rows) / len(rows), 2),
    "frames_total": sum(rows),
    "duration_s_min": round(min(rows) / 30.0, 3),
    "duration_s_median": round(statistics.median(rows) / 30.0, 3),
    "duration_s_mean": round(sum(rows) / len(rows) / 30.0, 3),
    "duration_s_max": round(max(rows) / 30.0, 3),
    "count_by_frames": {str(k): counts[k] for k in sorted(counts)},
    "npz_meta_mismatches": mismatch,
}
print(json.dumps(out, indent=1))
