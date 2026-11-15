"""Verifier V2 (P2-R8-PICK): per-episode grasp instants and top-view frames.
Read-only on the dataset. Usage: python3 -I extract.py DATASET_ROOT OUT_DIR
"""
import glob
import json
import os
import subprocess
import sys

import pyarrow.parquet as pq

ROOT = sys.argv[1]
OUT = sys.argv[2]
TOP = "observation.images.realsense 243622061421"
FPS = 30.0
os.makedirs(os.path.join(OUT, "frames"), exist_ok=True)

meta = {}
for f in sorted(glob.glob(ROOT + "/meta/episodes/**/*.parquet", recursive=True)):
    d = pq.read_table(f).to_pydict()
    for i in range(len(d["episode_index"])):
        ep = int(d["episode_index"][i])
        meta[ep] = {
            "length": int(d["length"][i]),
            "tasks": d["tasks"][i],
            "top_chunk": int(d[f"videos/{TOP}/chunk_index"][i]),
            "top_file": int(d[f"videos/{TOP}/file_index"][i]),
            "top_from_ts": float(d[f"videos/{TOP}/from_timestamp"][i]),
        }

rows = {}
for f in sorted(glob.glob(ROOT + "/data/chunk-000/*.parquet")):
    d = pq.read_table(f, columns=["observation.state", "frame_index", "episode_index"]).to_pydict()
    for i in range(len(d["episode_index"])):
        ep = int(d["episode_index"][i])
        rows.setdefault(ep, []).append((int(d["frame_index"][i]), [float(x) for x in d["observation.state"][i]]))

summary = {}
for ep in sorted(meta):
    seq = sorted(rows.get(ep, []), key=lambda r: r[0])
    states = [s for _, s in seq]
    g = [s[5] for s in states]
    n = len(states)
    entry = {"episode": ep, "frames": n, "length_meta": meta[ep]["length"], "tasks": meta[ep]["tasks"],
             "g_first": round(g[0], 2), "g_min": round(min(g), 2), "g_max": round(max(g), 2)}
    grasp = None
    reason = "no sustained closure below 4 after an open start"
    if g[0] < 8.0:
        reason = "starts closed (not an open-gripper pick)"
    else:
        for t in range(n - 15):
            if all(x < 4.0 for x in g[t:t + 15]):
                grasp = t
                reason = "ok"
                break
    entry["status"] = reason
    if grasp is not None:
        entry["grasp_frame"] = grasp
        entry["grasp_time_s"] = round(grasp / FPS, 3)
        entry["grasp_state"] = [round(x, 4) for x in states[grasp]]
        entry["start_state"] = [round(x, 4) for x in states[0]]
        entry["end_state"] = [round(x, 4) for x in states[-1]]
    summary[ep] = entry

# frames: top view at start (0), grasp, end, for every episode with a grasp, plus start for all
vid_dir = os.path.join(ROOT, "videos", "observation.images." + TOP.split("observation.images.")[-1], "chunk-%03d" % 0)
for ep, e in summary.items():
    m = meta[ep]
    path = os.path.join(ROOT, "videos", TOP, "chunk-%03d" % m["top_chunk"], "file-%03d.mp4" % m["top_file"])
    marks = {"start": 0, "end": e["frames"] - 1}
    if "grasp_frame" in e:
        marks["grasp"] = e["grasp_frame"]
    for name, idx in marks.items():
        ts = m["top_from_ts"] + idx / FPS
        out = os.path.join(OUT, "frames", "ep%02d_%s.png" % (ep, name))
        subprocess.run(["ffmpeg", "-v", "error", "-ss", "%.4f" % ts, "-i", path, "-frames:v", "1", "-y", out],
                       check=False)
        e["frame_" + name] = os.path.relpath(out, OUT)
    print("ep", ep, e["status"], e.get("grasp_frame"), flush=True)

with open(os.path.join(OUT, "episodes.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print("WROTE", len(summary), "episodes", flush=True)
