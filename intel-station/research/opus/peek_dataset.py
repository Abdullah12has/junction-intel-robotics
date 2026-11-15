"""Opus quick look at one Studio dataset copy: per-episode gripper/arm summary and a key-frame contact sheet."""
import glob
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd

D = sys.argv[1]
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
info = json.load(open(os.path.join(D, "meta/info.json")))
fps = info["fps"]
names = info["features"]["observation.state"].get("names")
print("state names:", names)
tasks = pd.read_parquet(os.path.join(D, "meta/tasks.parquet"))
print("tasks:", tasks.reset_index().to_dict("records"))
ep = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(D, "meta/episodes/*/*.parquet")))])
data = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(D, "data/*/*.parquet")))])
keys = [k for k in info["features"] if k.startswith("observation.images.")]
rows = []
for _, r in ep.sort_values("episode_index").iterrows():
    e = int(r["episode_index"])
    d = data[data.episode_index == e]
    st = np.stack(d["observation.state"].values)
    g = st[:, names.index("gripper.pos")] if names and "gripper.pos" in names else st[:, -1]
    per_s = [round(float(x), 1) for x in g[::fps]]
    print(f"ep{e} frames={len(d)} dur={len(d)/fps:.1f}s grip[min,max]=[{g.min():.1f},{g.max():.1f}] grip_per_s={per_s}")
    print(f"   start_pose={np.round(st[0], 1).tolist()}")
    print(f"   end_pose  ={np.round(st[-1], 1).tolist()}")
    row = []
    for key in keys:
        vf = os.path.join(D, "videos", key, "chunk-%03d" % int(r[f"videos/{key}/chunk_index"]), "file-%03d.mp4" % int(r[f"videos/{key}/file_index"]))
        t0 = float(r[f"videos/{key}/from_timestamp"])
        t1 = float(r[f"videos/{key}/to_timestamp"])
        for frac in (0.02, 0.35, 0.6, 0.97):
            p = os.path.join(OUT, f"f_{e}_{keys.index(key)}_{frac:.2f}.jpg")
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{t0 + (t1 - t0) * frac:.3f}", "-i", vf, "-frames:v", "1", "-y", p], check=False)
            im = cv2.imread(p)
            im = cv2.resize(im, (240, 180)) if im is not None else np.zeros((180, 240, 3), np.uint8)
            cv2.putText(im, f"ep{e} cam{keys.index(key)} {frac:.2f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            row.append(im)
    rows.append(np.hstack(row))
cv2.imwrite(os.path.join(OUT, "sheet.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
print("sheet:", os.path.join(OUT, "sheet.jpg"), "keys:", keys)
