"""Opus check: top-view frame at each gripper-release event, to see which plate each release goes to."""
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
names = info["features"]["observation.state"]["names"]
gi = names.index("gripper.pos")
ep = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(D, "meta/episodes/*/*.parquet")))])
data = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(D, "data/*/*.parquet")))])
key = "observation.images.realsense 243622061421"
tiles = []
for _, r in ep.sort_values("episode_index").iterrows():
    e = int(r["episode_index"])
    d = data[data.episode_index == e]
    st = np.stack(d["observation.state"].values)
    g = st[:, gi]
    events = [i for i in range(1, len(g)) if g[i - 1] < 6.0 <= g[i]]
    vf = os.path.join(D, "videos", key, "chunk-%03d" % int(r[f"videos/{key}/chunk_index"]), "file-%03d.mp4" % int(r[f"videos/{key}/file_index"]))
    t0 = float(r[f"videos/{key}/from_timestamp"])
    for k, i in enumerate(events):
        p = os.path.join(OUT, f"rel_{e}_{k}.jpg")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{t0 + i / fps + 0.2:.3f}", "-i", vf, "-frames:v", "1", "-y", p], check=False)
        im = cv2.imread(p)
        im = cv2.resize(im, (320, 240)) if im is not None else np.zeros((240, 320, 3), np.uint8)
        pose = np.round(st[i], 1).tolist()
        cv2.putText(im, f"ep{e} release{k} t={i / fps:.1f}s", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(im, f"pan {pose[0]} lift {pose[1]} elb {pose[2]}", (4, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        tiles.append(im)
        print(f"ep{e} release{k} frame={i} t={i / fps:.2f}s pose={pose}")
while len(tiles) % 4:
    tiles.append(np.zeros((240, 320, 3), np.uint8))
cv2.imwrite(os.path.join(OUT, "release_sheet.jpg"), np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]))
print("sheet", os.path.join(OUT, "release_sheet.jpg"))
