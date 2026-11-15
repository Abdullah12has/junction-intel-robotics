"""Opus fast G-M prep: plate<->home waypoints from recorded human returns (collision-free in this scene).

Each sequence is the recorded release->home path of one episode, simplified with Ramer-Douglas-Peucker
(max joint deviation EPS units), reversed for home->plate. The plate end point is taken LIFT_S seconds after
the release (a little above the release pose). Gripper fixed at GRIP (open, no brick).
Usage: python gm_fast.py OUT_YAML
"""
import glob
import sys

import numpy as np
import json

import pandas as pd

D = "/home/intel/intel-junction-2026/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af"
EPS = 3.0
LIFT_S = 0.3
GRIP = 25.0
SPEED = 15.0
FPS = 30
# (name, episode, release frame) from work/opus/release2 (Opus visual check: ep11 BLUE, ep14 GREEN)
SOURCES = [("blue", 11, 600), ("green", 14, 357)]

data = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(D + "/data/*/*.parquet"))])


def rdp(pts, eps):
    if len(pts) < 3:
        return [0, len(pts) - 1]
    a, b = pts[0], pts[-1]
    t = np.linspace(0.0, 1.0, len(pts))[:, None]
    dev = np.abs(pts - (a + t * (b - a))).max(axis=1)
    i = int(dev.argmax())
    if dev[i] <= eps:
        return [0, len(pts) - 1]
    left = rdp(pts[: i + 1], eps)
    right = rdp(pts[i:], eps)
    return left[:-1] + [j + i for j in right]


poses = {}
seqs = {}
for name, ep, rel in SOURCES:
    d = data[data.episode_index == ep].sort_values("frame_index")
    st = np.stack(d["observation.state"].values).astype(float)
    start = rel + int(round(LIFT_S * FPS))
    path = st[start:, :5]
    # stop at the end of the motion (when the arm is back at rest and still)
    keep = rdp(path, EPS)
    wps = []
    for k in keep:  # drop consecutive near-duplicates (zero-length moves)
        if not wps or np.abs(path[k] - wps[-1]).max() >= 0.5:
            wps.append(path[k])
    names = []
    for j, w in enumerate(wps):
        pn = f"{name}_wp{j}"
        poses[pn] = [round(float(x), 2) for x in w] + [GRIP]
        names.append(pn)
    print(name, "episode", ep, "waypoints", len(wps), "from", len(path), "frames")
    for pn in names:
        print("  ", pn, poses[pn])
    plate_to_home = [{"pose": pn, "speed": SPEED} for pn in names]
    plate_to_home[-1]["dwell_s"] = 0.5
    home_to_plate = [{"pose": pn, "speed": SPEED} for pn in reversed(names)]
    home_to_plate[-1]["dwell_s"] = 1.0
    seqs[f"home_to_{name}"] = home_to_plate
    seqs[f"{name}_to_home"] = plate_to_home
    seqs[f"{name}_round_trip"] = home_to_plate + plate_to_home[1:]

doc = {"placeholder": False, "unit": "normalized",
       "joint_order": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
       "poses": poses, "sequences": seqs}
with open(sys.argv[1], "w") as f:
    json.dump(doc, f, indent=1)  # JSON is valid YAML; data-env has no PyYAML
print("wrote", sys.argv[1])
