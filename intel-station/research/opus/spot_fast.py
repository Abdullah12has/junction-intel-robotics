"""Opus mode (c) 'spot cell' sequences from recorded, collision-free paths (D-082).

Spot = where episode EP put its brick down (S1 put-down). The recorded path release(+LIFT_S) -> home is
simplified (RDP, EPS units) and reversed for home -> above spot; the spot pose itself is the joint
state at the release. Plate legs reuse app/motion/gm/gm_fast.yaml (live-validated 14:29-14:32Z).
Writes app/motion/gm/spot_fast.yaml (JSON is valid YAML) and a top-view frame of the release moment.
Usage: python spot_fast.py EP
"""
import glob
import json
import subprocess
import sys

import numpy as np
import pandas as pd

R = "/home/intel/intel-junction-2026"
D = f"{R}/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af"
EP = int(sys.argv[1]) if len(sys.argv) > 1 else 5
EPS, LIFT_S, FPS, OPEN, CLOSED, SPEED = 3.0, 0.3, 30, 25.0, 2.0, 15.0

data = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(D + "/data/*/*.parquet"))])
d = data[data.episode_index == EP].sort_values("frame_index")
st = np.stack(d["observation.state"].values).astype(float)
g = st[:, 5]
grasp = next(i for i in range(1, len(g)) if g[i] < 4.0 and g[i - 1] >= 4.0)
rel = next(i for i in range(grasp + 1, len(g)) if g[i] >= 6.0 and g[i - 1] < 6.0)
print(f"ep{EP}: grasp frame {grasp}, release frame {rel} ({rel / FPS:.2f} s), end {len(g)}")
spot = st[rel, :5]
start = rel + int(round(LIFT_S * FPS))
path = st[start:, :5]


def rdp(pts, eps):
    if len(pts) < 3:
        return [0, len(pts) - 1]
    a, b = pts[0], pts[-1]
    t = np.linspace(0.0, 1.0, len(pts))[:, None]
    dev = np.abs(pts - (a + t * (b - a))).max(axis=1)
    i = int(dev.argmax())
    if dev[i] <= eps:
        return [0, len(pts) - 1]
    left, right = rdp(pts[: i + 1], eps), rdp(pts[i:], eps)
    return left[:-1] + [j + i for j in right]


wps = []
for k in rdp(path, EPS):
    if not wps or np.abs(path[k] - wps[-1]).max() >= 0.5:
        wps.append(path[k])
gm = json.load(open(f"{R}/app/motion/gm/gm_fast.yaml"))
poses = {k: v for k, v in gm["poses"].items()}
r2 = lambda v: [round(float(x), 2) for x in v]  # noqa: E731
poses["spot_open"] = r2(spot) + [OPEN]
poses["spot_closed"] = r2(spot) + [CLOSED]
names = []
for j, w in enumerate(wps):
    poses[f"spot_wp{j}_open"] = r2(w) + [OPEN]
    poses[f"spot_wp{j}_closed"] = r2(w) + [CLOSED]
    names.append(j)
seq = {}
step = lambda p, dw=0.0: ({"pose": p, "speed": SPEED, "dwell_s": dw} if dw else {"pose": p, "speed": SPEED})  # noqa: E731
seq["home_to_spot"] = [step(f"spot_wp{j}_open") for j in reversed(names)] + [step("spot_open", 0.5)]
seq["grasp_at_spot"] = [step("spot_open"), step("spot_closed", 0.8)]
seq["spot_to_home_carry"] = [step(f"spot_wp{j}_closed") for j in names]
seq["spot_to_home_open"] = [step(f"spot_wp{j}_open") for j in names]
for plate in ("blue", "green"):
    pn = sorted((k for k in gm["poses"] if k.startswith(f"{plate}_wp")), key=lambda k: int(k.split("wp")[1]))
    for k in pn:
        poses[f"{k}_closed"] = gm["poses"][k][:5] + [CLOSED]
    seq[f"home_to_{plate}_carry"] = [step(f"{k}_closed") for k in reversed(pn)]
    seq[f"release_{plate}"] = [step(pn[0], 0.8)]
    seq[f"{plate}_to_home"] = [step(k) for k in pn]
    seq[f"cycle_{plate}"] = (seq["home_to_spot"] + seq["grasp_at_spot"] + seq["spot_to_home_carry"]
                             + seq[f"home_to_{plate}_carry"] + seq[f"release_{plate}"] + seq[f"{plate}_to_home"][1:])
seq["spot_dry_touch"] = seq["home_to_spot"] + seq["spot_to_home_open"]
doc = {"placeholder": False, "unit": "normalized",
       "joint_order": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
       "poses": poses, "sequences": seq}
json.dump(doc, open(f"{R}/app/motion/gm/spot_fast.yaml", "w"), indent=1)
print("spot pose", r2(spot), "| waypoints", len(wps), "| sequences", list(seq))
key = "observation.images.realsense 243622061421"
ep = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(D + "/meta/episodes/*/*.parquet"))])
r = ep[ep.episode_index == EP].iloc[0]
vf = f"{D}/videos/{key}/chunk-{int(r[f'videos/{key}/chunk_index']):03d}/file-{int(r[f'videos/{key}/file_index']):03d}.mp4"
t0 = float(r[f"videos/{key}/from_timestamp"])
for tag, fr in (("release", rel), ("end", len(g) - 1)):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{t0 + fr / FPS:.3f}", "-i", vf,
                    "-frames:v", "1", "-y", f"{R}/work/opus/spot_ep{EP}_{tag}.jpg"], check=False)
print("frames", f"{R}/work/opus/spot_ep{EP}_release.jpg")
