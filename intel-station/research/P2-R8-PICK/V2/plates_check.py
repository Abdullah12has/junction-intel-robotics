"""Verifier V2: release and grasp positions in the robot base frame under convention A (no sign, no offset),
compared with plate XY that W3 reports (sim/results.json plates_xy_m, copied here as data).
Usage: python3 -I plates_check.py HERE"""
import json
import math
import os
import sys

import numpy as np
import pyarrow.parquet as pq

HERE = sys.argv[1]
ROOT = "/home/intel/intel-junction-2026/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af"
PROJ = json.load(open("/home/intel/intel-junction-2026/app/motion/project_robots.json"))
CAL = [c for c in PROJ if c["name"] == "SO101 Follower"][0]["payload"]["calibration"]
BODY = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
SCALE = np.array([(CAL[j]["range_max"] - CAL[j]["range_min"]) / 200.0 * 360.0 / 4095.0 for j in BODY])
W3_PLATES = {"blue": (0.1876, -0.0875), "green": (0.1868, 0.2306)}  # data: W3 sim/results.json plates_xy_m
JOINTS = [
    ((0.0388353, -8.97657e-09, 0.0624), (3.14159, 4.18253e-17, -3.14159)),
    ((-0.0303992, -0.0182778, -0.0542), (-1.5708, -1.5708, 0)),
    ((-0.11257, -0.028, 1.73763e-16), (-3.63608e-16, 8.74301e-16, 1.5708)),
    ((-0.1349, 0.0052, 3.62355e-17), (4.02456e-15, 8.67362e-16, -1.5708)),
    ((5.55112e-17, -0.0611, 0.0181), (1.5708, 0.0486795, 3.14159)),
]


def rot_rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


FIXED = []
for xyz, rpy in JOINTS:
    T = np.eye(4)
    T[:3, :3] = rot_rpy(*rpy)
    T[:3, 3] = xyz
    FIXED.append(T)
TCP = np.eye(4)
TCP[:3, :3] = rot_rpy(0, math.pi, 0)
TCP[:3, 3] = (-0.0079, -0.000218121, -0.0981274)


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def tcp(q):
    T = np.eye(4)
    for F, a in zip(FIXED, q):
        T = T @ F @ rz(a)
    return (T @ TCP)[:3, 3]


states = {}
for f in sorted(os.listdir(os.path.join(ROOT, "data", "chunk-000"))):
    if not f.endswith(".parquet"):
        continue
    d = pq.read_table(os.path.join(ROOT, "data", "chunk-000", f),
                      columns=["observation.state", "episode_index", "frame_index"]).to_pydict()
    for i in range(len(d["episode_index"])):
        states.setdefault(int(d["episode_index"][i]), []).append((int(d["frame_index"][i]), d["observation.state"][i]))
for ep in list(states):
    states[ep].sort(key=lambda r: r[0])
    states[ep] = np.array([s for _, s in states[ep]], float)

eps = json.load(open(os.path.join(HERE, "episodes.json")))
rows = []
for k in sorted(eps, key=int):
    e = eps[k]
    if e["status"] != "ok":
        continue
    S = states[int(k)]
    g = S[:, 5]
    gi = e["grasp_frame"]
    rel = next((t for t in range(gi + 15, len(g)) if g[t] > 7.0), None)
    q_g = np.radians(SCALE * S[gi, :5])
    xg = tcp(q_g)
    rec = {"ep": int(k), "task": " ".join(e["tasks"]), "grasp_xyz_m": [round(float(v), 4) for v in xg]}
    if rel is not None:
        xr = tcp(np.radians(SCALE * S[rel, :5]))
        rec.update({"release_frame": int(rel), "release_xyz_m": [round(float(v), 4) for v in xr]})
    rows.append(rec)

BLUE = [1, 7, 11, 12]  # ep7 released on BLUE per Opus visual review (FACTS-LIVE 13:45Z); 11, 12 'to the blue plate'
GREEN = [14, 15]       # 'to the green plate'
summary = {}
for name, group in (("blue", BLUE), ("green", GREEN)):
    pts = [r["release_xyz_m"] for r in rows if r["ep"] in group and "release_xyz_m" in r]
    arr = np.array(pts)
    summary[name] = {"eps": [r["ep"] for r in rows if r["ep"] in group and "release_xyz_m" in r],
                     "mean_xy_m": [round(float(v), 4) for v in arr[:, :2].mean(axis=0)],
                     "spread_xy_std_m": [round(float(v), 4) for v in arr[:, :2].std(axis=0)],
                     "release_z_mean_m": round(float(arr[:, 2].mean()), 4),
                     "w3_plate_xy_m": list(W3_PLATES[name]),
                     "diff_to_w3_mm": round(1000 * float(np.linalg.norm(arr[:, :2].mean(axis=0) - np.array(W3_PLATES[name]))), 1)}
grasp_z = np.array([r["grasp_xyz_m"][2] for r in rows])
summary["grasp_z_m"] = {"mean": round(float(grasp_z.mean()), 4), "std_mm": round(1000 * float(grasp_z.std()), 2),
                        "n": int(len(grasp_z))}
summary["grasp_xy_range_m"] = {"x": [round(min(r["grasp_xyz_m"][0] for r in rows), 3), round(max(r["grasp_xyz_m"][0] for r in rows), 3)],
                               "y": [round(min(r["grasp_xyz_m"][1] for r in rows), 3), round(max(r["grasp_xyz_m"][1] for r in rows), 3)]}
json.dump({"convention": "A_no_sign_no_offset (URDF angle = scale * normalized value, scale from project_robots.json)",
           "rows": rows, "summary": summary}, open(os.path.join(HERE, "plates_check.json"), "w"), indent=1)
print(json.dumps(summary))
for r in rows:
    print(r["ep"], r["task"][:40], r["grasp_xyz_m"], r.get("release_xyz_m"))
