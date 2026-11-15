"""Verifier V2: label-free convention test on dataset 6d6bdce3 (all 10158 frames).
For each joint convention: (1) URDF joint-limit violations over all frames, (2) TCP height at the 12 grasp instants
(the table is flat, so grasp heights should agree), (3) minimum TCP height over all frames.
Usage: python3 -I limits.py HERE"""
import glob
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
LO = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385])
HI = np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121])
HYP = {
    "A_no_sign_no_offset": ((1, 1, 1, 1, 1), (0, 0, 0, 0, 0)),
    "B_studio_sign_no_offset": ((1, -1, 1, 1, 1), (0, 0, 0, 0, 0)),
    "C_studio_sign_and_offset": ((1, -1, 1, 1, 1), (0, 90, 90, 0, 0)),
    "D_offset_no_sign": ((1, 1, 1, 1, 1), (0, 90, 90, 0, 0)),
    "E_pan_flip_and_lift_sign": ((-1, -1, 1, 1, 1), (0, 0, 0, 0, 0)),
}
JOINTS = [
    ((0.0388353, -8.97657e-09, 0.0624), (3.14159, 4.18253e-17, -3.14159)),
    ((-0.0303992, -0.0182778, -0.0542), (-1.5708, -1.5708, 0)),
    ((-0.11257, -0.028, 1.73763e-16), (-3.63608e-16, 8.74301e-16, 1.5708)),
    ((-0.1349, 0.0052, 3.62355e-17), (4.02456e-15, 8.67362e-16, -1.5708)),
    ((5.55112e-17, -0.0611, 0.0181), (1.5708, 0.0486795, 3.14159)),
]
TCP_T = np.array([[1, 0, 0, -0.0079], [0, -1, 0, -0.000218121], [0, 0, -1, -0.0981274], [0, 0, 0, 1.0]])
# TCP_T is rpy (0, pi, 0): diag(1,-1,-1) with the translation above


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


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def tcp_xyz(q):
    T = np.eye(4)
    for F, a in zip(FIXED, q):
        T = T @ F @ rz(a)
    return (T @ TCP_T)[:3, 3]


rows, grasp_states, release_states = [], {}, {}
for f in sorted(os.listdir(os.path.join(ROOT, "data", "chunk-000"))):
    if not f.endswith(".parquet"):
        continue
    d = pq.read_table(os.path.join(ROOT, "data", "chunk-000", f),
                      columns=["observation.state", "episode_index", "frame_index"]).to_pydict()
    rows += [(int(e), np.array(s, float)) for e, s in zip(d["episode_index"], d["observation.state"])]
S_all = np.array([s for _, s in rows])
E_all = np.array([e for e, _ in rows])
eps = json.load(open(os.path.join(HERE, "episodes.json")))
grasp_list = [(int(k), np.array(v["grasp_state"], float)) for k, v in eps.items() if v["status"] == "ok"]

out = {}
for hyp, (sg, of) in HYP.items():
    sg = np.array(sg, float)
    of = np.array(of, float)
    Q = np.radians(sg * SCALE * S_all[:, :5] + of)
    viol = np.maximum(LO - Q, 0) + np.maximum(Q - HI, 0)
    out_frames = int(np.sum(np.any(viol > 0.02, axis=1)))
    max_viol_deg = [round(math.degrees(float(v)), 1) for v in viol.max(axis=0)]
    Z = np.array([tcp_xyz(q)[2] for q in Q[::5]])  # every 5th frame to keep it fast
    zg = []
    for ep, s in grasp_list:
        q = np.radians(sg * SCALE * s[:5] + of)
        zg.append(float(tcp_xyz(q)[2]))
    zg = np.array(zg)
    out[hyp] = {"frames_total": int(len(Q)), "frames_outside_urdf_limits_by_0p02rad": out_frames,
                "frac_outside": round(out_frames / len(Q), 4), "max_violation_deg_per_joint": max_viol_deg,
                "tcp_z_all_frames_m": {"min": round(float(Z.min()), 4), "p01": round(float(np.percentile(Z, 1)), 4),
                                       "median": round(float(np.median(Z)), 4), "max": round(float(Z.max()), 4)},
                "tcp_z_at_12_grasps_m": {"mean": round(float(zg.mean()), 4), "std": round(float(zg.std()), 4),
                                          "min": round(float(zg.min()), 4), "max": round(float(zg.max()), 4),
                                          "per_episode": {str(k): round(float(z), 4) for (k, _), z in zip(grasp_list, zg)}}}
    print(hyp, out[hyp]["frac_outside"], out[hyp]["max_violation_deg_per_joint"], out[hyp]["tcp_z_at_12_grasps_m"]["std"],
          out[hyp]["tcp_z_all_frames_m"]["min"], flush=True)
json.dump(out, open(os.path.join(HERE, "limits_convention_test.json"), "w"), indent=1)
print("WROTE limits_convention_test.json frames", len(S_all))
