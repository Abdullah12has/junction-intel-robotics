"""De-duplicated real statistics (raw 6d6bdce3 eps only; curated a4ba6ede eps 0-3 == raw eps 3-6, verified in recheck.py)
and URDF FK grasp radius/height for sim vs real. Writes out/final_numbers.json.
"""
import json
import math
import xml.etree.ElementTree as ET

import numpy as np

W = "/home/intel/intel-junction-2026/work/P2-D-SIM2/W2"
OUT = W + "/out"
URDF = "/home/intel/intel-junction-2026/repos/physicalai/packages/physicalai-mujoco-plugin/urdf/so101/so101_new_calib.urdf"
D = json.load(open(OUT + "/rows.json"))
K = D["K"]
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
sim = D["sim"]
raw = [r for r in D["real"] if r["set"] == "raw6d6"]
EXCL = {8, 13}  # no release detected (8) / gripper closed only to ~11, team+Opus flag (13)
grasp_rows = [r for r in raw if r["grasp_n"] is not None and r["ep"] not in EXCL]
kv = np.array([K[j] for j in JOINTS[:5]])


def rpy_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


root = ET.parse(URDF).getroot()
joints = {j.get("name"): j for j in root.findall("joint")}
by_child = {j.find("child").get("link"): j for j in joints.values()}
link = joints["gripper_frame_joint"].find("child").get("link")
chain = []
while link in by_child:
    j = by_child[link]
    chain.append(j)
    link = j.find("parent").get("link")
chain.reverse()
steps = []
for j in chain:
    o = j.find("origin")
    xyz = np.array([float(v) for v in o.get("xyz").split()])
    rpy = [float(v) for v in o.get("rpy").split()] if o is not None and o.get("rpy") else [0, 0, 0]
    steps.append((j.get("name"), j.get("type"), xyz, rpy_R(*rpy)))


def fk(q5):
    qmap = dict(zip(JOINTS[:5], q5))
    T = np.eye(4)
    for name, typ, xyz, R in steps:
        A = np.eye(4)
        A[:3, :3] = R
        A[:3, 3] = xyz
        T = T @ A
        if typ == "revolute":
            B = np.eye(4)
            B[:3, :3] = Rz(qmap[name])
            T = T @ B
    return T[:3, 3]


def stats(a):
    a = np.asarray(a, float)
    return {"median": np.round(np.median(a, axis=0), 3).tolist(), "p25": np.round(np.percentile(a, 25, axis=0), 3).tolist(),
            "p75": np.round(np.percentile(a, 75, axis=0), 3).tolist(), "n": int(len(a))}


out = {}
g_n = [np.asarray(r["grasp_n"][:5]) for r in grasp_rows]
out["real_grasp_n_dedup_excl8_13"] = stats(g_n)
out["real_grasp_eps"] = [r["ep"] for r in grasp_rows]
g_tcp = [fk(np.asarray(r["grasp_n"][:5]) / kv) for r in grasp_rows]
out["real_grasp_tcp_base_xyz_dedup"] = stats(g_tcp)
out["real_grasp_radius_m_dedup"] = stats([[math.hypot(t[0], t[1])] for t in g_tcp])
s_tcp = [fk(np.asarray(r["grasp_q"][:5])) for r in sim]
out["sim_grasp_tcp_base_xyz"] = stats(s_tcp)
out["sim_grasp_radius_m"] = stats([[math.hypot(t[0], t[1])] for t in s_tcp])
home_real = [np.asarray(r["home_n"][:5]) for r in raw]
out["real_home_n_dedup"] = stats(home_real)
out["real_home_tcp_xyz_dedup"] = stats([fk(h / kv) for h in home_real])
out["sim_HOME_tcp_xyz"] = fk(np.array([0, -0.5, 0.8, 0.6, 0])).round(4).tolist()
out["real_durations_s_dedup"] = {"median": round(float(np.median([r["frames"] for r in raw]) / 30), 2),
                                  "min": round(min(r["frames"] for r in raw) / 30, 2), "max": round(max(r["frames"] for r in raw) / 30, 2), "n": len(raw)}
out["sim_durations_s"] = {"median": round(float(np.median([r["frames"] for r in sim]) / 30), 2),
                          "min": round(min(r["frames"] for r in sim) / 30, 2), "max": round(max(r["frames"] for r in sim) / 30, 2), "n": len(sim)}
out["real_speed_ratio_p90_per_joint"] = None
rs = np.asarray([r["vmax_n_s"] for r in raw])
ss = np.asarray([[K[j] * x for j, x in zip(JOINTS, r["vmax_radps"])] for r in sim])
out["vmax_median_sim_n_s"] = np.round(np.median(ss, axis=0), 1).tolist()
out["vmax_median_real_n_s"] = np.round(np.median(rs, axis=0), 1).tolist()
out["vmax_median_ratio_real_over_sim"] = np.round(np.median(rs, axis=0) / np.median(ss, axis=0), 2).tolist()
json.dump(out, open(OUT + "/final_numbers.json", "w"), indent=1)
for k, v in out.items():
    print(k, json.dumps(v))
