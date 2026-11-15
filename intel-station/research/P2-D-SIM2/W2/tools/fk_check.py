"""Forward-kinematics cross-check (URDF text parse, numpy only).

URDF: repos/physicalai/packages/physicalai-mujoco-plugin/urdf/so101/so101_new_calib.urdf (read-only).
Mapping: q_rad = n / K (body joints, sign +1, zero offset 0; see tools/validate_mapping.py).
Sim base placement (station_scene.py): quat Rz(-90 deg), pos (0.0388353, 0.0388353, 0).
Outputs: out/fk_check.json and stdout.
"""
import json
import math
import xml.etree.ElementTree as ET
import numpy as np

URDF = "/home/intel/intel-junction-2026/repos/physicalai/packages/physicalai-mujoco-plugin/urdf/so101/so101_new_calib.urdf"
OUT = "/home/intel/intel-junction-2026/work/P2-D-SIM2/W2/out"
D = json.load(open(OUT + "/rows.json"))
sim, real, K = D["sim"], [r for r in D["real"] if r["set"] == "raw6d6"], D["K"]
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
PLATE_WORLD = {"blue": (-0.105, -0.13), "green": (0.235, -0.16)}
BASE_POS = np.array([0.0388353, 0.0388353])


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
print("CHAIN", [(j.get("name"), j.get("type"), j.find("parent").get("link"), j.find("child").get("link")) for j in chain])
T_STEPS = []
for j in chain:
    o = j.find("origin")
    xyz = np.array([float(v) for v in o.get("xyz").split()])
    rpy = [float(v) for v in o.get("rpy").split()] if o is not None and o.get("rpy") else [0, 0, 0]
    T_STEPS.append((j.get("name"), j.get("type"), xyz, rpy_R(*rpy)))


def fk(q5):
    """q5: pan, lift, elbow, wrist_flex, wrist_roll (radians). Returns TCP in URDF base frame."""
    qmap = dict(zip(JOINTS[:5], q5))
    T = np.eye(4)
    for name, typ, xyz, R in T_STEPS:
        A = np.eye(4)
        A[:3, :3] = R
        A[:3, 3] = xyz
        T = T @ A
        if typ == "revolute" and name in qmap:
            B = np.eye(4)
            B[:3, :3] = Rz(qmap[name])
            T = T @ B
        elif typ == "revolute":
            raise SystemExit(f"revolute joint {name} not in q map")
    return T[:3, 3]


def to_world_rz(p_base, ang):
    """Base-frame TCP -> sim world xy, assuming base quat rotation ang (rad) and BASE_POS."""
    return Rz(ang)[:2, :2] @ p_base[:2] + BASE_POS


out = {}
for ang_deg in (-90.0, 90.0, 0.0):
    res = {}
    for dest, (px, py) in PLATE_WORLD.items():
        tcps = [fk(np.asarray(r["place_q"])[:5]) for r in sim if r["dest"] == dest]
        worlds = np.array([to_world_rz(t, math.radians(ang_deg)) for t in tcps])
        err = np.linalg.norm(worlds - np.array([px, py]), axis=1)
        res[dest] = {"sim_place_tcp_world_xy_median": np.median(worlds, axis=0).round(4).tolist(),
                     "plate_xy": [px, py],
                     "xy_err_median_m": round(float(np.median(err)), 4),
                     "xy_err_p90_m": round(float(np.percentile(err, 90)), 4),
                     "tcp_z_place_median_m": round(float(np.median([t[2] for t in tcps])), 4)}
    out[f"base_rot_{int(ang_deg)}"] = res
out["base_rot_rule"] = "world = Rz(ang) @ tcp_base + (0.0388353, 0.0388353); -90 deg is the station_scene quat"

# grasp TCP heights: sim (descend pose) vs real (grasp frames, n/K with sign +1)
sim_g = np.array([fk(np.asarray(r["grasp_q"])[:5]) for r in sim])
real_g = np.array([fk(np.asarray(r["grasp_n"])[:5] / np.array([K[j] for j in JOINTS[:5]])) for r in real if r["grasp_n"] is not None])
real_p = {}
for dest, eps in (("blue", {7, 11, 12}), ("green", {9, 10, 14, 15})):
    pts = [fk(np.asarray(r["place_n"])[:5] / np.array([K[j] for j in JOINTS[:5]])) for r in real if r["ep"] in eps and r["place_n"] is not None]
    real_p[dest] = {"tcp_base_xyz_median": np.median(np.array(pts), axis=0).round(4).tolist(), "n": len(pts)}
out["sim_grasp_tcp_base_z"] = {"median": round(float(np.median(sim_g[:, 2])), 4), "p25": round(float(np.percentile(sim_g[:, 2], 25)), 4), "p75": round(float(np.percentile(sim_g[:, 2], 75)), 4), "n": int(len(sim_g))}
out["real_grasp_tcp_base_z"] = {"median": round(float(np.median(real_g[:, 2])), 4), "p25": round(float(np.percentile(real_g[:, 2], 25)), 4), "p75": round(float(np.percentile(real_g[:, 2], 75)), 4), "n": int(len(real_g))}
out["sim_grasp_tcp_base_xy_r_median"] = round(float(np.median(np.hypot(sim_g[:, 0], sim_g[:, 1]))), 4)
out["real_grasp_tcp_base_xy_r_median"] = round(float(np.median(np.hypot(real_g[:, 0], real_g[:, 1]))), 4)
out["real_place_tcp_base_xyz_median"] = real_p
real_home = np.array([r["home_n"][:5] for r in real]) / np.array([K[j] for j in JOINTS[:5]])
out["real_home_tcp_base_xyz_median"] = np.median(np.array([fk(q) for q in real_home]), axis=0).round(4).tolist()
out["sim_HOME_tcp_base_xyz"] = fk(np.array([0, -0.5, 0.8, 0.6, 0])).round(4).tolist()
json.dump(out, open(OUT + "/fk_check.json", "w"), indent=1)
for k, v in out.items():
    print(k, v)
