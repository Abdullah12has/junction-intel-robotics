#!/usr/bin/env python3
"""P2-D-SIM2 W3: forward-kinematics cross-check. Model = DeepMind MuJoCo Menagerie robotstudio_so101 at the revision
pinned by the team's setup_assets.py (370c4396...). Downloads only the MJCF and its meshes into work/P2-D-SIM2/W3/mjcf.
Compares sim expert release poses (rad), the real release poses (Studio normalized units mapped with the calibration),
and the joint ranges. Writes work/P2-D-SIM2/W3/numbers_fk.json only."""
import json
import math
import os
import urllib.request
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

W = "/home/intel/intel-junction-2026/work/P2-D-SIM2/W3"
U = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/370c43965323c0944b433b202aac5058b5f907a9/robotstudio_so101"
XML = f"{W}/mjcf/so101.xml"
K = 2 * math.pi / 4096.0
R2D = 180.0 / math.pi
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CAL = [r for r in json.load(open("/home/intel/intel-junction-2026/app/motion/project_robots.json"))
       if r["type"] == "SO101_Follower"][0]["payload"]["calibration"]
UPR = {j: (100.0 if j == "gripper" else 200.0) / (K * (CAL[j]["range_max"] - CAL[j]["range_min"])) for j in J}
N0 = {j: ((100.0 if j == "gripper" else 200.0) * (2047.5 - CAL[j]["range_min"]) / (CAL[j]["range_max"] - CAL[j]["range_min"]))
      - (0.0 if j == "gripper" else 100.0) for j in J}
GRASP = np.array([0.008, 0.0, -0.093])   # expert.py grasp_point (gripper frame)
ROT_BASE = -math.pi / 2                  # station_scene.py base quaternion (about z)
BASE_XY = np.array([0.0388353, 0.0388353])

# 1. fetch the MJCF and its meshes (same files as setup_assets.py)
os.makedirs(f"{W}/mjcf/assets", exist_ok=True)
if not os.path.exists(XML):
    urllib.request.urlretrieve(f"{U}/so101.xml", XML)
root = ET.parse(XML).getroot()
for m in root.findall("./asset/mesh"):
    dest = f"{W}/mjcf/assets/{m.attrib['file']}"
    if not os.path.exists(dest):
        urllib.request.urlretrieve(f"{U}/assets/{m.attrib['file']}", dest)
model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)
qadr = {j: model.jnt_qposadr[model.joint(j).id] for j in J}
rng = {j: model.jnt_range[model.joint(j).id] for j in J}
out = {"mjcf_joint_ranges_deg": {j: [round(float(rng[j][0] * R2D), 1), round(float(rng[j][1] * R2D), 1)] for j in J},
       "calibrated_half_range_rad_plus_minus100units": {j: round(100.0 / UPR[j], 3) for j in J if j != "gripper"}}


def tip_base(q5):
    for i, j in enumerate(J[:5]):
        data.qpos[qadr[j]] = q5[i]
    mujoco.mj_kinematics(model, data)
    b = data.body("gripper")
    return b.xpos + b.xmat.reshape(3, 3) @ GRASP


def to_world(p):
    x, y, z = p
    # world = Rz(-90 deg) * local + base offset (station_scene.py)
    return np.array([y + BASE_XY[0], -x + BASE_XY[1], z])


# 2. validation: sim expert release poses should land on the sim plates (expert tolerance 8 mm)
PLATES = {"blue": np.array([-0.105, -0.13]), "green": np.array([0.235, -0.16])}
sim_rel = {"blue": [0.861, 0.01, 0.327, 1.233, 0.783], "green": [-0.906, 0.553, -0.45, 1.468, -0.822]}
out["sim_release_check"] = {}
for k, q in sim_rel.items():
    w = to_world(tip_base(q))
    out["sim_release_check"][k] = {"tip_world_xyz": [round(float(v), 4) for v in w],
                                   "err_to_plate_mm": round(float(1000 * np.linalg.norm(w[:2] - PLATES[k])), 1)}

# 3. real release poses -> radians with the calibration (sign +1, mid-range zero) -> tips in the base frame
REAL = {"blue_ep7": [32.1, 5.47, 25.59, 61.85, -36.78], "green_ep8": [-54.67, 24.51, 1.91, 54.22, -55.03],
        "green_ep9": [-57.26, 18.52, 7.89, 36.16, -52.63], "green_ep10": [-51.83, 47.71, -49.82, 80.08, -52.37]}
res = {}
for k, u in REAL.items():
    q = [(u[i] - N0[J[i]]) / UPR[J[i]] for i in range(5)]
    p = tip_base(q)
    res[k] = {"q_rad": [round(float(v), 3) for v in q], "tip_base_xyz": [round(float(v), 4) for v in p],
              "azimuth_deg": round(float(math.degrees(math.atan2(p[1], p[0]))), 1)}
bl = np.array(res["blue_ep7"]["tip_base_xyz"])
gr = np.array(res["green_ep8"]["tip_base_xyz"])
out["real_release_tips"] = res
out["real_chord_blue_green_m(z-included)"] = round(float(np.linalg.norm(bl - gr)), 4)
out["real_chord_xy_m"] = round(float(np.linalg.norm(bl[:2] - gr[:2])), 4)
sb = tip_base(sim_rel["blue"])
sg = tip_base(sim_rel["green"])
out["sim_chord_xy_m_base_frame"] = round(float(np.linalg.norm(sb[:2] - sg[:2])), 4)
out["sim_azimuth_blue_green_deg"] = [round(float(math.degrees(math.atan2(sb[1], sb[0]))), 1),
                                     round(float(math.degrees(math.atan2(sg[1], sg[0]))), 1)]
out["real_reach_radius_m_blue_green"] = [round(float(np.hypot(bl[0], bl[1])), 4), round(float(np.hypot(gr[0], gr[1])), 4)]
out["sim_reach_radius_m_blue_green"] = [round(float(np.hypot(sb[0], sb[1])), 4), round(float(np.hypot(sg[0], sg[1])), 4)]
# 4. rest pose: real home median (normalized) -> rad, does the MJCF range contain it?
home = [-5.99, -98.32, 95.37, 60.32, -48.81]
qh = [(home[i] - N0[J[i]]) / UPR[J[i]] for i in range(5)]
out["real_home_rad"] = [round(float(v), 3) for v in qh]
out["real_home_inside_mjcf_range"] = {J[i]: bool(rng[J[i]][0] <= qh[i] <= rng[J[i]][1]) for i in range(5)}
json.dump(out, open(f"{W}/numbers_fk.json", "w"), indent=1)
print(json.dumps(out))
