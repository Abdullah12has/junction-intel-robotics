#!/usr/bin/env python3
"""P2-D-SIM2 W3: sensitivity of the real-plate FK check to the joint zero convention (read-only; reuses mjcf/ from fk_check.py).
Hypothesis Z1: zero at tick 2047.5 (LeRobot half-turn). Hypothesis Z2: zero at the calibrated mid-range (min+max)/2.
Writes work/P2-D-SIM2/W3/numbers_fk_zero.json only."""
import json
import math

import mujoco
import numpy as np

W = "/home/intel/intel-junction-2026/work/P2-D-SIM2/W3"
K = 2 * math.pi / 4096.0
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CAL = [r for r in json.load(open("/home/intel/intel-junction-2026/app/motion/project_robots.json"))
       if r["type"] == "SO101_Follower"][0]["payload"]["calibration"]
UPR = {j: (100.0 if j == "gripper" else 200.0) / (K * (CAL[j]["range_max"] - CAL[j]["range_min"])) for j in J}
RANGE = {j: (100.0 if j == "gripper" else 200.0) for j in J}
N0_Z1 = {j: RANGE[j] * (2047.5 - CAL[j]["range_min"]) / (CAL[j]["range_max"] - CAL[j]["range_min"]) - (0.0 if j == "gripper" else 100.0) for j in J}
N0_Z2 = {j: RANGE[j] * ((CAL[j]["range_min"] + CAL[j]["range_max"]) / 2 - CAL[j]["range_min"]) / (CAL[j]["range_max"] - CAL[j]["range_min"]) - (0.0 if j == "gripper" else 100.0) for j in J}
GRASP = np.array([0.008, 0.0, -0.093])
BASE_XY = np.array([0.0388353, 0.0388353])
model = mujoco.MjModel.from_xml_path(f"{W}/mjcf/so101.xml")
data = mujoco.MjData(model)
qadr = {j: model.jnt_qposadr[model.joint(j).id] for j in J}
rng = {j: model.jnt_range[model.joint(j).id] for j in J}


def tip(q5):
    for i, j in enumerate(J[:5]):
        data.qpos[qadr[j]] = q5[i]
    mujoco.mj_kinematics(model, data)
    b = data.body("gripper")
    return b.xpos + b.xmat.reshape(3, 3) @ GRASP


REAL = {"blue_ep7": [32.1, 5.47, 25.59, 61.85, -36.78], "green_ep8": [-54.67, 24.51, 1.91, 54.22, -55.03],
        "green_ep10": [-51.83, 47.71, -49.82, 80.08, -52.37]}
HOME = [-5.99, -98.32, 95.37, 60.32, -48.81]
out = {}
for name, N0 in (("Z1_tick2047.5", N0_Z1), ("Z2_mid_range", N0_Z2)):
    q_of = lambda u: [(u[i] - N0[J[i]]) / UPR[J[i]] for i in range(5)]  # noqa: E731
    tips = {k: tip(q_of(u)) for k, u in REAL.items()}
    b, g8, g10 = tips["blue_ep7"], tips["green_ep8"], tips["green_ep10"]
    qh = q_of(HOME)
    out[name] = {"N0_units": {j: round(N0[j], 2) for j in J[:5]},
                 "chord_blue_green8_xy_m": round(float(np.linalg.norm(b[:2] - g8[:2])), 4),
                 "chord_blue_green10_xy_m": round(float(np.linalg.norm(b[:2] - g10[:2])), 4),
                 "radius_m_blue_green8_green10": [round(float(np.hypot(*p[:2])), 4) for p in (b, g8, g10)],
                 "tip_z_m_blue_green8_green10": [round(float(p[2]), 4) for p in (b, g8, g10)],
                 "home_rad": [round(float(v), 3) for v in qh],
                 "home_inside_mjcf_range": {J[i]: bool(rng[J[i]][0] <= qh[i] <= rng[J[i]][1]) for i in range(5)}}
json.dump(out, open(f"{W}/numbers_fk_zero.json", "w"), indent=1)
print(json.dumps(out))
