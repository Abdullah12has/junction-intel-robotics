#!/usr/bin/env python3
"""Geometry hand-off for P2-R8-PICK: real plate centres in the robot base frame.

Uses the MJCF forward kinematics from sim_gate.py (TCP = site gripperframe) on the real
release poses. Reports the spread, the zero-convention variant (W1: raw tick 2048) and
the elbow +-5 deg sensitivity (the pinned MJCF notes a 5-degree calibration offset on
the elbow range). Reads data only.
Usage: python3 -I geometry_handoff.py
"""
import ast, glob, json, math, os, re, sys
import numpy as np

R = "/home/intel/intel-junction-2026"
sys.path.insert(0, R + "/app/simgate")
import sim_gate as g  # noqa: E402  (our own module; sim data is never executed)

OUT = R + "/work/P2-D-SIM2/V1/out/geometry_handoff.json"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
NAMES5 = JOINTS[:5]
TP = 2 * math.pi

rob = json.load(open(R + "/app/motion/project_robots.json"))
fol = [r for r in rob if r.get("type") == "SO101_Follower"][0]
cal = fol["payload"]["calibration"]
rmin = np.array([cal[j]["range_min"] for j in JOINTS], float)
rmax = np.array([cal[j]["range_max"] for j in JOINTS], float)
span = rmax - rmin
K_mid = (200 * 4095 / (TP * span))[:5]
K_w1 = (200 * 4096 / (TP * span))[:5]
Z_w1 = ((2048 - rmin) / span * 200 - 100)[:5]

base_el, lim = g.load_mjcf(R + "/app/simgate/ref/so101.xml")
S, E, F = g.load_real(R + "/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af")
st_txt = open(R + "/data/sim/live/2026-11-14/dataset-150-raw/source/station_scene.py", encoding="utf-8", errors="replace").read()
Tw = g.station_T(st_txt)
Tinv = np.linalg.inv(Tw)
bowl = ast.literal_eval(re.search(r"BOWL_CENTERS\s*=\s*(\{.*?\})", st_txt, re.S).group(1))


def tcp(q5):
    return g.fk_sites(base_el, dict(zip(NAMES5, q5)))["gripperframe"][:3, 3]


out = {"frame": "robot base frame = MJCF so101 'base' body (pinned Menagerie revision); x,y,z in metres",
       "tcp": "site gripperframe (same TCP the sim expert reaches with: 0.39 cm median xy error at sim releases)",
       "real_plates": {}, "sim_stated_plates_base_m": {}, "real_minus_sim_cm": {}}
for p, eps in g.REAL_SETS.items():
    nrm = np.array([g.real_release(S, E, F, e) for e in eps])
    q_mid = nrm / K_mid
    q_w1 = (nrm - Z_w1) / K_w1
    P = np.array([tcp(q) for q in q_mid])
    P_w1 = np.array([tcp(q) for q in q_w1])
    P_e_p = np.array([tcp(q + np.array([0, 0, math.radians(5), 0, 0])) for q in q_mid])
    P_e_m = np.array([tcp(q - np.array([0, 0, math.radians(5), 0, 0])) for q in q_mid])
    mean = P.mean(axis=0)
    sd = P.std(axis=0, ddof=1)
    pc = (Tinv @ np.array([bowl[p][0], bowl[p][1], 0.004, 1.0]))[:3]
    out["sim_stated_plates_base_m"][p] = pc.round(4).tolist()
    out["real_plates"][p] = {
        "episodes": eps, "n": len(eps),
        "tcp_mean_m": mean.round(4).tolist(),
        "tcp_sd_m": sd.round(4).tolist(),
        "tcp_se_m": (sd / math.sqrt(len(eps))).round(4).tolist(),
        "per_episode_tcp_m": P.round(4).tolist(),
        "zero_alt_w1_2048_mean_m": P_w1.mean(axis=0).round(4).tolist(),
        "zero_alt_shift_cm": round(float(np.linalg.norm(P_w1.mean(axis=0) - mean)) * 100, 2),
        "elbow_plus5deg_shift_cm": round(float(np.linalg.norm(P_e_p.mean(axis=0) - mean)) * 100, 2),
        "elbow_minus5deg_shift_cm": round(float(np.linalg.norm(P_e_m.mean(axis=0) - mean)) * 100, 2),
    }
    out["real_minus_sim_cm"][p] = round(float(math.hypot(mean[0] - pc[0], mean[1] - pc[1])) * 100, 2)

# top-view pixel positions from the latest gate run (green plate only; blue is confounded by bricks)
gate_files = sorted(glob.glob(R + "/data/sim/gate/2026-11-14-*.json"), key=os.path.getmtime)
if gate_files:
    gj = json.load(open(gate_files[-1]))
    cams = gj["checks"]["cameras"]
    out["top_view_px_640x480"] = {
        "green_plate_real_top": [cams["real_top"]["green"]["x_px"], cams["real_top"]["green"]["y_px"]],
        "green_plate_sim_overhead": [cams["sim_overhead"]["green"]["x_px"], cams["sim_overhead"]["green"]["y_px"]],
        "green_offset_px": gj["checks"]["top_green_offset_px"],
        "blue_plate": "not measured (bricks of similar colour confound the mask)",
        "robot_base": "not measured in pixels; Opus 13:52Z: base at top centre of the top view",
        "source_gate_json": gate_files[-1]}
with open(OUT, "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1)[:4000])
