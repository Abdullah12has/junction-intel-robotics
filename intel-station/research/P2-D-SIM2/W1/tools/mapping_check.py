#!/usr/bin/env python3
"""P2-D-SIM2 W1 (read-only): candidate sim-radian <-> Studio normalized mapping from the calibration,
checked against real plate-release poses, joint excursions, speeds and the home pose.

Usage: python3 -I mapping_check.py <sim_summary.json> <real_summary.json> <project_robots.json> <out dir>
Model (assumption, stated in report): normalized = scale * (raw - range_min) / (range_max - range_min) - offset,
scale 200 and offset 100 for arm joints (RANGE_M100_100), scale 100 and offset 0 for the gripper (RANGE_0_100);
raw = 4096 ticks per turn; sim zero assumed at raw 2048 (middle of the calibrated turn).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

SIM = json.loads(Path(sys.argv[1]).read_text())
REAL = json.loads(Path(sys.argv[2]).read_text())
ROBOTS = json.loads(Path(sys.argv[3]).read_text())
OUT = Path(sys.argv[4])
OUT.mkdir(parents=True, exist_ok=True)
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TICKS_PER_RAD = 4096 / (2 * math.pi)
# Real plate releases. Eps 0-6 are hold / put-down episodes (not plate drops), so they are excluded.
# Ep 13 is pan-classified blue but its gripper only closes to ~11 (team note 14:00Z), so it is kept apart.
BLUE_STRICT = ["7", "11", "12"]
BLUE_FLAG = ["13"]
GREEN = ["8", "10", "14", "15"]  # ep 9 excluded: release-onset detector found no gripper rise (rise=0)

follower = next(r for r in ROBOTS if r["name"] == "SO101 Follower")
cal = follower["payload"]["calibration"]
cmap = {}
for j in JOINTS:
    c = cal[j]
    span = c["range_max"] - c["range_min"]
    scale = 100.0 if j == "gripper" else 200.0
    slope = scale * TICKS_PER_RAD / span
    n0 = scale * (2048 - c["range_min"]) / span - (0.0 if j == "gripper" else 100.0)
    cmap[j] = {"homing_offset": c["homing_offset"], "range_min": c["range_min"], "range_max": c["range_max"],
               "slope_units_per_rad": slope, "n_at_zero_rad": n0}


def to_rad(n, j):
    return (n - cmap[j]["n_at_zero_rad"]) / cmap[j]["slope_units_per_rad"]


def to_norm(th, j):
    return cmap[j]["slope_units_per_rad"] * th + cmap[j]["n_at_zero_rad"]


RB = REAL["release_pose_norm_by_plate"]["blue"]["per_episode"]
RG = REAL["release_pose_norm_by_plate"]["green"]["per_episode"]


def stats(keys, table, idx):
    vals = [table[k][idx] for k in keys]
    return float(np.mean(vals)), float(np.std(vals))


plates = {}
for idx, j in enumerate(JOINTS):
    th_b = SIM["release_pose_rad_by_destination"]["blue"]["mean"][j]
    th_g = SIM["release_pose_rad_by_destination"]["green"]["mean"][j]
    rb, sb = stats(BLUE_STRICT, RB, idx)
    rg, sg = stats(GREEN, RG, idx)
    rbf = float(RB[BLUE_FLAG[0]][idx])
    pb, pg = to_norm(th_b, j), to_norm(th_g, j)
    slope_fit = (rb - rg) / (th_b - th_g)
    plates[j] = {
        "sim_blue_rad": th_b, "sim_green_rad": th_g,
        "real_blue_norm_mean": rb, "real_blue_norm_sd": sb, "real_blue_ep13_norm": rbf,
        "real_green_norm_mean": rg, "real_green_norm_sd": sg,
        "calib_pred_blue_norm": pb, "calib_pred_green_norm": pg,
        "resid_blue_units": rb - pb, "resid_green_units": rg - pg,
        "resid_blue_deg": (rb - pb) / cmap[j]["slope_units_per_rad"] * 180 / math.pi,
        "resid_green_deg": (rg - pg) / cmap[j]["slope_units_per_rad"] * 180 / math.pi,
        "two_plate_fit_slope_units_per_rad": slope_fit,
        "two_plate_fit_slope_over_calib": slope_fit / cmap[j]["slope_units_per_rad"],
        "two_plate_fit_intercept_units": rb - slope_fit * th_b,
    }

excursion = {}
for j in JOINTS:
    real_lo, real_hi = to_rad(REAL["state_min_norm"][j], j), to_rad(REAL["state_max_norm"][j], j)
    sim_lo, sim_hi = SIM["state_min_rad"][j], SIM["state_max_rad"][j]
    excursion[j] = {"real_rad_min_max": [real_lo, real_hi], "sim_rad_min_max": [sim_lo, sim_hi],
                    "sim_span_over_real_span": (sim_hi - sim_lo) / (real_hi - real_lo)}

home_norm = {"shoulder_pan": -6.6, "shoulder_lift": -98.1, "elbow_flex": 99.4, "wrist_flex": 54.3,
             "wrist_roll": -53.1, "gripper": 25.7}  # live home pose read through Studio (FACTS-LIVE 14:13Z)
home_rad = {j: to_rad(home_norm[j], j) for j in JOINTS}
sim_home = dict(zip(JOINTS, [0, -0.5, 0.8, 0.6, 0, 0.5]))  # so101_env.py / station_scene.py HOME (model-only)

speed = {j: {"real_vp99_rad_s": REAL["vp99_norm_per_s_median"][j] / cmap[j]["slope_units_per_rad"],
             "sim_vp99_rad_s": SIM["vp99_rad_s_median"][j],
             "real_vmax_rad_s": REAL["vmax_norm_per_s"][j] / cmap[j]["slope_units_per_rad"],
             "sim_vmax_rad_s": SIM["vmax_rad_s_over_episodes"][j]} for j in JOINTS}

gripper_closed = {"real_grasp_norm_median": REAL["gripper_grasp_norm_median"],
                  "real_grasp_rad": to_rad(REAL["gripper_grasp_norm_median"], "gripper"),
                  "sim_release_state_rad_blue": SIM["release_pose_rad_by_destination"]["blue"]["mean"]["gripper"],
                  "sim_release_state_rad_green": SIM["release_pose_rad_by_destination"]["green"]["mean"]["gripper"]}

result = {"calibration_model": cmap, "plate_release_comparison": plates, "excursions": excursion,
          "home_pose_rad_from_live_norm": home_rad, "sim_HOME_rad": sim_home, "speeds": speed,
          "gripper_closed_check": gripper_closed,
          "episodes_used": {"blue_strict": BLUE_STRICT, "blue_flag": BLUE_FLAG, "green": GREEN}}
(OUT / "mapping_check.json").write_text(json.dumps(result, indent=1, default=float) + "\n")

print("joint slope(units/rad) n0 | blue: sim_rad real_mean pred resid_units resid_deg | green: sim_rad real pred resid_units resid_deg | fit/calib")
for j in JOINTS:
    c, p = cmap[j], plates[j]
    print(f"{j:14s} {c['slope_units_per_rad']:7.2f} {c['n_at_zero_rad']:6.2f} | {p['sim_blue_rad']:+.3f} {p['real_blue_norm_mean']:+7.2f} "
          f"{p['calib_pred_blue_norm']:+7.2f} {p['resid_blue_units']:+7.2f} {p['resid_blue_deg']:+7.1f} | "
          f"{p['sim_green_rad']:+.3f} {p['real_green_norm_mean']:+7.2f} {p['calib_pred_green_norm']:+7.2f} "
          f"{p['resid_green_units']:+7.2f} {p['resid_green_deg']:+7.1f} | {p['two_plate_fit_slope_over_calib']:+.3f}")
print(json.dumps({"excursion_sim_over_real_span": {j: round(excursion[j]["sim_span_over_real_span"], 3) for j in JOINTS},
                  "real_excursion_rad": {j: [round(a, 3) for a in excursion[j]["real_rad_min_max"]] for j in JOINTS},
                  "sim_excursion_rad": {j: [round(a, 3) for a in excursion[j]["sim_rad_min_max"]] for j in JOINTS},
                  "home_real_rad": {j: round(home_rad[j], 3) for j in JOINTS},
                  "speed_real_vs_sim_vp99": {j: [round(speed[j]["real_vp99_rad_s"], 3), round(speed[j]["sim_vp99_rad_s"], 3)] for j in JOINTS},
                  "gripper": gripper_closed}, indent=1))
