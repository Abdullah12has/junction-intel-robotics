#!/usr/bin/env python3
"""Adds the three-column summary (friend sim | real | twin) to compare.json; twin status from twin_status.json. Data only.
Usage: python3 -I finalize.py OUT"""
import json, sys
OUT = sys.argv[1].rstrip("/")
C = json.load(open(OUT + "/compare.json"))
M = json.load(open(OUT + "/metrics.json"))["metrics"]
TW = C["twin"]
pend = "PENDING (twin data dirs empty at %s)" % TW["checked_utc"] if TW["status"].startswith("EMPTY") else "PRESENT: see twin block"
mp = M["top_plate_offset_px_real_minus_sim"]["magnitude"]
mm = M["top_plate_offset_mm_image_via_sim_model"]["magnitude"]
sp = M["top_plate_spacing_mm"]
cl = M["close_plate_offset_px_real_minus_sim_median"]
br = M["brightness"]
w = M["wrist"]
ju = M["joint_units"]
mo = M["motion_speed_p95_units_s"]
du = M["episode_duration_s"]
g = M["gate_rerun"]["decisions"]
rows = [
    ["top view plate offset, real minus sim (blue / green), px", "%s / %s" % (mp["blue"], mp["green"]), "reference", pend],
    ["top view plate offset via sim camera model (blue / green), mm", "%s / %s" % (mm["blue"], mm["green"]), "reference", pend],
    ["plate spacing, mm", "truth %s; image %s" % (sp["sim_world_truth_from_bowl_centres"], sp["sim_image_via_model"]),
     "image %s; FK gate %s" % (sp["real_image_via_sim_model"], M["fk_plate_spacing_mm_gate"]["real"]), pend],
    ["close view green plate offset, px", "%s (medians); %s (gate)" % (cl["green_magnitude"], cl["gate_green_median"]), "reference", pend],
    ["top view mean luma (darker %)", "%s vs %s (%s%% darker)" % (br["top_luma"][0], br["top_luma"][1], br["top_sim_darker_pct"]), "reference", pend],
    ["close view mean luma (darker %)", "%s vs %s (%s%% darker)" % (br["close_luma"][0], br["close_luma"][1], br["close_sim_darker_pct"]), "reference", pend],
    ["wrist frames at MJCF limit / real max", "%s of frames at %s units" % (w["sim_frac_within_0p5_of_limit"], w["mjcf_limit_norm"]), "max %s units" % w["real_max_norm_joints_csv"], pend],
    ["gripper open max, units", str(ju["gripper_open_max"][0]), str(ju["gripper_open_max"][1]), pend],
    ["p95 speed, lift joint, units/s", str(mo["sim"][1]), str(mo["real_joints_csv"][1]), pend],
    ["episode duration, s", str(du["sim"]), "Studio %s; loop %s" % (du["real_studio_mean"], du["real_joints_csv_loop_mean"]), pend],
    ["OD-10 routing contradictions", "%s of %s" % (M["od10_contradictions"], M["versions"]["episodes"]), "not measured here", pend],
    ["gate decision (SmolVLA / ACT)", "%s / %s" % (g["a_smolvla_cotraining"], g["b_act_pretrain_augment"]), "n/a", pend],
]
C["three_column"] = {"columns": ["measure", "friend_sim", "real", "twin"], "rows": rows,
                     "twin_note": "Twin column cannot be filled: R/twin/data/{calib,model,captures,episodes/raw,episodes/lerobot} had no files at the last check. Re-run twin_check.py then finalize.py when they are populated."}
with open(OUT + "/compare.json", "w") as fh:
    json.dump(C, fh, indent=1, default=str)
print("three_column rows:", len(rows), "twin status:", TW["status"][:60])
