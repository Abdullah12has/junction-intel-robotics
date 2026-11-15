#!/usr/bin/env python3
"""Builds metrics.json (stage friend-sim-v2) from compare.json and the re-run gate JSON. Data only.
Usage: python3 -I make_metrics.py OUT"""
import json, math, sys, time
import numpy as np
OUT = sys.argv[1].rstrip("/")
C = json.load(open(OUT + "/compare.json"))
G = json.load(open(OUT + "/gate/2026-11-14-95fa40dbf3.json"))
W0, H0 = 640, 480


def make_cam(pos, tgt, fovy):
    p = np.array(pos, float)
    z = p - np.array(tgt, float)
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return {"p": p, "x": x, "y": y, "z": z, "f": (H0 / 2) / math.tan(math.radians(fovy) / 2)}


def back(cam, u, v, zp):
    dc = np.array([(u - W0 / 2) / cam["f"], -(v - H0 / 2) / cam["f"], -1.0])
    dw = dc[0] * cam["x"] + dc[1] * cam["y"] + dc[2] * cam["z"]
    return cam["p"] + ((zp - cam["p"][2]) / dw[2]) * dw


cm = C["camera_models"]["sim_overhead"]
CT = make_cam(cm["pos"], cm["target"], cm["fovy"])
FL = C["bowl_floor_m"]
TD = C["plates"]["top"]["sim_detected_median"]
TR = C["plates"]["top"]["real_detected_median"]
sim_sp = float(np.linalg.norm(back(CT, TD["blue"]["cx"], TD["blue"]["cy"], FL)[:2] - back(CT, TD["green"]["cx"], TD["green"]["cy"], FL)[:2]) * 1000)
real_sp = float(np.linalg.norm(back(CT, TR["blue"]["cx"], TR["blue"]["cy"], FL)[:2] - back(CT, TR["green"]["cx"], TR["green"]["cy"], FL)[:2]) * 1000)
geo = G["checks"]["geometry"]
fk_sim = geo["plate_centre_base_m"]
fk_real = geo["real_tcp_mean_base_m"]
fk_sim_sp = float(np.linalg.norm(np.array(fk_sim["blue"][:2]) - np.array(fk_sim["green"][:2])) * 1000)
fk_real_sp = float(np.linalg.norm(np.array(fk_real["blue"][:2]) - np.array(fk_real["green"][:2])) * 1000)
mm = C["plates"]["top"]["real_minus_sim_mm_via_sim_model"]
px = C["plates"]["top"]["real_minus_sim_px"]
mag_px = {c: round(float(math.hypot(*px[c])), 2) for c in px}
mag_mm = {c: round(float(math.hypot(*mm[c])), 2) for c in mm}
close_px = C["plates"]["close"]["real_minus_sim_px"]
br = C["brightness"]
J = C["joints"]
M = C["motion"]
n_real = [r["n"] for r in M["real_episode_meta"] if r["used"]]
dt_real = [r["dt_med_s"] for r in M["real_episode_meta"] if r["used"]]
real_loop_dur = float(np.mean([a * b for a, b in zip(n_real, dt_real)]))
metrics = {
    "versions": {"live_equals_2026_11_14": C["version_identity"]["live_equals_2026_11_14"],
                 "episodes": C["routing"]["episodes"], "manifest_entries_verified": G["provenance"]["ok"],
                 "manifest_entries": G["provenance"]["entries"]},
    "gate_rerun": {"identical_to_existing": C["gate_rerun"]["identical_to_existing_gate_json"], "decisions": C["gate_rerun"]["decisions"]},
    "top_plate_offset_px_real_minus_sim": {"blue": px["blue"], "green": px["green"], "magnitude": mag_px},
    "top_plate_offset_mm_image_via_sim_model": {"blue": mm["blue"], "green": mm["green"], "magnitude": mag_mm},
    "top_plate_spacing_mm": {"sim_world_truth_from_bowl_centres": round(float(np.linalg.norm(np.array(C["bowl_centres_world_m"]["green"]) - np.array(C["bowl_centres_world_m"]["blue"])) * 1000), 1),
                             "sim_image_via_model": round(sim_sp, 1), "real_image_via_sim_model": round(real_sp, 1)},
    "fk_plate_spacing_mm_gate": {"sim": round(fk_sim_sp, 1), "real": round(fk_real_sp, 1)},
    "fk_plate_offset_cm_gate": {"blue": geo["real_to_sim_plate_cm"]["blue"], "green": geo["real_to_sim_plate_cm"]["green"]},
    "close_plate_offset_px_real_minus_sim_median": {"blue": close_px["blue"], "green": close_px["green"],
                                                    "green_magnitude": round(float(math.hypot(*close_px["green"])), 2),
                                                    "gate_green_median": G["checks"]["close_green_offset_px"]},
    "top_camera_green_offset_px_gate": G["checks"]["top_green_offset_px"],
    "brightness": {"top_luma": [br["top"]["sim"]["luma"], br["top"]["real"]["luma"]], "top_sim_darker_pct": br["top"]["sim_darker_pct_vs_real"],
                   "close_luma": [br["close"]["sim"]["luma"], br["close"]["real"]["luma"]], "close_sim_darker_pct": br["close"]["sim_darker_pct_vs_real"]},
    "joint_units": {"K_units_per_rad": J["K_units_per_rad"], "sim_first_frame_norm": J["sim_first_frame_norm_median"],
                    "real_first_row_norm": J["real_first_row_norm_median"], "gate_release_pan_residual_deg": G["checks"]["release"]["pan_residual_deg"],
                    "gripper_span_ratio_gate": G["checks"]["gripper"]["span_ratio_sim_over_real"], "gripper_open_max": [J["gripper_sim_open_max"], J["gripper_real_open_max"]]},
    "wrist": {"mjcf_limit_norm": J["wrist_flex_mjcf_limit_norm"], "sim_frac_within_0p5_of_limit": J["wrist_flex_sim_frac_within_0p5_of_limit"],
              "sim_frac_gate_method": G["checks"]["units"]["wrist_pinned_frac_sim"], "real_max_norm_joints_csv": J["real_joints_csv_range_norm"]["max"][3]},
    "rest_like_frac_sim": G["checks"]["units"]["rest_like_frac_sim"],
    "motion_speed_p95_units_s": {"sim": M["sim"]["speed_p95"], "real_joints_csv": M["real_joints_csv"]["speed_p95"]},
    "motion_still_frac": {"sim": M["sim"]["still_frac_below_0p5_units_per_s"], "real_joints_csv": M["real_joints_csv"]["still_frac_below_0p5_units_per_s"]},
    "episode_duration_s": {"sim": round(361 / 30, 2), "real_joints_csv_loop_mean": round(real_loop_dur, 1), "real_studio_mean": round(10158 / 16 / 30, 2)},
    "od10_contradictions": C["routing"]["od10_contradictions_sim"],
}
previous = {"source": "kb ledger DECISIONS.md D-069 (14:52Z)", "close_camera_offset_px_range": [32, 72], "sim_darker_pct": 32,
            "top_camera_offset_px_within": 8}
delta = {
    "close_camera_offset_px": {"new_gate_median": G["checks"]["close_green_offset_px"], "previous_range": [32, 72],
                               "change_vs_previous_max_px": round(G["checks"]["close_green_offset_px"] - 72, 1),
                               "label": "declined (above D-069 range); method differs"},
    "sim_darker_pct": {"new_top": br["top"]["sim_darker_pct_vs_real"], "previous": 32,
                       "change_pp": round(br["top"]["sim_darker_pct_vs_real"] - 32, 2), "label": "declined; D-069 method not recorded, not reproduced"},
    "top_camera_offset_px": {"new_gate_green": G["checks"]["top_green_offset_px"], "new_mean_blue_green_image": round((mag_px["blue"] + mag_px["green"]) / 2, 2),
                             "previous_within": 8, "label": "within D-069 bound; method differs"},
    "note": "live data identical to the 2026-11-14 copy (verified), so these deltas reflect re-measurement only, not a change by the friend",
}
out = {"stage": "friend-sim-v2", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "metrics": metrics,
       "previous": previous, "delta": delta}
with open(OUT + "/metrics.json", "w") as fh:
    json.dump(out, fh, indent=1, default=str)
print(json.dumps({"sim_spacing_image": round(sim_sp, 1), "real_spacing_image": round(real_sp, 1), "fk_sim_sp": round(fk_sim_sp, 1), "fk_real_sp": round(fk_real_sp, 1),
                  "top_px": mag_px, "top_mm": mag_mm, "close_green_px_median": round(float(math.hypot(*close_px["green"])), 2), "real_loop_dur": round(real_loop_dur, 1),
                  "delta": delta}, indent=1, default=str)[:4000])
