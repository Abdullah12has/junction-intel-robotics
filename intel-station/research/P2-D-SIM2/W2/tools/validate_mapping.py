"""Validate the calibration-derived sim-radian -> Studio-normalized mapping n = s*K*q (zero offset 0).

Inputs: out/rows.json written by tools/analyse.py. Outputs: out/mapping_validation.json, stdout table.
"""
import json
import numpy as np

OUT = "/home/intel/intel-junction-2026/work/P2-D-SIM2/W2/out"
D = json.load(open(OUT + "/rows.json"))
sim, real, K = D["sim"], D["real"], D["K"]
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
BLUE_EPS = {("raw6d6", 7), ("raw6d6", 11), ("raw6d6", 12)}
GREEN_EPS = {("raw6d6", 9), ("raw6d6", 10), ("raw6d6", 14), ("raw6d6", 15)}
EXCLUDED_UNCERTAIN = {("raw6d6", 8), ("raw6d6", 13)}  # no release detected / gripper missed (team + Opus notes)
res = {"K": K}


def kvec(q_list, sign=1.0):
    q = np.asarray(q_list, float)
    k = np.array([K[j] for j in JOINTS[:5]])
    out = sign * k * q[:5]
    return out.tolist()


def med_iqr(a):
    a = np.asarray(a, float)
    return {"median": np.median(a, axis=0).round(2).tolist(),
            "p25": np.percentile(a, 25, axis=0).round(2).tolist(),
            "p75": np.percentile(a, 75, axis=0).round(2).tolist(), "n": int(len(a))}


# ---- 1. plate-drop (release) poses: sim (radians) vs real (normalized) ----
sim_place = {d: [r["place_q"] for r in sim if r["dest"] == d] for d in ("blue", "green")}
real_place = {"blue": [], "green": []}
for r in real:
    key = (r["set"], r["ep"])
    if key in EXCLUDED_UNCERTAIN or r["place_n"] is None:
        continue
    if key in BLUE_EPS:
        real_place["blue"].append(r["place_n"][:5])
    if key in GREEN_EPS:
        real_place["green"].append(r["place_n"][:5])
res["real_place_n_blue_eps"] = [(r["set"], r["ep"]) for r in real if (r["set"], r["ep"]) in BLUE_EPS]
res["real_place_n_green_eps"] = [(r["set"], r["ep"]) for r in real if (r["set"], r["ep"]) in GREEN_EPS]
for sgn in (1.0, -1.0):
    tot = {}
    for d in ("blue", "green"):
        sim_n = [kvec(q, sgn) for q in sim_place[d]]
        s_med = np.median(np.asarray(sim_n), axis=0)
        r_med = np.median(np.asarray(real_place[d]), axis=0)
        tot[d] = {"sim_median_n": s_med.round(2).tolist(), "real_median_n": r_med.round(2).tolist(),
                  "residual_real_minus_sim": (r_med - s_med).round(2).tolist(),
                  "sim_iqr_n": med_iqr(sim_n), "real_n_per_ep": np.round(np.asarray(real_place[d]), 2).tolist()}
    res[f"place_sign_{int(sgn)}"] = tot
    absres = np.abs(np.r_[tot["blue"]["residual_real_minus_sim"], tot["green"]["residual_real_minus_sim"]])
    res[f"place_sign_{int(sgn)}_abs_residual_sum"] = round(float(absres.sum()), 2)
    res[f"place_sign_{int(sgn)}_pan_abs_residual"] = round(float(max(abs(tot["blue"]["residual_real_minus_sim"][0]), abs(tot["green"]["residual_real_minus_sim"][0]))), 2)

# ---- 2. grasp (pick) poses: sim (ali1 pose) vs real (grasp_frame pose) ----
sim_grasp = [r["grasp_q"] for r in sim]
real_grasp = [r["grasp_n"] for r in real if r["grasp_n"] is not None and (r["set"], r["ep"]) not in EXCLUDED_UNCERTAIN]
res["sim_grasp_n_s1"] = med_iqr([kvec(q, 1.0) for q in sim_grasp])
res["real_grasp_n"] = med_iqr([g[:5] for g in real_grasp]) if real_grasp else None
res["real_grasp_n_count"] = len(real_grasp)

# ---- 3. home pose: real first frame vs sim HOME (model-only) ----
real_home = np.asarray([r["home_n"] for r in real])
res["real_home_n_median"] = np.median(real_home, axis=0).round(2).tolist()
res["sim_HOME_as_n"] = [round(v, 2) for v in kvec([0, -0.5, 0.8, 0.6, 0, 0], 1.0)] + [None]

# ---- 4. distributions over all frames is not stored per-frame; use per-episode vmax / p99 / jerk ----
for name, key in (("vmax", "vmax_radps"), ("vp99", "vp99_radps")):
    sv = np.asarray([[K[j] * x for j, x in zip(JOINTS, r[key])] for r in sim])
    res[f"sim_{name}_n_s_median_over_eps"] = np.median(sv, axis=0).round(2).tolist()
    res[f"sim_{name}_n_s_p90_over_eps"] = np.percentile(sv, 90, axis=0).round(2).tolist()
rv = np.asarray([r["vmax_n_s"] for r in real if r["set"] == "raw6d6"])
rp = np.asarray([r["vp99_n_s"] for r in real if r["set"] == "raw6d6"])
res["real_vmax_n_s_median_over_eps"] = np.median(rv, axis=0).round(2).tolist()
res["real_vmax_n_s_p90_over_eps"] = np.percentile(rv, 90, axis=0).round(2).tolist()
res["real_vp99_n_s_median_over_eps"] = np.median(rp, axis=0).round(2).tolist()
res["real_fps_est"] = sorted({r["fps_est"] for r in real})
sj = np.asarray([[K[j] * x for j, x in zip(JOINTS, r["jerk_rms"])] for r in sim])
res["sim_jerk_rms_n_s3_median"] = np.median(sj, axis=0).round(1).tolist()
rj = np.asarray([r["jerk_rms_n"] for r in real if r["set"] == "raw6d6"])
res["real_jerk_rms_n_s3_median"] = np.median(rj, axis=0).round(1).tolist()

# ---- 5. gripper (not mapped by convention; report raw) ----
gq = [r["grip_closed_on_brick_rad"] for r in sim if r["grip_closed_on_brick_rad"] is not None]
res["sim_grip_closed_on_brick_rad_median"] = round(float(np.median(gq)), 4)
res["sim_grip_closed_on_brick_rad_iqr"] = [round(float(np.percentile(gq, 25)), 4), round(float(np.percentile(gq, 75)), 4)]
res["real_grip_min_max_n"] = [[round(r["grip_min"], 2), round(r["grip_max"], 2)] for r in real if r["set"] == "raw6d6"]

json.dump(res, open(OUT + "/mapping_validation.json", "w"), indent=1)
print("SIGN+1 residual sum", res["place_sign_1_abs_residual_sum"], "pan", res["place_sign_1_pan_abs_residual"])
print("SIGN-1 residual sum", res["place_sign_-1_abs_residual_sum"], "pan", res["place_sign_-1_pan_abs_residual"])
for sgn in ("1", "-1"):
    for d in ("blue", "green"):
        t = res[f"place_sign_{sgn}"][d]
        print("SIGN", sgn, d, "sim_med", t["sim_median_n"], "real_med", t["real_median_n"], "resid", t["residual_real_minus_sim"], "sim_n", t["sim_iqr_n"]["n"], "real_n", len(t["real_n_per_ep"]))
print("REAL_BLUE_EPS", res["real_place_n_blue_eps"], "REAL_GREEN_EPS", res["real_place_n_green_eps"])
print("SIM_GRASP_N med", res["sim_grasp_n_s1"]["median"], "iqr25", res["sim_grasp_n_s1"]["p25"], "iqr75", res["sim_grasp_n_s1"]["p75"])
print("REAL_GRASP_N med", res["real_grasp_n"]["median"] if res["real_grasp_n"] else None, "count", res["real_grasp_n_count"])
print("REAL_HOME_N median", res["real_home_n_median"], "SIM_HOME_n", res["sim_HOME_as_n"])
print("SIM_VMAX_N_S med", res["sim_vmax_n_s_median_over_eps"], "p90", res["sim_vmax_n_s_p90_over_eps"])
print("REAL_VMAX_N_S med", res["real_vmax_n_s_median_over_eps"], "p90", res["real_vmax_n_s_p90_over_eps"])
print("SIM_VP99 med", res["sim_vp99_n_s_median_over_eps"], "REAL_VP99 med", res["real_vp99_n_s_median_over_eps"])
print("SIM_JERK med", res["sim_jerk_rms_n_s3_median"], "REAL_JERK med", res["real_jerk_rms_n_s3_median"])
print("SIM_GRIP_CLOSED rad med", res["sim_grip_closed_on_brick_rad_median"], res["sim_grip_closed_on_brick_rad_iqr"])
print("REAL_FPS", res["real_fps_est"])
