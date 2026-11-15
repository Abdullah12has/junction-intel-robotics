"""Post-run patch for the friend-sim-v2 deliverables (E1), v2.

Run after tools/sim_real_compare.py and tools/make_metrics.py. Reads only files in E1 and
rewrites only compare.json and metrics.json:
  - twin block: NOT COMPARED (R/twin deleted at 19:48Z, D-102); last read-only check copied
    from twin_status.json (19:43Z, all twin data dirs empty).
  - three_column removed if present; comparison_table (friend_sim | real) built from the JSON values.
  - motion.sim_episode_frames added from logs/L17-episode-frames.txt (trajectory.npz lengths).
  - metrics.episode_duration_s corrected: make_metrics.py hard-codes round(361/30, 2) = 12.03.
Reads nothing under R/twin and writes nothing outside E1.
"""
import datetime
import hashlib
import json
import math
import os

E1 = "/home/intel/intel-junction-2026/work/P3-FRIEND-SIM/E1"
P = {
    "cmp": os.path.join(E1, "compare.json"),
    "met": os.path.join(E1, "metrics.json"),
    "twin": os.path.join(E1, "twin_status.json"),
    "l17": os.path.join(E1, "logs", "L17-episode-frames.txt"),
    "gate": os.path.join(E1, "gate", "2026-11-14-95fa40dbf3.json"),
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def dump(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, indent=1) + "\n")


def join(vals, fmt="%.2f"):
    return "[" + ", ".join(fmt % v for v in vals) + "]"


def xy(d):
    return "%.1f, %.1f (n=%d)" % (d["cx"], d["cy"], d["n"])


now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
C = load(P["cmp"])
M = load(P["met"])
T = load(P["twin"])
L = load(P["l17"])
G = load(P["gate"])
before = {"compare.json": sha256_file(P["cmp"]), "metrics.json": sha256_file(P["met"])}

# 1. sim frame distribution (trajectory.npz lengths, all attempts)
frames_keys = ["trajectory_files", "metadata_files", "frames_min", "frames_max", "frames_median",
               "frames_mean", "frames_total", "duration_s_min", "duration_s_median",
               "duration_s_mean", "duration_s_max", "count_by_frames", "npz_meta_mismatches"]
C["motion"]["sim_episode_frames"] = {k: L[k] for k in frames_keys}
C["motion"]["sim_episode_frames_source"] = ("logs/L17-episode-frames.txt from tools/episode_frames_check.py "
                                            "(read-only; trajectory.npz length vs metadata frames)")

# 2. twin: not compared
C["twin"] = {
    "status": ("NOT COMPARED. R/twin was deleted at 2026-11-14T19:48Z (D-102). The coordinator (19:49Z) asked "
               "for the friend sim against the real recordings only; the team twin will be R/sim/twin, compared "
               "by Opus once it passes its gates."),
    "last_read_only_check": {"result": T.get("status"), "files": T.get("files"), "py_files": T.get("py_files")},
}

# 3. episode duration (make_metrics.py hard-codes 361/30)
n_att = L["trajectory_files"]
M["metrics"]["episode_duration_s"] = {
    "sim": round(L["frames_median"] / 30.0, 2),
    "sim_mean": round(L["frames_total"] / n_att / 30.0, 2),
    "sim_min": round(L["frames_min"] / 30.0, 2),
    "sim_max": round(L["frames_max"] / 30.0, 2),
    "sim_note": ("median over 150 attempts from trajectory.npz lengths (L17); replaces 12.03 = 361/30, "
                 "hard-coded in make_metrics.py line 76"),
    "real_joints_csv_loop_mean": M["metrics"]["episode_duration_s"]["real_joints_csv_loop_mean"],
    "real_studio_mean": M["metrics"]["episode_duration_s"]["real_studio_mean"],
}
ed = M["metrics"]["episode_duration_s"]
mx = M["metrics"]

# 4. comparison table, built from the values
pl, br, jt, mo, gr = C["plates"], C["brightness"], C["joints"], C["motion"], C["gate_rerun"]
top_real_mm = pl["top"]["real_minus_sim_mm_via_sim_model"]
rows = []
for colour in ("blue", "green"):
    d = pl["top"]["real_minus_sim_px"][colour]
    rows.append(["top view, %s plate centroid (px, x y)" % colour,
                 xy(pl["top"]["sim_detected_median"][colour]), xy(pl["top"]["real_detected_median"][colour]),
                 "real minus sim %.2f, %.2f px (%.2f px)" % (d[0], d[1], math.hypot(d[0], d[1]))])
rows.append(["top view, plate offset through sim camera model (mm)", "n/a", "n/a",
             "magnitude blue %.2f mm, green %.2f mm" % (math.hypot(*top_real_mm["blue"]),
                                                       math.hypot(*top_real_mm["green"]))])
rows.append(["top view, plate spacing",
             "%.1f px; truth %.1f mm; image %.1f mm" % (pl["top"]["plate_spacing_px"]["sim"],
                                                       mx["top_plate_spacing_mm"]["sim_world_truth_from_bowl_centres"],
                                                       mx["top_plate_spacing_mm"]["sim_image_via_model"]),
             "%.1f px; image %.1f mm; gate FK %.1f mm (between mean real release points)"
             % (pl["top"]["plate_spacing_px"]["real"], mx["top_plate_spacing_mm"]["real_image_via_sim_model"],
                mx["fk_plate_spacing_mm_gate"]["real"]),
             "real/sim ratio %.4f in px" % pl["top"]["plate_spacing_px"]["real_over_sim"]])
cg = mx["close_plate_offset_px_real_minus_sim_median"]
rows.append(["close view, green plate centroid (median px)",
             xy(pl["close"]["sim_detected_median"]["green"]), xy(pl["close"]["real_detected_median"]["green"]),
             "real minus sim %.2f px (magnitude); gate median %.1f px" % (cg["green_magnitude"], cg["gate_green_median"])])
cb_s, cb_r = pl["close"]["sim_detected_median"]["blue"], pl["close"]["real_detected_median"]["blue"]
rows.append(["close view, blue plate centroid (median px)",
             "%.1f, %.1f (x sd %.1f px)" % (cb_s["cx"], cb_s["cy"], cb_s["cx_sd"]),
             "%.1f, %.1f (x sd %.1f px)" % (cb_r["cx"], cb_r["cy"], cb_r["cx_sd"]),
             "not used for the offset (real x sd %.0f px)" % cb_r["cx_sd"]])
for view, label in (("top", "top view"), ("close", "close view")):
    b = br[view]
    rows.append(["%s, mean luma" % label, "%.2f (%d frames)" % (b["sim"]["luma"], b["sim"]["frames"]),
                 "%.2f (%d frames)" % (b["real"]["luma"], b["real"]["frames"]),
                 "sim %.2f%% darker" % b["sim_darker_pct_vs_real"]])
rows.append(["mean saturation S, top / close",
             "%.2f / %.2f" % (br["top"]["sim"]["S"], br["close"]["sim"]["S"]),
             "%.2f / %.2f" % (br["top"]["real"]["S"], br["close"]["real"]["S"]),
             "sim minus real: top %.2f, close %.2f" % (br["top"]["sim"]["S"] - br["top"]["real"]["S"],
                                                      br["close"]["sim"]["S"] - br["close"]["real"]["S"])])
rows.append(["release pan, blue plate (units, median)", "%.2f" % G["checks"]["release"]["sim_pan_norm_median"]["blue"],
             "%.2f" % G["real"]["release_pan_norm_median"]["blue"],
             "residual %.1f deg (gate tolerance 3 deg)" % gr["release_pan_residual_deg"]["blue"]])
rows.append(["release pan, green plate (units, median)", "%.2f" % G["checks"]["release"]["sim_pan_norm_median"]["green"],
             "%.2f" % G["real"]["release_pan_norm_median"]["green"],
             "residual %.1f deg" % gr["release_pan_residual_deg"]["green"]])
sr, rr = jt["sim_range_norm"], jt["real_joints_csv_range_norm"]
rows.append(["lift range (units)", "%.2f to %.2f" % (sr["min"][1], sr["max"][1]),
             "%.2f to %.2f" % (rr["min"][1], rr["max"][1]),
             "sim minimum %.1f vs real rest %.1f (gate)" % (sr["min"][1], G["real"]["rest_ref_norm"][1])])
rows.append(["elbow range (units)", "%.2f to %.2f" % (sr["min"][2], sr["max"][2]),
             "%.2f to %.2f" % (rr["min"][2], rr["max"][2]),
             "sim maximum %.1f vs real rest %.1f (gate)" % (sr["max"][2], G["real"]["rest_ref_norm"][2])])
rows.append(["wrist_flex range (units)", "%.2f to %.2f (MJCF limit %.2f)" % (sr["min"][3], sr["max"][3],
                                                                             jt["wrist_flex_mjcf_limit_norm"]),
             "%.2f to %.2f" % (rr["min"][3], rr["max"][3]),
             "%.2f%% of sim frames within 0.5 units of limit (gate method %.2f%%); real frames at or above 98: %.1f%%"
             % (jt["wrist_flex_sim_frac_within_0p5_of_limit"] * 100, G["checks"]["units"]["wrist_pinned_frac_sim"] * 100,
                jt["wrist_flex_real_frac_ge_98"] * 100)])
rows.append(["wrist roll median (units)", "%.2f" % sr["median"][4], "%.2f" % rr["median"][4],
             "not separable from these data (grasp strategy or convention)"])
rows.append(["gripper open max / min (units)", "%.2f / %.2f" % (jt["gripper_sim_open_max"], jt["gripper_sim_min"]),
             "%.2f / %.2f" % (jt["gripper_real_open_max"], jt["gripper_real_min"]),
             "span ratio %.3f (gate tolerance 10%%)" % gr["gripper"]["span_ratio_sim_over_real"]])
rows.append(["start pose, first-row median (units)", join(jt["sim_first_frame_norm_median"]),
             join(jt["real_first_row_norm_median"]), "different start convention; not a zero-convention test"])
sp, rp = mo["sim"]["speed_p95"], mo["real_joints_csv"]["speed_p95"]
rows.append(["p95 speed, five arm joints (units/s)", join(sp[:5]), join(rp[:5]),
             "sim/real " + join([s / r for s, r in zip(sp[:5], rp[:5])])])
rows.append(["p95 speed, gripper (units/s)", "%.2f" % sp[5], "%.2f" % rp[5],
             "no ratio (real p95 is 0)" if rp[5] == 0 else "ratio %.2f" % (sp[5] / rp[5])])
rows.append(["still fraction below 0.5 units/s, six joints (pan..gripper)",
             join(mo["sim"]["still_frac_below_0p5_units_per_s"]),
             join(mo["real_joints_csv"]["still_frac_below_0p5_units_per_s"]), "sim moves more of the time"])
jr_s, jr_r = mo["sim"]["jerk_rms"][:5], mo["real_joints_csv"]["jerk_rms"][:5]
rows.append(["jerk RMS, five arm joints", join(jr_s, "%.0f"), join(jr_r, "%.0f"),
             "sim/real " + join([s / r for s, r in zip(jr_s, jr_r)], "%.1f")
             + "; real may be inflated by encoder quantisation (inconclusive)"])
rows.append(["episode duration (s)",
             "median %.2f; mean %.2f; range %.1f-%.1f (%d-%d frames, n=%d)" % (
                 ed["sim"], ed["sim_mean"], ed["sim_min"], ed["sim_max"], L["frames_min"], L["frames_max"], n_att),
             "Studio mean %.2f (16 episodes); joints.csv mean %.1f (15 files)" % (
                 ed["real_studio_mean"], ed["real_joints_csv_loop_mean"]),
             "different recordings; not directly comparable"])
rt = C["routing"]
rows.append(["brick routing vs OD-10", "%d of %d contradict (2x2 to green %d; 2x4 to blue %d)" % (
    rt["od10_contradictions_sim"], rt["episodes"], rt["pairs_sim"]["2x2->green"], rt["pairs_sim"]["2x4->blue"]),
    "not measured", "sim has no tape, so every 2x2 counts as good"])
rows.append(["gate decisions (a SmolVLA co-training; b ACT pretrain)",
             "a %s; b %s" % (gr["decisions"]["a_smolvla_cotraining"], gr["decisions"]["b_act_pretrain_augment"]),
             "n/a", "failed for a and b: " + ", ".join(G["decisions"]["a_smolvla_cotraining"]["failed"])])

C.pop("three_column", None)
C["comparison_table"] = {
    "columns": ["measure", "friend_sim", "real"],
    "rows": rows,
    "note": ("Friend sim against the real recordings only (twin column removed per D-102). Sample sizes are in "
             "each cell. Real image statistics use all four chunk-000 files per camera (10158 frames) after the "
             "L19 re-run. Close-view centroids depend on plate clipping at the frame edge and are approximate."),
}
C["patched_utc"] = now
C["patch_note"] = ("D-102 v2: twin not compared; three_column replaced by comparison_table; motion.sim_episode_frames "
                   "added (L17); sim episode duration corrected.")
M["patched_utc"] = now
M["patch_note"] = "D-102 v2: twin not compared; sim episode duration corrected (L17)."

dump(P["cmp"], C)
dump(P["met"], M)
after = {"compare.json": sha256_file(P["cmp"]), "metrics.json": sha256_file(P["met"])}
print(json.dumps({"patched_utc": now, "before": before, "after": after, "sim_duration": ed,
                  "table_rows": rows}, indent=1))
