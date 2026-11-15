#!/usr/bin/env python3
"""P3-FRIEND-SIM sim-vs-real comparison (data only). Sim and real media are decoded with ffmpeg;
sim source is parsed as text (never executed). Usage: python3 -I sim_real_compare.py OUT_DIR"""
import ast, glob, json, math, os, re, subprocess, sys, time
import xml.etree.ElementTree as ET
import numpy as np
import cv2

R = "/home/intel/intel-junction-2026"
OUT = sys.argv[1].rstrip("/")
SIM = R + "/data/sim/live/2026-11-14/dataset-150-raw"
REAL = R + "/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af"
REAL_TOP_FILES = [REAL + "/videos/observation.images.realsense 243622061421/chunk-000/file-00%d.mp4" % k for k in range(4)]
REAL_TOP = REAL_TOP_FILES[0]
REAL_CLOSE_FILES = [REAL + "/videos/observation.images.realsense 243722064316/chunk-000/file-00%d.mp4" % k for k in range(4)]
REAL_CLOSE = REAL_CLOSE_FILES[0]
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
W0, H0 = 640, 480
T0 = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
os.makedirs(OUT + "/side_by_side", exist_ok=True)

# ---------- sim scene constants, parsed from station_scene.py (text only)
txt = open(SIM + "/source/station_scene.py", encoding="utf-8").read()
BOWL = ast.literal_eval(re.search(r"BOWL_CENTERS\s*=\s*(\{.*?\})", txt, re.S).group(1))
FLOOR = float(re.search(r"BOWL_FLOOR\s*=\s*([0-9.]+)", txt).group(1))


def cam_spec(name):
    m = re.search(r"camera_look_at\(%s,\s*\[([^\]]+)\],\s*\[([^\]]+)\]\)" % name, txt)
    fv = re.search(r'%s\.set\("fovy",\s*"([0-9.]+)"\)' % name, txt)
    return ([float(v) for v in m.group(1).split(",")], [float(v) for v in m.group(2).split(",")], float(fv.group(1)))


def make_cam(pos, tgt, fovy):
    p = np.array(pos, float)
    z = p - np.array(tgt, float)
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return {"p": p, "x": x, "y": y, "z": z, "f": (H0 / 2) / math.tan(math.radians(fovy) / 2), "fovy": fovy, "pos": pos, "tgt": tgt}


def project(cam, X):
    d = np.array(X, float) - cam["p"]
    depth = -float(d @ cam["z"])
    return (W0 / 2 + cam["f"] * float(d @ cam["x"]) / depth, H0 / 2 - cam["f"] * float(d @ cam["y"]) / depth)


def back(cam, u, v, zp):
    dc = np.array([(u - W0 / 2) / cam["f"], -(v - H0 / 2) / cam["f"], -1.0])
    dw = dc[0] * cam["x"] + dc[1] * cam["y"] + dc[2] * cam["z"]
    t = (zp - cam["p"][2]) / dw[2]
    return cam["p"] + t * dw


CAM_T = make_cam(*cam_spec("overhead"))
CAM_C = make_cam(*cam_spec("wrist"))


def dims(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height", "-of", "csv=p=0", path], capture_output=True, text=True)
    return r.stdout.strip()


def frames(path):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-vf", "fps=1",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = W0 * H0 * 3
    try:
        while True:
            b = p.stdout.read(n)
            if len(b) < n:
                break
            yield np.frombuffer(b, np.uint8).reshape(H0, W0, 3).copy()
    finally:
        p.stdout.close()
        p.wait()


HSV = {"blue": ((80, 105), 90, 100), "green": ((50, 90), 60, 60)}


def plate(bgr, color):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    (h0, h1), smin, vmin = HSV[color]
    m = cv2.inRange(hsv, (h0, smin, vmin), (h1, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    area = int(st[k, cv2.CC_STAT_AREA])
    if area < 4000:
        return None
    ys, xs = np.nonzero(lab == k)
    return {"cx": float(xs.mean()), "cy": float(ys.mean()), "area": area,
            "w": int(st[k, cv2.CC_STAT_WIDTH]), "h": int(st[k, cv2.CC_STAT_HEIGHT])}


def frame_stats(bgr):
    b, g, r = [bgr[..., i].astype(np.float32) for i in range(3)]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return {"R": float(r.mean()), "G": float(g.mean()), "B": float(b.mean()), "luma": float(y.mean()),
            "luma_p50": float(np.median(y)), "luma_p98": float(np.percentile(y, 98)),
            "dark_frac": float((y < 40).mean()), "S": float(hsv[..., 1].mean()), "V": float(hsv[..., 2].mean())}


# ---------- media pass: 1 fps frames, stats and plate detections
agg = {k: {"stats": [], "blue": [], "green": []} for k in ["sim_top", "sim_close", "real_top", "real_close"]}
keep = {}


def run(key, path, keep_first):
    first = True
    for fr in frames(path):
        agg[key]["stats"].append(frame_stats(fr))
        for c in ("blue", "green"):
            d = plate(fr, c)
            if d:
                agg[key][c].append(d)
        if first and keep_first:
            keep[key] = {"img": fr, "blue": plate(fr, "blue"), "green": plate(fr, "green")}
        first = False


attempts = sorted(glob.glob(SIM + "/attempt-*"))
for i, a in enumerate(attempts):
    run("sim_top", a + "/overhead.mp4", i == 0)
    run("sim_close", a + "/wrist.mp4", i == 0)
PERFILE = []
for key, paths in [("real_top", REAL_TOP_FILES), ("real_close", REAL_CLOSE_FILES)]:
    for i, p in enumerate(paths):
        n0 = len(agg[key]["stats"])
        run(key, p, i == 0)
        seg = agg[key]["stats"][n0:]
        PERFILE.append({"view": key, "file": p.rsplit("/", 1)[-1], "frames": len(seg),
                        "luma_mean": round(float(np.mean([x["luma"] for x in seg])), 2)})


def summ(key):
    s = agg[key]["stats"]
    o = {"frames": len(s)}
    for k in ["R", "G", "B", "luma", "luma_p50", "luma_p98", "dark_frac", "S", "V"]:
        o[k] = round(float(np.mean([x[k] for x in s])), 4)
    o["luma_frame_sd"] = round(float(np.std([x["luma"] for x in s])), 3)
    return o


def psum(dets):
    if not dets:
        return None
    a = {k: np.array([d[k] for d in dets], float) for k in ["cx", "cy", "area", "w", "h"]}
    return {"n": len(dets), "cx": round(float(np.median(a["cx"])), 2), "cy": round(float(np.median(a["cy"])), 2),
            "cx_sd": round(float(np.std(a["cx"])), 2), "cy_sd": round(float(np.std(a["cy"])), 2),
            "area_med": float(np.median(a["area"])), "w_med": float(np.median(a["w"])), "h_med": float(np.median(a["h"]))}


# ---------- plate geometry: px, then mm through the sim camera model on the plate plane
plates = {}
marks = {}
for view, CAM, sk, rk in [("top", CAM_T, "sim_top", "real_top"), ("close", CAM_C, "sim_close", "real_close")]:
    S = {c: psum(agg[sk][c]) for c in BOWL}
    Rr = {c: psum(agg[rk][c]) for c in BOWL}
    pred = {c: project(CAM, (BOWL[c][0], BOWL[c][1], FLOOR)) for c in BOWL}
    entry = {"sim_detected_median": S, "real_detected_median": Rr,
             "sim_model_predicted_px": {c: [round(pred[c][0], 2), round(pred[c][1], 2)] for c in BOWL},
             "sim_detected_minus_predicted_px": {}, "real_minus_sim_px": {}, "real_minus_sim_mm_via_sim_model": {},
             "sim_detected_minus_bowl_centre_mm_via_sim_model": {}}
    if all(S[c] and Rr[c] for c in BOWL):
        for c in BOWL:
            entry["sim_detected_minus_predicted_px"][c] = [round(S[c]["cx"] - pred[c][0], 2), round(S[c]["cy"] - pred[c][1], 2)]
            entry["real_minus_sim_px"][c] = [round(Rr[c]["cx"] - S[c]["cx"], 2), round(Rr[c]["cy"] - S[c]["cy"], 2)]
            br = back(CAM, Rr[c]["cx"], Rr[c]["cy"], FLOOR)
            bs = back(CAM, S[c]["cx"], S[c]["cy"], FLOOR)
            entry["real_minus_sim_mm_via_sim_model"][c] = [round(1000 * (br[0] - bs[0]), 1), round(1000 * (br[1] - bs[1]), 1)]
            entry["sim_detected_minus_bowl_centre_mm_via_sim_model"][c] = [round(1000 * (bs[0] - BOWL[c][0]), 1), round(1000 * (bs[1] - BOWL[c][1]), 1)]
        ds = math.hypot(S["blue"]["cx"] - S["green"]["cx"], S["blue"]["cy"] - S["green"]["cy"])
        dr = math.hypot(Rr["blue"]["cx"] - Rr["green"]["cx"], Rr["blue"]["cy"] - Rr["green"]["cy"])
        entry["plate_spacing_px"] = {"sim": round(ds, 1), "real": round(dr, 1), "real_over_sim": round(dr / ds, 4)}
        entry["area_ratio_real_over_sim"] = {c: round(Rr[c]["area_med"] / S[c]["area_med"], 3) for c in BOWL}
    plates[view] = entry
    marks[view] = pred

# side-by-side images at matching first frames (attempt 0 sim vs first real frame)
def composite(left, right, lbl_l, lbl_r, ml, mr, out):
    canvas = np.full((H0 + 36, 2 * W0 + 8, 3), 255, np.uint8)
    canvas[36:, :W0] = left
    canvas[36:, W0 + 8:] = right
    for x0, mk in ((0, ml), (W0 + 8, mr)):
        for u, v, col, text in mk:
            X, Y = int(round(x0 + u)), int(round(36 + v))
            cv2.circle(canvas, (X, Y), 11, (255, 255, 255), 2)
            cv2.drawMarker(canvas, (X, Y), col, cv2.MARKER_CROSS, 22, 2)
            cv2.putText(canvas, text, (X + 12, Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    cv2.putText(canvas, lbl_l, (6, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, lbl_r, (W0 + 14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(out, canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])


COL = {"blue": (255, 0, 255), "green": (0, 255, 255)}
for view, sk, rk, lbl in [("top", "sim_top", "real_top", "top view"), ("close", "sim_close", "real_close", "close view")]:
    CAM = CAM_T if view == "top" else CAM_C
    ks, kr = keep[sk], keep[rk]
    ml = [(ks[c]["cx"], ks[c]["cy"], COL[c], "sim " + c, ) for c in BOWL if ks[c]] + [(marks[view][c][0], marks[view][c][1], (255, 255, 255), "model " + c) for c in BOWL]
    mr = [(kr[c]["cx"], kr[c]["cy"], COL[c], "real " + c) for c in BOWL if kr[c]] + [(marks[view][c][0], marks[view][c][1], (255, 255, 255), "sim-model " + c) for c in BOWL]
    ml = [(u, v, col, t) for (u, v, col, t) in ml]
    ml_fix = [m if len(m) == 4 else m for m in ml]
    composite(ks["img"], kr["img"], "SIM " + lbl + " (sim attempt-0000 frame 0)", "REAL " + lbl + " (real top/close frame 0)",
              [m for m in ml_fix], [m for m in mr], OUT + "/side_by_side/%s_sim_a0000_f0_vs_real_t0.jpg" % view)

# ---------- brightness and colour
bright = {}
for view, sk, rk in [("top", "sim_top", "real_top"), ("close", "sim_close", "real_close")]:
    s, r_ = summ(sk), summ(rk)
    bright[view] = {"sim": s, "real": r_, "luma_ratio_sim_over_real": round(s["luma"] / r_["luma"], 4),
                    "sim_darker_pct_vs_real": round(100 * (1 - s["luma"] / r_["luma"]), 2),
                    "dark_frac_sim_minus_real": round(s["dark_frac"] - r_["dark_frac"], 4)}

# ---------- joints: units, zero convention, ranges, pinned, gripper, motion
rob = json.load(open(R + "/app/motion/project_robots.json"))
fol = [r for r in rob if r.get("type") == "SO101_Follower"][0]
cal = fol["payload"]["calibration"]
rmin = np.array([cal[j]["range_min"] for j in JOINTS], float)
rmax = np.array([cal[j]["range_max"] for j in JOINTS], float)
span = rmax - rmin
K = np.empty(6)
K[:5] = 200 * 4095 / (2 * math.pi * span[:5])
K[5] = 100 * 4095 / (2 * math.pi * span[5])
mj = ET.parse(R + "/app/simgate/ref/so101.xml").getroot()
rng = {j.get("name"): [float(v) for v in j.get("range").split()] for j in mj.iter("joint") if j.get("range") and j.get("name")}
wrist_lim_norm = rng["wrist_flex"][1] * K[3]

sim, sim_first, meta_pairs, od10 = [], [], {}, 0
for a in attempts:
    st = np.load(a + "/trajectory.npz")["state"].astype(float)
    sim.append(st * K)
    sim_first.append(st[0] * K)
    md = json.load(open(a + "/metadata.json"))
    key = "%s->%s" % (md["brick_type"], md["destination"])
    meta_pairs[key] = meta_pairs.get(key, 0) + 1
    if (md["brick_type"] == "2x2" and md["destination"] == "green") or (md["brick_type"] != "2x2" and md["destination"] == "blue"):
        od10 += 1

real, real_first, real_meta = [], [], []
for f in sorted(glob.glob(R + "/logs/sort/any_*/ep_*/joints.csv")):
    rows = np.genfromtxt(f, delimiter=",", names=True)
    t = rows["t"]
    pos = np.column_stack([rows["pos%d" % i] for i in range(6)])
    if len(t) < 60 or np.diff(t).max() > 0.5:
        real_meta.append({"file": f.replace(R, ""), "used": False, "n": int(len(t))})
        continue
    tg = np.arange(t[0], t[-1], 1 / 30)
    pg = np.column_stack([np.interp(tg, t, pos[:, i]) for i in range(6)])
    real.append(pg)
    real_first.append(pos[0])
    real_meta.append({"file": f.replace(R, ""), "used": True, "n": int(len(t)), "dt_med_s": round(float(np.median(np.diff(t))), 4),
                      "dt_max_s": round(float(np.diff(t).max()), 4)})


def rng_of(arrs):
    A = np.concatenate(arrs)
    return {"min": A.min(0).round(2).tolist(), "max": A.max(0).round(2).tolist(), "median": np.median(A, 0).round(2).tolist()}


def motion_stats(arrs):
    V = np.concatenate([np.diff(X, axis=0) * 30.0 for X in arrs])
    Acc = np.concatenate([np.diff(X, n=2, axis=0) * 900.0 for X in arrs])
    Jk = np.concatenate([np.diff(X, n=3, axis=0) * 27000.0 for X in arrs])
    sp = np.abs(V)
    ep_max = np.array([np.abs(np.diff(X, axis=0) * 30.0).max(0) for X in arrs])
    return {"speed_p50": np.percentile(sp, 50, axis=0).round(2).tolist(), "speed_p95": np.percentile(sp, 95, axis=0).round(2).tolist(),
            "speed_max": sp.max(0).round(2).tolist(), "episode_max_speed_median": np.median(ep_max, 0).round(2).tolist(),
            "accel_rms": np.sqrt((Acc ** 2).mean(0)).round(1).tolist(), "jerk_rms": np.sqrt((Jk ** 2).mean(0)).round(0).tolist(),
            "still_frac_below_0p5_units_per_s": (sp < 0.5).mean(0).round(4).tolist(), "frames": int(sum(len(X) for X in arrs))}


SA = np.concatenate(sim)
RA = np.concatenate(real)
joints = {
    "K_units_per_rad": K.round(3).tolist(),
    "K_note": "normalized = rad x K, zero at range midpoint, sign +1 (same as sim_gate.py); gripper encoder-linear (assumption)",
    "sim_range_norm": rng_of(sim), "real_joints_csv_range_norm": rng_of(real),
    "sim_first_frame_norm_median": np.median(np.array(sim_first), 0).round(2).tolist(),
    "real_first_row_norm_median": np.median(np.array(real_first), 0).round(2).tolist(),
    "wrist_flex_mjcf_limit_norm": round(float(wrist_lim_norm), 2),
    "wrist_flex_sim_frac_within_0p5_of_limit": round(float(np.mean(SA[:, 3] >= wrist_lim_norm - 0.5)), 4),
    "wrist_flex_real_frac_ge_98": round(float(np.mean(RA[:, 3] >= 98.0)), 4),
    "gripper_sim_open_max": round(float(SA[:, 5].max()), 2), "gripper_real_open_max": round(float(RA[:, 5].max()), 2),
    "gripper_sim_min": round(float(SA[:, 5].min()), 2), "gripper_real_min": round(float(RA[:, 5].min()), 2),
}
motion = {"sim": motion_stats(sim), "real_joints_csv": motion_stats(real), "real_episode_meta": real_meta,
          "n_sim_episodes": len(sim), "n_real_episodes_used": len(real)}

# ---------- gate re-run check and version identity
gate_new = json.load(open(OUT + "/gate/2026-11-14-95fa40dbf3.json"))
gate_old = json.load(open(R + "/data/sim/gate/2026-11-14-95fa40dbf3.json"))
inv = json.load(open(OUT + "/inventory_compare.json"))
gate_block = {"version": gate_new["version"], "results": gate_new["results"], "decisions": {k: v["decision"] for k, v in gate_new["decisions"].items()},
              "identical_to_existing_gate_json": gate_new["results"] == gate_old["results"] and gate_new["decisions"] == gate_old["decisions"] and gate_new["checks"] == gate_old["checks"],
              "geometry": gate_new["checks"]["geometry"], "release_pan_residual_deg": gate_new["checks"]["release"]["pan_residual_deg"],
              "gripper": gate_new["checks"]["gripper"], "units": gate_new["checks"]["units"]}
identity = {"live_equals_2026_11_14": bool(inv["tar_sha_equal"] and all(v["changed"] == 0 for v in inv["file_hash"].values()) and inv["attempts"]["common"] == 150 and not inv["metadata_fields_changed"] and not inv["recording_context_changed_keys"] and not inv["source_files_changed"]),
            "attempts": inv["attempts"], "file_hash": inv["file_hash"], "tar_sha_equal": inv["tar_sha_equal"],
            "outside_raw_only_in_2026_11_14": inv["outside_raw_only_old_count"], "metadata_fields_changed": inv["metadata_fields_changed"]}
routing = {"pairs_sim": meta_pairs, "od10_contradictions_sim": od10, "episodes": len(sim), "tape_word_in_sim_source": bool(re.search(r"tape", txt + open(SIM + "/source/sorting_env.py", encoding="utf-8").read(), re.I))}

res = {"stage": "friend-sim-v2", "utc": T0, "scope": "R/data/sim/live (= 2026-11-14 copy, verified) vs real cell (studio dataset 6d6bdce3 + R/logs/sort joints.csv)",
       "dims": {"real_top": dims(REAL_TOP), "real_close": dims(REAL_CLOSE), "sim_overhead": dims(SIM + "/attempt-0000/overhead.mp4"), "sim_wrist": dims(SIM + "/attempt-0000/wrist.mp4")},
       "version_identity": identity, "gate_rerun": gate_block, "plates": plates, "brightness": bright, "real_per_file": PERFILE, "joints": joints,
       "motion": motion, "routing": routing,
       "camera_models": {"sim_overhead": {"pos": CAM_T["pos"], "target": CAM_T["tgt"], "fovy": CAM_T["fovy"]}, "sim_close_fixed_wrist_cam": {"pos": CAM_C["pos"], "target": CAM_C["tgt"], "fovy": CAM_C["fovy"]}},
       "bowl_centres_world_m": BOWL, "bowl_floor_m": FLOOR}
with open(OUT + "/compare.json", "w") as fh:
    json.dump(res, fh, indent=1, default=str)
print(json.dumps({"dims": res["dims"], "identity": identity["live_equals_2026_11_14"], "gate_same": gate_block["identical_to_existing_gate_json"],
                  "plates_top": plates["top"], "plates_close": plates["close"], "bright": {k: {kk: v[kk] for kk in ["luma_ratio_sim_over_real", "sim_darker_pct_vs_real"]} for k, v in bright.items()},
                  "bright_luma": {k: (v["sim"]["luma"], v["real"]["luma"]) for k, v in bright.items()},
                  "joints_first": [joints["sim_first_frame_norm_median"], joints["real_first_row_norm_median"]],
                  "wrist": [joints["wrist_flex_sim_frac_within_0p5_of_limit"], joints["wrist_flex_real_frac_ge_98"]], "pairs": meta_pairs, "od10": od10,
                  "n_real": len(real), "n_sim": len(sim)}, indent=1, default=str)[:9000])
