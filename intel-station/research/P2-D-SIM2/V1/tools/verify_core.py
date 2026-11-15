import json, os, glob, math, subprocess
import numpy as np
import pyarrow.parquet as pq

R = "/home/intel/intel-junction-2026"
SIM = R + "/data/sim/2026-11-14/dataset-150-raw"
REAL = R + "/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af"
ROBOTS = R + "/app/motion/project_robots.json"
OUT = R + "/work/P2-D-SIM2/V1/out"
os.makedirs(OUT, exist_ok=True)
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TP = 2 * math.pi
PLATE = {"blue": (-0.105, -0.13), "green": (0.235, -0.16)}
res = {}

# ---- calibration (follower) and the two zero conventions
rob = json.load(open(ROBOTS))
fol = [r for r in rob if r.get("type") == "SO101_Follower"][0]
cal = fol["payload"]["calibration"]
rmin = np.array([cal[j]["range_min"] for j in J], float)
rmax = np.array([cal[j]["range_max"] for j in J], float)
span = rmax - rmin
K_mid = (200 * 4095 / (TP * span))[:5]          # zero = recorded range midpoint (W2)
K_2048 = (200 * 4096 / (TP * span))[:5]         # zero = raw tick 2048 (W1)
Z_2048 = ((2048 - rmin) / span * 200 - 100)[:5]
Z_2048 = Z_2048.astype(float)
res["calibration"] = {"range_min": rmin.tolist(), "range_max": rmax.tolist(),
                      "K_mid_per_rad": K_mid.round(3).tolist(), "K_2048_per_rad": K_2048.round(3).tolist(),
                      "z_2048_units": Z_2048.round(3).tolist()}
# where the calibrated range ends (normalized +-100) fall in rad and deg, under each convention
def ends_deg(K, z):
    lo = (-100 - z) / K; hi = (100 - z) / K
    return {"lo_deg": (lo * 180 / math.pi).round(2).tolist(), "hi_deg": (hi * 180 / math.pi).round(2).tolist()}
res["calibrated_range_ends_deg"] = {"mid_convention": ends_deg(K_mid, np.zeros(5)), "w1_2048_convention": ends_deg(K_2048, Z_2048)}
res["urdf_limit_deg"] = {"shoulder_lift": 100.0, "elbow_flex": round(1.69 * 180 / math.pi, 2)}

# ---- sim episodes
sims = []
for d in sorted(glob.glob(SIM + "/attempt-*")):
    md = json.load(open(d + "/metadata.json"))
    z = np.load(d + "/trajectory.npz")
    tr = json.load(open(d + "/trace.json"))
    st = z["state"].astype(float)
    obj = np.array([t["object_xyz"] for t in tr], float)
    n = min(len(st), len(obj))
    sims.append((os.path.basename(d), md, st[:n], obj[:n], d))
print("sim episodes", len(sims))
print("sim metadata keys", sorted(sims[0][1].keys()))
bt = [s[1].get("brick_type") for s in sims]
dest = [s[1]["destination"] for s in sims]
res["sim"] = {
    "episodes": len(sims),
    "accepted": sum(bool(s[1]["accepted"]) for s in sims),
    "success": sum(bool(s[1]["success"]) for s in sims),
    "grasp_verified": sum(bool(s[1]["physical_grasp_verified"]) for s in sims),
    "brick_type_counts": {k: bt.count(k) for k in sorted(set(map(str, bt)))},
    "destination_counts": {k: dest.count(k) for k in sorted(set(dest))},
    "task_text_matches": sum(s[1]["task"] == "move the %s brick to the %s plate" % (s[1]["target_color"], s[1]["destination"]) for s in sims),
}
od10 = [((b == "2x2" and d == "green") or (b != "2x2" and d == "blue")) for b, d in zip(bt, dest)]
res["sim"]["od10_contradictions"] = int(sum(od10))
res["sim"]["od10_2x2_on_green"] = int(sum(1 for b, d in zip(bt, dest) if b == "2x2" and d == "green"))
res["sim"]["od10_2x4_on_blue"] = int(sum(1 for b, d in zip(bt, dest) if b != "2x2" and d == "blue"))
errs = [math.hypot(s[1]["final_object_xyz"][0] - PLATE[s[1]["destination"]][0], s[1]["final_object_xyz"][1] - PLATE[s[1]["destination"]][1]) for s in sims]
res["sim"]["final_xy_err_m_max"] = round(max(errs), 4)
res["sim"]["final_xy_err_m_median"] = round(float(np.median(errs)), 4)

qs = np.concatenate([s[2][:, :5] for s in sims])            # all sim frames, rad
n_mid = qs * K_mid
n_2048 = qs * K_2048 + Z_2048
res["sim"]["frames_total"] = int(len(qs))
res["sim"]["wrist_flex_ge_1.6579rad_frac"] = round(float(np.mean(qs[:, 3] >= 1.6579)), 4)
res["sim"]["min_norm_mid"] = n_mid.min(axis=0).round(2).tolist()
res["sim"]["max_norm_mid"] = n_mid.max(axis=0).round(2).tolist()
res["sim"]["rest_like_frames_mid"] = int(np.sum((n_mid[:, 1] <= -90) & (n_mid[:, 2] >= 90)))
res["sim"]["rest_like_frames_w1"] = int(np.sum((n_2048[:, 1] <= -90) & (n_2048[:, 2] >= 90)))
res["sim"]["home_model_pose_norm_mid"] = (np.array([0, -0.5, 0.8, 0.6, 0, 0.5])[:5] * K_mid).round(2).tolist()

# sim drop pose: first frame the brick is within 1.5 cm of its final plate xy after moving >5 cm
drops = {"blue": [], "green": []}
grip_open, grip_closed = [], []
for name, md, st, obj, d in sims:
    init = np.array(md["initial_object_xyz"][:2])
    fx, fy = md["final_object_xyz"][0], md["final_object_xyz"][1]
    dfin = np.hypot(obj[:, 0] - fx, obj[:, 1] - fy)
    mov = np.hypot(obj[:, 0] - init[0], obj[:, 1] - init[1])
    idx = np.where((dfin < 0.015) & (mov > 0.05))[0]
    if len(idx):
        drops[md["destination"]].append(st[int(idx[0]), :5])
    grip_open.append(float(st[:, 5].max()))
    lift = np.where(obj[:, 2] > 0.03)[0]
    if len(lift):
        grip_closed.append(float(st[int(lift[0]), 5]))
sim_drop_med_mid, sim_drop_med_2048 = {}, {}
res["sim_drop"] = {}
for p in ["blue", "green"]:
    A = np.array(drops[p])
    med_rad = np.median(A, axis=0)
    sim_drop_med_mid[p] = med_rad * K_mid
    sim_drop_med_2048[p] = med_rad * K_2048 + Z_2048
    res["sim_drop"][p] = {"n": int(len(A)), "median_norm_mid": sim_drop_med_mid[p].round(2).tolist(),
                          "median_norm_w1": sim_drop_med_2048[p].round(2).tolist(),
                          "p25_norm_mid": (np.percentile(A, 25, axis=0) * K_mid).round(2).tolist(),
                          "p75_norm_mid": (np.percentile(A, 75, axis=0) * K_mid).round(2).tolist()}
res["sim_gripper"] = {"open_rad_median": round(float(np.median(grip_open)), 4),
                      "closed_on_brick_rad_median": round(float(np.median(grip_closed)), 4), "n_closed": len(grip_closed)}

# ---- real episodes (6d6bdce3)
files = sorted(glob.glob(REAL + "/data/chunk-*/file-*.parquet"))
S, E, F, TI = [], [], [], []
for f in files:
    tb = pq.read_table(f, columns=["observation.state", "episode_index", "frame_index", "task_index"]).to_pydict()
    S += tb["observation.state"]; E += tb["episode_index"]; F += tb["frame_index"]; TI += tb["task_index"]
S = np.array(S, float); E = np.array(E); F = np.array(F); TI = np.array(TI)
tasks = {}
for f in glob.glob(REAL + "/meta/tasks*"):
    if f.endswith(".parquet"):
        for r in pq.read_table(f).to_pylist():
            txt = [v for k, v in r.items() if isinstance(v, str)]
            tasks[r.get("task_index")] = txt[0] if txt else None
print("meta files", sorted(os.path.basename(x) for x in glob.glob(REAL + "/meta/*")))
print("tasks", tasks)
real_eps = []
for e in sorted(set(E.tolist())):
    m = np.where(E == e)[0]
    m = m[np.argsort(F[m])]
    st = S[m]
    g = st[:, 5]
    tmin = int(np.argmin(g))
    r = g[tmin + 5:] - g[tmin:-5]
    k = int(np.argmax(r))
    amp = float(r[k]); trel = tmin + k
    det = amp >= 5.0
    pose = st[trel, :5] if det else None
    plate = ("blue" if pose[0] > 0 else "green") if det else None
    real_eps.append({"ep": int(e), "task": tasks.get(int(TI[m][0]), str(int(TI[m][0]))), "frames": int(len(m)),
                     "tmin": tmin, "g_at_tmin": round(float(g[tmin]), 2), "g_max": round(float(g.max()), 2),
                     "release_amp_5f": round(amp, 2), "release_detected": bool(det), "t_release": int(trel),
                     "plate_by_pan_sign": plate, "release_norm": None if pose is None else pose.round(2).tolist()})
for x in real_eps:
    print("EP", x["ep"], x["task"], "frames", x["frames"], "amp", x["release_amp_5f"], "det", x["release_detected"], "plate", x["plate_by_pan_sign"], "rel", x["release_norm"], "gmin", x["g_at_tmin"], "gmax", x["g_max"])
res["real_episodes"] = real_eps
res["real"] = {"episodes": len(real_eps), "frames": int(len(S)),
               "duration_s_median": round(float(np.median([x["frames"] for x in real_eps]) / 30), 2),
               "rest_like_frames": int(np.sum((S[:, 1] <= -90) & (S[:, 2] >= 90))),
               "rest_like_frac": round(float(np.mean((S[:, 1] <= -90) & (S[:, 2] >= 90))), 4),
               "min_norm": S[:, :5].min(axis=0).round(2).tolist(), "max_norm": S[:, :5].max(axis=0).round(2).tolist(),
               "wrist_ge_98_frac": round(float(np.mean(S[:, 3] >= 98)), 4)}
rel = {"blue": [], "green": []}
for x in real_eps:
    if x["plate_by_pan_sign"]:
        rel[x["plate_by_pan_sign"]].append(x["release_norm"])
res["real_release"] = {}
for p in ["blue", "green"]:
    A = np.array(rel[p]) if rel[p] else np.zeros((0, 5))
    res["real_release"][p] = {"n": int(len(A)), "eps": [x["ep"] for x in real_eps if x["plate_by_pan_sign"] == p],
                              "median_norm": np.median(A, axis=0).round(2).tolist() if len(A) else None,
                              "sd_norm": (A.std(axis=0, ddof=1).round(2).tolist() if len(A) > 1 else None)}

# residuals real minus sim, under each convention
res["residual_real_minus_sim"] = {}
for p in ["blue", "green"]:
    if len(rel[p]):
        med = np.median(np.array(rel[p]), axis=0)
        res["residual_real_minus_sim"][p] = {"mid": (med - sim_drop_med_mid[p]).round(2).tolist(),
                                              "w1": (med - sim_drop_med_2048[p]).round(2).tolist()}
# sign test on body joints 0..3, sum of absolute residuals, both plates
def sign_sum(sgn):
    tot = 0.0
    for p in ["blue", "green"]:
        if len(rel[p]):
            med = np.median(np.array(rel[p]), axis=0)
            tot += float(np.sum(np.abs(med[:4] - sgn * sim_drop_med_mid[p][:4])))
    return round(tot, 2)
res["sign_test_abs_sum_joints0_3"] = {"plus1": sign_sum(1), "minus1": sign_sum(-1)}

# gripper scale: real open-minus-closed vs sim open-minus-closed
real_closed = float(np.median([x["g_at_tmin"] for x in real_eps]))
real_open = float(np.median([x["g_max"] for x in real_eps]))
sim_open = res["sim_gripper"]["open_rad_median"]; sim_closed = res["sim_gripper"]["closed_on_brick_rad_median"]
res["gripper_scale"] = {"real_open_units_median": round(real_open, 2), "real_closed_units_median": round(real_closed, 2),
                        "sim_open_rad_median": sim_open, "sim_closed_rad_median": sim_closed,
                        "K_units_per_rad_from_open_minus_closed": round((real_open - real_closed) / (sim_open - sim_closed), 2),
                        "encoder_linear_K_per_rad": round(100 * 4095 / (TP * float(span[5])) * 1.0, 2)}

# ---- brightness and sample frames (ffmpeg, gray, 640x480, 1 fps)
def frames(path, dur=None):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path]
    if dur:
        cmd += ["-t", str(dur)]
    cmd += ["-vf", "fps=1,scale=640:480", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0 or not p.stdout:
        return None, p.stderr.decode(errors="replace")[-300:]
    return np.frombuffer(p.stdout, dtype=np.uint8).reshape(-1, 480, 640), ""

def acc_init():
    return {"hist": np.zeros(256, np.int64), "sum": 0.0, "n": 0, "fm": []}
def acc_add(acc, a):
    acc["hist"] += np.bincount(a.ravel(), minlength=256)
    acc["sum"] += float(a.sum(dtype=np.float64)); acc["n"] += int(a.size)
    acc["fm"] += a.mean(axis=(1, 2)).tolist()
def acc_stats(acc):
    if acc["n"] == 0:
        return {"frames": 0}
    cdf = np.cumsum(acc["hist"]) / acc["hist"].sum()
    fm = np.array(acc["fm"])
    return {"frames": int(len(fm)), "mean_luma": round(acc["sum"] / acc["n"], 2),
            "p2": int(np.searchsorted(cdf, 0.02)), "p98": int(np.searchsorted(cdf, 0.98)),
            "frame_mean_min": round(float(fm.min()), 2), "frame_mean_max": round(float(fm.max()), 2),
            "frac_frames_mean_lt30": round(float((fm < 30).mean()), 3)}

bright = {}
errs_ff = {}
for label, pattern in [("sim_overhead", "overhead.mp4"), ("sim_close", "wrist.mp4")]:
    acc = acc_init()
    for d in sorted(glob.glob(SIM + "/attempt-00*"))[:10]:
        a, err = frames(d + "/" + pattern)
        if a is None:
            errs_ff[label] = err; continue
        acc_add(acc, a)
    bright[label] = acc_stats(acc)
for label, cam in [("real_top_243622061421", "243622061421"), ("real_close_243722064316", "243722064316")]:
    acc = acc_init()
    path = REAL + "/videos/observation.images.realsense " + cam + "/chunk-000/file-000.mp4"
    a, err = frames(path, dur=60)
    if a is None:
        errs_ff[label] = err
    else:
        acc_add(acc, a)
    bright[label] = acc_stats(acc)
res["brightness"] = bright
res["ffmpeg_errors"] = errs_ff
# keep W1 and W2 comparison stats for context
res["worker_claims"] = {"W1_mean_luma_sim_overhead_vs_real_top": "53.0 vs 91.8 (first 5 s, 1 fps)",
                        "W2_mean_luma_sim_overhead_vs_real_top": "49.0 vs 86.2 (release frame)"}

with open(OUT + "/verify_core.json", "w") as fh:
    json.dump(res, fh, indent=1, default=str)
print("== SUMMARY ==")
for k in ["sim", "sim_drop", "sim_gripper", "real", "real_release", "residual_real_minus_sim", "sign_test_abs_sum_joints0_3", "gripper_scale", "calibrated_range_ends_deg", "brightness", "ffmpeg_errors"]:
    print(k, json.dumps(res[k], default=str)[:1800])
