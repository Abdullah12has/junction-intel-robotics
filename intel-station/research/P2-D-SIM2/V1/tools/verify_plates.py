import json, os, glob, math
import numpy as np
import pyarrow.parquet as pq

R = "/home/intel/intel-junction-2026"
SIM = R + "/data/sim/2026-11-14/dataset-150-raw"
REAL = R + "/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af"
ROBOTS = R + "/app/motion/project_robots.json"
OUT = R + "/work/P2-D-SIM2/V1/out"
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TP = 2 * math.pi
PLATE = {"blue": (-0.105, -0.13), "green": (0.235, -0.16)}
rob = json.load(open(ROBOTS))
fol = [r for r in rob if r.get("type") == "SO101_Follower"][0]
cal = fol["payload"]["calibration"]
rmin = np.array([cal[j]["range_min"] for j in J], float)
rmax = np.array([cal[j]["range_max"] for j in J], float)
span = rmax - rmin
K_mid = (200 * 4095 / (TP * span))[:5]
K_2048 = (200 * 4096 / (TP * span))[:5]
Z_2048 = ((2048 - rmin) / span * 200 - 100)[:5]
res = {"K_mid": K_mid.round(3).tolist(), "K_2048": K_2048.round(3).tolist(), "Z_2048": Z_2048.round(3).tolist()}

# ---- sim: release = largest 5-frame gripper rise after the brick first reaches its final plate xy
rows = []
for d in sorted(glob.glob(SIM + "/attempt-*")):
    md = json.load(open(d + "/metadata.json"))
    z = np.load(d + "/trajectory.npz")
    tr = json.load(open(d + "/trace.json"))
    st = z["state"].astype(float)
    obj = np.array([t["object_xyz"] for t in tr], float)
    n = min(len(st), len(obj)); st = st[:n]; obj = obj[:n]
    init = np.array(md["initial_object_xyz"][:2])
    fx, fy = md["final_object_xyz"][0], md["final_object_xyz"][1]
    dfin = np.hypot(obj[:, 0] - fx, obj[:, 1] - fy)
    mov = np.hypot(obj[:, 0] - init[0], obj[:, 1] - init[1])
    idx = np.where((dfin < 0.015) & (mov > 0.05))[0]
    if not len(idx):
        continue
    ta = int(idx[0]); g = st[:, 5]
    r = g[ta + 5:] - g[ta:-5]
    if len(r) == 0:
        continue
    k = int(np.argmax(r)); trel = ta + k
    rows.append({"attempt": os.path.basename(d), "plate": md["destination"], "ta": ta, "trel": trel,
                 "rise_rad": float(r[k]), "pose_rel": st[trel, :5], "pose_arr": st[ta, :5],
                 "wrist_at_rel": float(st[trel, 3]), "brick": md.get("brick_type")})
res["sim_release_count"] = len(rows)
res["sim_release_min_rise_rad"] = round(min(r["rise_rad"] for r in rows), 4)
res["sim_release_frames_after_arrival_median"] = float(np.median([r["trel"] - r["ta"] for r in rows]))
res["sim_plate_count"] = {p: sum(1 for r in rows if r["plate"] == p) for p in ["blue", "green"]}
sim_med = {}
for p in ["blue", "green"]:
    A = np.array([r["pose_rel"] for r in rows if r["plate"] == p])
    B = np.array([r["pose_arr"] for r in rows if r["plate"] == p])
    W = np.array([r["wrist_at_rel"] for r in rows if r["plate"] == p])
    sim_med[p] = np.median(A, axis=0) * K_mid
    res["sim_release_" + p] = {"n": int(len(A)), "norm_mid_median": (np.median(A, axis=0) * K_mid).round(2).tolist(),
                               "norm_mid_p25": (np.percentile(A, 25, axis=0) * K_mid).round(2).tolist(),
                               "norm_mid_p75": (np.percentile(A, 75, axis=0) * K_mid).round(2).tolist(),
                               "norm_w1_median": (np.median(A, axis=0) * K_2048 + Z_2048).round(2).tolist(),
                               "arrival_norm_mid_median": (np.median(B, axis=0) * K_mid).round(2).tolist(),
                               "wrist_at_release_ge_1.6579_frac": round(float(np.mean(W >= 1.6579)), 3),
                               "brick_2x2": int(sum(1 for r in rows if r["plate"] == p and r["brick"] == "2x2")),
                               "brick_2x4": int(sum(1 for r in rows if r["plate"] == p and r["brick"] == "2x4"))}
sim_sim_pan_deg = {p: round(float(sim_med[p][0] / K_mid[0] * 180 / math.pi), 2) for p in sim_med}
res["sim_pan_deg_median"] = sim_sim_pan_deg

# ---- real: release episodes (explicit sets from Opus 13:45Z/13:52Z and W1/W2 review)
files = sorted(glob.glob(REAL + "/data/chunk-*/file-*.parquet"))
S, E, F = [], [], []
for f in files:
    tb = pq.read_table(f, columns=["observation.state", "episode_index", "frame_index"]).to_pydict()
    S += tb["observation.state"]; E += tb["episode_index"]; F += tb["frame_index"]
S = np.array(S, float); E = np.array(E); F = np.array(F)

def real_release(e):
    m = np.where(E == e)[0]
    m = m[np.argsort(F[m])]
    st = S[m]; g = st[:, 5]
    tmin = int(np.argmin(g))
    r = g[tmin + 5:] - g[tmin:-5]
    k = int(np.argmax(r))
    return st[tmin + k, :5], float(r[k])

sets = {"blue": [7, 11, 12], "green": [8, 10, 14, 15], "blue_incl13_sensitivity": [7, 11, 12, 13]}
real_rel = {}
res["real_release_detail"] = {}
for p, eps in sets.items():
    P = []
    for e in eps:
        pose, amp = real_release(e)
        P.append(pose)
        res["real_release_detail"].setdefault(p, []).append({"ep": e, "norm": pose.round(2).tolist(), "amp": round(amp, 2)})
    real_rel[p] = np.array(P)
    res["real_release_detail"][p + "_median"] = np.median(np.array(P), axis=0).round(2).tolist()
    res["real_release_detail"][p + "_sd"] = (np.std(np.array(P), axis=0, ddof=1).round(2).tolist() if len(P) > 1 else None)

res["residual_real_minus_sim_mid"] = {}
res["residual_real_minus_sim_w1"] = {}
for p in ["blue", "green"]:
    med = np.median(real_rel[p], axis=0)
    res["residual_real_minus_sim_mid"][p] = (med - sim_med[p]).round(2).tolist()
    res["residual_real_minus_sim_w1"][p] = (med - (np.median(np.array([r["pose_rel"] for r in rows if r["plate"] == p]), axis=0) * K_2048 + Z_2048)).round(2).tolist()
res["residual_pan_deg"] = {p: round(float((np.median(real_rel[p], axis=0)[0] - sim_med[p][0]) / K_mid[0] * 180 / math.pi), 2) for p in ["blue", "green"]}
res["residual_real_minus_sim_incl13_blue"] = (np.median(real_rel["blue_incl13_sensitivity"], axis=0) - sim_med["blue"]).round(2).tolist()

def sign_sum(s):
    tot = 0.0
    for p in ["blue", "green"]:
        med = np.median(real_rel[p], axis=0)
        tot += float(np.sum(np.abs(med[:4] - s * sim_med[p][:4])))
    return round(tot, 2)
res["sign_test_joints0_3"] = {"plus1": sign_sum(1), "minus1": sign_sum(-1)}
pan_sign = {"plus1": round(float(sum(abs(np.median(real_rel[p], axis=0)[0] - sim_med[p][0]) for p in ["blue", "green"])), 2),
            "minus1": round(float(sum(abs(np.median(real_rel[p], axis=0)[0] + sim_med[p][0]) for p in ["blue", "green"])), 2)}
res["sign_test_pan_only"] = pan_sign

with open(OUT + "/verify_plates.json", "w") as fh:
    json.dump(res, fh, indent=1)
for k, v in res.items():
    print(k, json.dumps(v)[:900])
