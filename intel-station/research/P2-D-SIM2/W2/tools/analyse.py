"""W2 analysis: sim (MuJoCo, absolute radians) vs real (Studio normalized units).

Parses data files only (json, npz, parquet). Never imports team code.
Mapping: physicalai_so101_plugin/_conversion.py (read-only repo file):
  body normalized n = (tick - rmin)/R*200 - 100 ; degrees = (tick - mid)*360/4095 ; mid=(rmin+rmax)/2
  => radians q = (tick - mid) * 2*pi/4095 ; n = K*q with K = 200*4095/(2*pi*R), zero offset 0.
"""
import glob
import json
import math
import sys

import numpy as np
import pyarrow.parquet as pq

OUT = "/home/intel/intel-junction-2026/work/P2-D-SIM2/W2/out"
SIM = "/home/intel/intel-junction-2026/data/sim/2026-11-14/dataset-150-raw"
REAL = "/home/intel/intel-junction-2026/data/studio-datasets"
DS_RAW = REAL + "/6d6bdce3-b382-478d-9697-fb9e573b97af"
DS_CUR = REAL + "/a4ba6ede-092c-450f-8c76-559f66991dbd"
FPS = 30.0
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# SO101_Follower calibration, app/motion/project_robots.json (raw ticks)
CAL = {"shoulder_pan": (891, 3212), "shoulder_lift": (822, 3090), "elbow_flex": (898, 3102),
       "wrist_flex": (812, 3171), "wrist_roll": (74, 3899), "gripper": (2046, 3577)}
RAD_PER_TICK = 2 * math.pi / 4095.0
K = {}
for j in JOINTS[:5]:
    K[j] = 200.0 / ((CAL[j][1] - CAL[j][0]) * RAD_PER_TICK)
K["gripper"] = 100.0 / ((CAL["gripper"][1] - CAL["gripper"][0]) * RAD_PER_TICK)
PLATE_XY = {"blue": (-0.105, -0.13), "green": (0.235, -0.16)}  # sim station_scene.BOWL_CENTERS
SIM_HOME = [0, -0.5, 0.8, 0.6, 0, 0.5]  # sim so101_env/station_scene HOME (model-only)
REAL_BLUE_EPS = {7, 11, 12}   # FACTS 13:52Z, 14:26Z (Opus-verified, to be cross-checked)
REAL_GREEN_EPS = {8, 9, 10, 14, 15}


def sim_rows():
    rows = []
    for d in sorted(glob.glob(SIM + "/attempt-*")):
        meta = json.load(open(d + "/metadata.json"))
        trace = json.load(open(d + "/trace.json"))
        z = np.load(d + "/trajectory.npz")
        S = z["state"].astype(float)
        A = z["action"].astype(float)
        ph = [t["phase"] for t in trace]

        def span(p):
            idx = [i for i, x in enumerate(ph) if x == p]
            return (idx[0], idx[-1]) if idx else (None, None)

        rel0, _ = span("release")
        _, low1 = span("lower")
        _, ali1 = span("align")
        _, gr1 = span("grasp")
        V = np.diff(S, axis=0) * FPS
        Jk = np.diff(S, n=3, axis=0) * FPS ** 3
        fin = np.array(meta["final_object_xyz"][:2])
        dest = meta["destination"]
        exp_dest = "blue" if meta["brick_type"] == "2x2" else "green"
        rows.append({
            "id": d.split("/")[-1], "accepted": meta["accepted"], "success": meta["success"],
            "grasp_verified": meta["physical_grasp_verified"], "frames": meta["frames"],
            "brick": meta["brick_type"], "color": meta["target_color"], "dest": dest,
            "seed": meta["seed"], "task": meta["task"], "od10_expected": exp_dest,
            "od10_contradiction": dest != exp_dest,
            "final_xy_err_m": float(np.linalg.norm(fin - np.array(PLATE_XY[dest]))),
            "final_z_m": float(meta["final_object_xyz"][2]),
            "place_q": S[rel0].tolist(), "grasp_q": S[ali1].tolist(),
            "grip_closed_on_brick_rad": float(S[gr1][5]) if gr1 is not None else None,
            "vmax_radps": np.abs(V).max(axis=0).tolist(),
            "vp99_radps": np.percentile(np.abs(V), 99, axis=0).tolist(),
            "jerk_rms": np.sqrt((Jk ** 2).mean(axis=0)).tolist(),
            "dur_s": len(ph) / FPS,
            "phase_frames": {p: ph.count(p) for p in sorted(set(ph))},
        })
    return rows


def load_parquet_dir(root, pattern):
    tabs = [pq.read_table(f) for f in sorted(glob.glob(root + pattern))]
    return tabs


def real_rows(root, label):
    import pandas as pd
    data = pd.concat([t.to_pandas() for t in load_parquet_dir(root, "/data/chunk-000/file-*.parquet")],
                     ignore_index=True)
    tasks = pq.read_table(root + "/meta/tasks.parquet").to_pandas()
    if "task" not in tasks.columns:
        tasks = tasks.reset_index().rename(columns={"index": "task"})
    tmap = dict(zip(tasks["task_index"], tasks["task"]))
    rows = []
    for ep, g in data.groupby("episode_index"):
        g = g.sort_values("frame_index")
        S = np.stack(g["observation.state"].values).astype(float)
        A = np.stack(g["action"].values).astype(float)
        ts = g["timestamp"].values.astype(float)
        fps_est = 1.0 / np.median(np.diff(ts)) if len(ts) > 2 else FPS
        grip = S[:, 5]
        d5 = grip[5:] - grip[:-5]
        rel_idx = np.flatnonzero(d5 > 6)
        grasp_idx = np.flatnonzero(d5 < -6)
        rel = int(rel_idx[-1]) if len(rel_idx) else None
        grasp = None
        if rel is not None:
            prior = grasp_idx[grasp_idx < rel]
            grasp = int(prior[-1]) if len(prior) else None
        V = np.diff(S, axis=0) * fps_est
        Jk = np.diff(S, n=3, axis=0) * fps_est ** 3
        task = tmap.get(int(g["task_index"].iloc[0]), "?")
        rows.append({
            "set": label, "ep": int(ep), "task": task, "frames": int(len(g)),
            "fps_est": round(float(fps_est), 2), "home_n": S[0].tolist(),
            "release_frame": rel, "grasp_frame": grasp,
            "place_n": S[rel].tolist() if rel is not None else None,
            "grasp_n": S[grasp].tolist() if grasp is not None else None,
            "grip_min": float(grip.min()), "grip_max": float(grip.max()),
            "vmax_n_s": np.abs(V).max(axis=0).tolist(),
            "vp99_n_s": np.percentile(np.abs(V), 99, axis=0).tolist(),
            "jerk_rms_n": np.sqrt((Jk ** 2).mean(axis=0)).tolist(),
        })
    return rows


def main():
    sim = sim_rows()
    real = real_rows(DS_RAW, "raw6d6") + real_rows(DS_CUR, "cur_a4ba")
    json.dump({"sim": sim, "real": real, "K": K}, open(OUT + "/rows.json", "w"))
    acc = sum(r["accepted"] for r in sim)
    print("SIM_N", len(sim), "accepted", acc, "grasp_verified", sum(r["grasp_verified"] for r in sim))
    print("SIM_BRICK", {b: sum(r["brick"] == b for r in sim) for b in ("2x2", "2x4")})
    print("SIM_DEST", {d: sum(r["dest"] == d for r in sim) for d in ("blue", "green")})
    print("SIM_COLOR", {c: sum(r["color"] == c for r in sim) for c in sorted({r["color"] for r in sim})})
    print("SIM_OD10_CONTRADICTIONS", sum(r["od10_contradiction"] for r in sim))
    print("SIM_TASK_MATCH_FINAL_ON_DEST_xy<0.015", sum(r["final_xy_err_m"] < 0.015 for r in sim),
          "max_xy_err", round(max(r["final_xy_err_m"] for r in sim), 4))
    print("SIM_FINAL_Z mean/min/max", round(np.mean([r["final_z_m"] for r in sim]), 4),
          round(min(r["final_z_m"] for r in sim), 4), round(max(r["final_z_m"] for r in sim), 4))
    print("SIM_FRAMES total", sum(r["frames"] for r in sim), "dur_s median", round(float(np.median([r["dur_s"] for r in sim])), 2))
    print("K_per_rad", {j: round(K[j], 2) for j in JOINTS})
    print("REAL_N", len(real), [ (r["set"], r["ep"], r["frames"], r["task"][:40], r["release_frame"], r["grasp_frame"]) for r in real])


if __name__ == "__main__":
    main()
