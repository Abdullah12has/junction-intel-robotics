import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq

R = "/home/intel/intel-junction-2026"
D = R + "/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af"
OUT = R + "/work/P2-D-DATA/V2/out"
os.makedirs(OUT, exist_ok=True)
FPS = 30.0
info = json.load(open(D + "/meta/info.json"))
NAMES = info["features"]["observation.state"]["names"]
task_text = {r["task_index"]: r["task"] for r in pq.read_table(D + "/meta/tasks.parquet").to_pylist()}
eps = []
for f in sorted(glob.glob(D + "/meta/episodes/chunk-*/file-*.parquet")):
    eps += pq.read_table(f).to_pylist()
eps.sort(key=lambda r: r["episode_index"])
S_l, A_l, F_l, E_l, T_l, TS_l = [], [], [], [], [], []
for f in sorted(glob.glob(D + "/data/chunk-*/file-*.parquet")):
    t = pq.read_table(f, columns=["action", "observation.state", "timestamp", "frame_index", "episode_index", "task_index"]).to_pydict()
    S_l.append(np.array(t["observation.state"], dtype=float))
    A_l.append(np.array(t["action"], dtype=float))
    F_l.append(np.array(t["frame_index"]))
    E_l.append(np.array(t["episode_index"]))
    T_l.append(np.array(t["task_index"]))
    TS_l.append(np.array(t["timestamp"], dtype=float))
S = np.vstack(S_l)
A = np.vstack(A_l)
F = np.concatenate(F_l)
E = np.concatenate(E_l)
TI = np.concatenate(T_l)
TS = np.concatenate(TS_l)
order = np.lexsort((F, E))
S, A, F, E, TI, TS = S[order], A[order], F[order], E[order], TI[order], TS[order]


def runs(mask, minlen):
    out = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= minlen:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


per = {}
starts = []
for m in eps:
    e = int(m["episode_index"])
    idx = np.where(E == e)[0]
    s = S[idx]
    N = len(idx)
    starts.append(np.median(s[:15, :5], axis=0))
home = np.median(np.array(starts), axis=0)
for m in eps:
    e = int(m["episode_index"])
    idx = np.where(E == e)[0]
    s = S[idx]
    f = F[idx]
    ti = TI[idx]
    ts = TS[idx]
    N = len(idx)
    g = s[:, 5]
    spd = np.zeros(N)
    spd[1:] = np.max(np.abs(np.diff(s[:, :5], axis=0)), axis=1)
    rec = {
        "episode": e,
        "length_meta": int(m["length"]),
        "length_data": N,
        "tasks_meta": list(m["tasks"]),
        "task_idx_data": sorted(set(int(x) for x in ti)),
        "frame_ok": bool(np.array_equal(f, np.arange(N))),
        "ts_max_dev_s": round(float(np.max(np.abs(ts - f / FPS))), 4),
        "dur_s": round(N / FPS, 3),
        "g_min": round(float(g.min()), 2),
        "g_max": round(float(g.max()), 2),
        "start_pose": np.median(s[:15, :5], axis=0).round(2).tolist(),
        "end_pose": np.median(s[-15:, :5], axis=0).round(2).tolist(),
        "start_home_dist": round(float(np.max(np.abs(np.median(s[:15, :5], axis=0)[1:4] - home[1:4]))), 2),
        "end_home_dist": round(float(np.max(np.abs(np.median(s[-15:, :5], axis=0)[1:4] - home[1:4]))), 2),
        "speed_p50": round(float(np.median(spd[1:])), 3),
        "speed_p95": round(float(np.percentile(spd[1:], 95)), 3),
    }
    grasps = []
    for (c0, c1) in runs(g < 4.0, 10):
        gr = {"c0": c0, "c1": c1, "start_closed": c0 == 0}
        if c0 > 0:
            k = c0
            while k > 0 and g[k - 1] > g[k] + 0.3:
                k -= 1
            gr["onset"] = k
            gr["open_peak_2s"] = round(float(g[max(0, k - 60):k + 1].max()), 2)
        else:
            gr["onset"] = None
        gr["closed_med"] = round(float(np.median(g[c0 + 5:c1 - 5] if c1 - c0 > 12 else g[c0:c1])), 3)
        gr["closed_min"] = round(float(g[c0:c1].min()), 3)
        if c1 < N:
            gr["release"] = c1
            gr["release_pose"] = s[c1, :5].round(2).tolist()
            gr["release_g_peak_1s"] = round(float(g[c1:c1 + 30].max()), 2)
            gr["approach_pose_0p5_1s"] = np.median(s[max(0, c1 - 30):max(0, c1 - 15), :5], axis=0).round(2).tolist()
            ret = None
            for t in range(c1, N):
                if np.max(np.abs(s[t, 1:4] - home[1:4])) < 8.0:
                    ret = t
                    break
            gr["return_home_idx"] = ret
        else:
            gr["release"] = None
        grasps.append(gr)
    rec["closed_runs"] = grasps
    per[e] = rec
out = {"home_median_start": home.round(2).tolist(), "joint_names": NAMES, "episodes": per}
with open(OUT + "/partA.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
for e in sorted(per):
    r = per[e]
    print("EP", e, "len", r["length_data"], "meta", r["length_meta"], "frame_ok", r["frame_ok"], "tsdev", r["ts_max_dev_s"],
          "tasks", r["tasks_meta"], "idx", r["task_idx_data"], "gmin", r["g_min"], "gmax", r["g_max"])
    print("  start", r["start_pose"], "dHome0", r["start_home_dist"], "end", r["end_pose"], "dHomeEnd", r["end_home_dist"],
          "spd p50 p95", r["speed_p50"], r["speed_p95"])
    for gr in r["closed_runs"]:
        print("  RUN c0", gr["c0"], "c1", gr["c1"], "startClosed", gr["start_closed"], "onset", gr["onset"],
              "openpk2s", gr.get("open_peak_2s"), "closedMed", gr["closed_med"], "closedMin", gr["closed_min"],
              "rel", gr["release"], "relPose", gr.get("release_pose"), "relGpk", gr.get("release_g_peak_1s"),
              "ret", gr.get("return_home_idx"))
print("HOME", home.round(2).tolist())
