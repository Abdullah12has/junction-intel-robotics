#!/usr/bin/env python3
"""P3-TWIN-CAMS E1 step 1: still-arm windows from every sort joints.csv (read-only inputs).

A window starting at sample j is still when, over [t_j, t_j + WIN], the max-min range of every joint pos0..pos5
(normalized units, measured positions, not goals) is below THR. Consecutive still starts are merged into one still
segment that runs from the first start to the end of the last window.
Run: envs/data-env/bin/python -I tools/s1_windows.py
"""
import glob
import json
import os

import numpy as np

R = "/home/intel/intel-junction-2026"
OUT = f"{R}/work/P3-TWIN-CAMS/E1/out"
THR = 0.5   # normalized units, max-min over the window, all six measured joints
WIN = 0.5   # seconds


def load(path):
    a = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    t = a["t"]
    pos = np.stack([a[f"pos{i}"] for i in range(6)], axis=1)
    return t, pos


def still_segments(t, pos):
    n = len(t)
    k_end = np.searchsorted(t, t + WIN)
    ok = np.zeros(n, dtype=bool)
    for j in range(n):
        k = k_end[j]
        if k >= n:
            break
        seg = pos[j:k + 1]
        if np.all(seg.max(axis=0) - seg.min(axis=0) < THR):
            ok[j] = True
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return []
    segs = []
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        i0 = int(run[0])
        i1 = int(min(k_end[run[-1]], n - 1))
        seg_pos = pos[i0:i1 + 1]
        segs.append({
            "t0": float(t[i0]), "t1": float(t[i1]), "dur": float(t[i1] - t[i0]),
            "n_samples": int(i1 - i0 + 1),
            "pose_median": [round(float(v), 3) for v in np.median(seg_pos, axis=0)],
            "pose_range": [round(float(v), 3) for v in (seg_pos.max(axis=0) - seg_pos.min(axis=0))],
        })
    return segs


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(f"{R}/logs/sort/any_*/ep_*/joints.csv"))
    allsegs = []
    per_ep = {}
    total_samples = 0
    for f in files:
        ep = os.path.relpath(os.path.dirname(f), f"{R}/logs/sort")
        t, pos = load(f)
        total_samples += len(t)
        dt = np.diff(t)
        segs = still_segments(t, pos)
        for s in segs:
            s["episode"] = ep
        allsegs.extend(segs)
        per_ep[ep] = {"samples": int(len(t)), "t_first": float(t[0]), "t_last": float(t[-1]),
                      "median_dt_s": round(float(np.median(dt)), 4), "still_segments": len(segs),
                      "still_s": round(float(sum(s["dur"] for s in segs)), 2)}
    out = {"thr_units": THR, "win_s": WIN, "joints_files": len(files), "samples": total_samples,
           "segments": allsegs, "per_episode": per_ep}
    with open(f"{OUT}/still_windows.json", "w") as fh:
        json.dump(out, fh, indent=1)
    durs = np.array([s["dur"] for s in allsegs])
    print("joints files", len(files), "samples", total_samples, "still segments", len(allsegs))
    if durs.size:
        print("still duration s: min %.2f median %.2f max %.2f total %.1f" % (
            durs.min(), np.median(durs), durs.max(), durs.sum()))
    for ep, v in per_ep.items():
        print(ep, "samples", v["samples"], "dt_med", v["median_dt_s"], "segments", v["still_segments"],
              "still_s", v["still_s"])


if __name__ == "__main__":
    main()
