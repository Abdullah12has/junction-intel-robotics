#!/usr/bin/env python3
"""P3-TWIN-CAMS E1 step 5: refined camera-to-joint clock offsets (top, close) from motion onsets.

Signals on a 30 Hz grid: joint activity a(t) = log1p(max_i |dpos_i/dt|), i = pos0..pos5 (measured), and video
activity v(t) = log(motion energy + 0.05) from out/motion_<cam>.npz (mean abs frame difference, 160x120 gray).
(1) Per episode: normalized cross-correlation of a(t) and v(t) over the whole episode, lags -0.5..+0.5 s.
(2) Per move event (move starts from still, as in s4_offsets.py): same correlation over [t_j-1.5, t_j+1.5].
Offset convention: video_time = joint_time + offset; offset = lag* / 30 s.
Run: envs/data-env/bin/python -I tools/s5_offsets2.py
"""
import glob
import json
import os

import numpy as np

R = "/home/intel/intel-junction-2026"
E1 = f"{R}/work/P3-TWIN-CAMS/E1"
GRID = 1.0 / 30.0
MAXLAG = 15
sys_path_s4 = f"{E1}/tools"
import sys  # noqa: E402

sys.path.insert(0, sys_path_s4)
from s4_offsets import events, load_joints, summarize  # noqa: E402


def xcorr_peak(a, v, maxlag=MAXLAG):
    best = (-2.0, 0)
    corrs = {}
    for lag in range(-maxlag, maxlag + 1):
        if lag >= 0:
            x, y = a[:len(a) - lag], v[lag:]
        else:
            x, y = a[-lag:], v[:len(v) + lag]
        m = min(len(x), len(y))
        if m < 30:
            continue
        x = x[:m] - x[:m].mean()
        y = y[:m] - y[:m].mean()
        den = np.sqrt((x * x).sum() * (y * y).sum())
        if den <= 0:
            continue
        c = float((x * y).sum() / den)
        corrs[lag] = c
        if c > best[0]:
            best = (c, lag)
    return best


def joint_activity(t, pos, grid):
    dt = np.maximum(np.diff(t), 1e-3)
    sp = (np.abs(np.diff(pos, axis=0)) / dt[:, None]).max(axis=1)
    return np.log1p(np.interp(grid, t[1:], sp))


def main():
    res = {"method": "log-activity cross-correlation, lags -0.5..+0.5 s, offset = video - joint",
           "cameras": {}}
    files = sorted(glob.glob(f"{R}/logs/sort/any_*/ep_*/joints.csv"))
    ep_data = {}
    for f in files:
        ep = os.path.relpath(os.path.dirname(f), f"{R}/logs/sort")
        ep_data[ep] = load_joints(f)
    ev_all = []
    for ep, (t, pos) in ep_data.items():
        for tj, kind in events(t, pos):
            ev_all.append({"episode": ep, "t_joint": tj, "kind": kind})
    for cam in ("top", "close"):
        d = np.load(f"{E1}/out/motion_{cam}.npz")
        pv, mv = d["pts"], d["motion"]
        lv = np.log(mv + 0.05)
        per_ep = []
        for ep, (t, pos) in ep_data.items():
            grid = np.arange(t[0] + 0.5, t[-1] - 0.5, GRID)
            a = joint_activity(t, pos, grid)
            v = np.interp(grid, pv, lv)
            c, lag = xcorr_peak(a, v)
            per_ep.append({"episode": ep, "lag_frames": lag, "offset_s": round(lag * GRID, 4), "corr": round(c, 4)})
        ev_rows = []
        for e in ev_all:
            t, pos = ep_data[e["episode"]]
            t0, t1 = e["t_joint"] - 1.5, e["t_joint"] + 1.5
            if t0 < pv[0] or t1 > pv[-1] or t0 < t[0] or t1 > t[-1]:
                continue
            grid = np.arange(t0, t1, GRID)
            a = joint_activity(t, pos, grid)
            v = np.interp(grid, pv, lv)
            c, lag = xcorr_peak(a, v)
            ev_rows.append({"episode": e["episode"], "kind": e["kind"], "t_joint": round(e["t_joint"], 4),
                            "lag_frames": lag, "offset_s": round(lag * GRID, 4), "corr": round(c, 4)})
        good = [r["offset_s"] for r in ev_rows if r["corr"] >= 0.6]
        ep_offs = [r["offset_s"] for r in per_ep if r["corr"] >= 0.3]
        res["cameras"][cam] = {
            "per_episode": per_ep,
            "per_episode_summary": summarize(ep_offs),
            "events_considered": len(ev_rows),
            "events_corr_ge_0p6": len(good),
            "per_event_summary_corr_ge_0p6": summarize(good) if good else None,
            "events": ev_rows,
        }
        print(cam, "episodes", len(per_ep), "episode_median_s", round(float(np.median(ep_offs)), 4),
              "episode_mad_s", round(summarize(ep_offs)["robust_sd_mad"], 4))
        if good:
            s = summarize(good)
            print(cam, "events", len(ev_rows), "corr>=0.6", len(good), "median_s", round(s["median"], 4),
                  "mad_s", round(s["robust_sd_mad"], 4))
        print(cam, "episode lags", [r["lag_frames"] for r in per_ep])
    with open(f"{E1}/out/clock_offsets_stage2.json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
