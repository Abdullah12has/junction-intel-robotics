#!/usr/bin/env python3
"""P3-TWIN-CAMS E1 step 4: camera-to-joint clock offset from motion onsets (top and close recordings).

Joint events: a move starts when the largest per-joint speed (pos0..pos5, normalized units/s, measured positions)
rises above SPEED_ON after at least STILL_S seconds below SPEED_OFF. Gripper-only events are those where pos5 moves
and the arm joints do not (speed of pos0..pos4 < SPEED_OFF in the window).
Video: per-frame global motion energy (out/motion_<cam>.npz from s2_motion.py) on a 30 Hz grid.
Per event: normalized cross-correlation of joint speed and video motion for lags -0.5..+0.5 s over
[t_j - 1.0, t_j + 1.0]; lag* = argmax. Offset convention: video_time = joint_time + offset, so offset = lag*.
Run: envs/data-env/bin/python -I tools/s4_offsets.py
"""
import glob
import json
import os

import numpy as np

R = "/home/intel/intel-junction-2026"
E1 = f"{R}/work/P3-TWIN-CAMS/E1"
SPEED_ON = 8.0      # units/s
SPEED_OFF = 2.0     # units/s
STILL_S = 0.5
GRID = 1.0 / 30.0
MAXLAG = 15         # frames (+-0.5 s)


def load_joints(path):
    a = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    t = a["t"]
    pos = np.stack([a[f"pos{i}"] for i in range(6)], axis=1)
    return t, pos


def events(t, pos):
    dt = np.maximum(np.diff(t), 1e-3)
    dp = np.abs(np.diff(pos, axis=0)) / dt[:, None]
    v_all = dp.max(axis=1)
    v_arm = dp[:, :5].max(axis=1)
    tv = t[1:]
    out = []
    n = len(v_all)
    nstill = max(int(round(STILL_S / np.median(dt))), 1)
    last = -1e9
    for k in range(nstill, n):
        if v_all[k] > SPEED_ON and v_all[k - 1] <= SPEED_ON:
            if np.all(v_all[k - nstill:k] < SPEED_OFF) and tv[k] - last > 2.0:
                kind = "arm" if v_arm[k:k + 5].max() > SPEED_ON else "gripper"
                out.append((float(tv[k]), kind))
                last = tv[k]
    return out


def xcorr_lag(sj, vm):
    best = (-2.0, 0)
    for lag in range(-MAXLAG, MAXLAG + 1):
        a = sj[max(0, lag):len(sj) + min(0, lag)]
        b = vm[max(0, -lag):len(vm) + min(0, -lag)]
        m = min(len(a), len(b))
        if m < 20:
            continue
        a = a[:m] - a[:m].mean()
        b = b[:m] - b[:m].mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        if den <= 0:
            continue
        c = float((a * b).sum() / den)
        if c > best[0]:
            best = (c, lag)
    return best


def summarize(x):
    x = np.asarray(x, float)
    med = float(np.median(x))
    mad = float(1.4826 * np.median(np.abs(x - med)))
    q1, q3 = np.percentile(x, [25, 75])
    return {"n": int(x.size), "median": med, "robust_sd_mad": mad, "iqr": float(q3 - q1),
            "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90))}


def main():
    files = sorted(glob.glob(f"{R}/logs/sort/any_*/ep_*/joints.csv"))
    ev = []
    for f in files:
        ep = os.path.relpath(os.path.dirname(f), f"{R}/logs/sort")
        t, pos = load_joints(f)
        for tj, kind in events(t, pos):
            ev.append({"episode": ep, "t_joint": tj, "kind": kind})
    result = {"speed_on": SPEED_ON, "speed_off": SPEED_OFF, "still_s": STILL_S, "events_total": len(ev),
              "events_by_kind": {k: sum(1 for e in ev if e["kind"] == k) for k in ("arm", "gripper")},
              "cameras": {}}
    for cam in ("top", "close"):
        d = np.load(f"{E1}/out/motion_{cam}.npz")
        pv, mv = d["pts"], d["motion"]
        rows = []
        for e in ev:
            t0 = e["t_joint"] - 1.0
            t1 = e["t_joint"] + 1.0
            grid = np.arange(t0, t1, GRID)
            if grid[0] < pv[0] or grid[-1] > pv[-1]:
                continue
            # joint speed (all six joints) on the grid
            tj_ = np.linspace(t0, t1, 2)
            vm_g = np.interp(grid, pv, mv)
            sj = np.interp(grid, [t0, t1], [0.0, 0.0])  # placeholder, replaced below
            e["_grid"] = grid
            rows.append((e, vm_g))
        # joint speed for the same grid, recomputed per episode file
        out_rows = []
        for e, vm_g in rows:
            f = f"{R}/logs/sort/{e['episode']}/joints.csv"
            t, pos = load_joints(f)
            dt = np.maximum(np.diff(t), 1e-3)
            sp = (np.abs(np.diff(pos, axis=0)) / dt[:, None]).max(axis=1)
            sj = np.interp(e["_grid"], t[1:], sp)
            c, lag = xcorr_lag(sj, vm_g)
            out_rows.append({"episode": e["episode"], "kind": e["kind"], "t_joint": round(e["t_joint"], 4),
                             "lag_frames": lag, "offset_s": round(lag * GRID, 4), "corr": round(c, 4)})
        offs = [r["offset_s"] for r in out_rows if r["corr"] > 0.3]
        arm = [r["offset_s"] for r in out_rows if r["corr"] > 0.3 and r["kind"] == "arm"]
        grp = [r["offset_s"] for r in out_rows if r["corr"] > 0.3 and r["kind"] == "gripper"]
        result["cameras"][cam] = {
            "events_used_corr_gt_0p3": len(offs),
            "events_considered": len(out_rows),
            "all": summarize(offs) if offs else None,
            "arm_moves": summarize(arm) if arm else None,
            "gripper_moves": summarize(grp) if grp else None,
            "events": out_rows,
        }
        print(cam, "considered", len(out_rows), "used", len(offs),
              "median_offset_s", round(float(np.median(offs)), 4) if offs else None,
              "mad_s", round(summarize(offs)["robust_sd_mad"], 4) if offs else None)
    with open(f"{E1}/out/clock_offsets_stage1.json", "w") as fh:
        json.dump(result, fh, indent=1)
    print("events", len(ev), result["events_by_kind"])


if __name__ == "__main__":
    main()
