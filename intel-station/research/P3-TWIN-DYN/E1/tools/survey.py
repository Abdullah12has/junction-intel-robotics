#!/usr/bin/env python3
"""P3-TWIN-DYN E1: read-only survey of every logs/sort/any_17*/ep_*/joints.csv (t, goal0..5, pos0..5)."""
import csv
import glob
import json

import numpy as np

R = "/home/intel/intel-junction-2026"
OUT = f"{R}/work/P3-TWIN-DYN/E1"


def load(path):
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        rows = [r for r in rd if r and len(r) == len(header)]
    return header, np.array(rows, dtype=float)


def main():
    files = sorted(glob.glob(f"{R}/logs/sort/any_17*/ep_*/joints.csv"))
    print("files", len(files))
    summary = {}
    for k, f in enumerate(files):
        header, a = load(f)
        if k == 0:
            print("header", header)
        name = "/".join(f.split("/")[-3:-1])
        t = a[:, 0]
        goal = a[:, 1:7]
        pos = a[:, 7:13]
        dt = np.diff(t)
        moving = np.any(np.abs(np.diff(goal, axis=0)) > 1e-6, axis=1)
        vel = np.diff(goal, axis=0) / dt[:, None]
        acc = np.diff(vel, axis=0) / dt[1:, None]
        e = {
            "rows": int(len(t)),
            "duration_s": float(t[-1] - t[0]),
            "dt_median_s": float(np.median(dt)),
            "dt_min_s": float(dt.min()),
            "dt_max_s": float(dt.max()),
            "gaps_gt_0.1s": int(np.sum(dt > 0.1)),
            "goal_change_rows": int(moving.sum()),
            "vmax_goal_units_s": [round(float(x), 2) for x in np.max(np.abs(vel), axis=0)],
            "amax_goal_units_s2": [round(float(x), 1) for x in np.max(np.abs(acc), axis=0)],
            "pos_min": [round(float(x), 2) for x in pos.min(axis=0)],
            "pos_max": [round(float(x), 2) for x in pos.max(axis=0)],
            "goal_min": [round(float(x), 2) for x in goal.min(axis=0)],
            "goal_max": [round(float(x), 2) for x in goal.max(axis=0)],
            "end_goal_minus_pos_last15": [round(float(x), 2) for x in (goal[-15:] - pos[-15:]).mean(axis=0)],
        }
        summary[name] = e
        print(f"{name} rows={e['rows']} dur={e['duration_s']:.1f}s dt_med={e['dt_median_s']*1000:.1f}ms "
              f"dtmax={e['dt_max_s']*1000:.0f}ms gaps={e['gaps_gt_0.1s']} moving={e['goal_change_rows']}")
    with open(f"{OUT}/survey.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print("total rows", sum(v["rows"] for v in summary.values()),
          "total duration s", round(sum(v["duration_s"] for v in summary.values()), 1))
    print("wrote", f"{OUT}/survey.json")


if __name__ == "__main__":
    main()
