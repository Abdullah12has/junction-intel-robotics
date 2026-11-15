#!/usr/bin/env python3
"""E3: what the real gripper does when closing. For each logged closing episode (goal5 == 0 while closed) report the
gripper reading plateaus (pos5 while goal5 is 0): empty stop, brick stop, and how far below the held plateau the
reading goes. Output: outputs/gripper_logs.json. Run with: data-env/bin/python -I -B gripper_logs.py"""
import csv
import glob
import json
import os

import numpy as np

R = "/home/intel/intel-junction-2026"
OUT = f"{R}/work/P3-TWIN-ROBOT/E3/outputs/gripper_logs.json"


def load(path):
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        rows = [r for r in rd if r and len(r) == len(header)]
    return header, np.array(rows, dtype=float)


def main():
    eps = []
    for f in sorted(glob.glob(f"{R}/logs/sort/any_17*/ep_*/joints.csv")):
        header, a = load(f)
        if a.shape[1] < 13:
            continue
        t = a[:, 0] - a[0, 0]
        goal5, pos5 = a[:, 6], a[:, 12]
        closed = goal5 <= 1.0
        if not closed.any():
            continue
        idx = np.where(closed)[0]
        seg = pos5[idx]
        # plateau: values reached while commanded closed, after the first 0.5 s of closing
        t_close = t[idx[0]]
        m = (t >= t_close + 0.5) & closed
        if m.sum() < 5:
            continue
        p = pos5[m]
        vals, counts = np.unique(np.round(p, 2), return_counts=True)
        top = vals[np.argsort(-counts)[:3]].tolist()
        eps.append({"episode": "/".join(f.split("/")[-3:-1]), "rows_closed": int(m.sum()),
                    "closed_pos5_min": round(float(p.min()), 3), "closed_pos5_median": round(float(np.median(p)), 3),
                    "closed_pos5_top_plateaus": top, "closed_pos5_max": round(float(p.max()), 3),
                    "goal5_when_closed_unique": sorted(set(np.round(goal5[m], 2).tolist()))[:5],
                    "min_pos5_any_row": round(float(seg.min()), 3)})
    med = [e["closed_pos5_median"] for e in eps]
    summary = {"episodes_with_closing": len(eps),
               "median_of_closed_medians": round(float(np.median(med)), 3) if med else None}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"summary": summary, "episodes": eps}, open(OUT, "w"), indent=2)
    print(json.dumps(summary))
    for e in eps[:40]:
        print(e["episode"], e["closed_pos5_median"], e["closed_pos5_top_plateaus"], e["goal5_when_closed_unique"])


if __name__ == "__main__":
    main()
