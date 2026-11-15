#!/usr/bin/env python3
"""E1 gripper check. kin.fk uses the five body joints, so the gripper is outside G1. This script reads the logged
gripper columns (goal5, pos5) of the 17 episodes that have joints.csv, reports them, and tests the three candidate
gripper laws in units.py against the URDF gripper limits. Writes gripper_check.json in E1.
Run: nice -n 19 $DATA_PY -I -B gripper_check.py"""
import csv
import datetime
import glob
import json
import math
import os
import sys

import numpy as np

R = "/home/intel/intel-junction-2026"
E1 = f"{R}/work/P3-TWIN-ROBOT/E1"
sys.path.insert(0, f"{R}/sim/twin/robot")
import units  # noqa: E402

LOG_ROOT = f"{R}/logs/sort"
SUMMARY_EMPTY_CLOSE = 0.59      # value from the earlier log summary; compared against the logs, not assumed
SUMMARY_HELD = (1.57, 2.42)
SUMMARY_OPEN = 25.0


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load(path):
    goal, pos = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            goal.append(float(row["goal5"]))
            pos.append(float(row["pos5"]))
    return np.array(goal), np.array(pos)


def summ(x):
    if x.size == 0:
        return None
    return {"n": int(x.size), "median": float(np.median(x)), "min": float(x.min()), "max": float(x.max())}


def main():
    all_dirs = sorted(glob.glob(f"{LOG_ROOT}/any_17*/ep_*"))
    files = sorted(glob.glob(f"{LOG_ROOT}/any_17*/ep_*/joints.csv"))
    assert len(files) == 17 and len(all_dirs) == 22, (len(files), len(all_dirs))
    missing = [os.path.relpath(d, LOG_ROOT) for d in all_dirs if not os.path.isfile(os.path.join(d, "joints.csv"))]

    per, goals, poss = [], [], []
    for f in files:
        g, p = load(f)
        per.append({"episode": os.path.relpath(os.path.dirname(f), LOG_ROOT), "rows": int(len(p)),
                    "goal5_min": float(g.min()), "goal5_max": float(g.max()), "pos5_min": float(p.min()),
                    "pos5_median": float(np.median(p)), "pos5_max": float(p.max())})
        goals.append(g)
        poss.append(p)
    G = np.concatenate(goals)
    P = np.concatenate(poss)

    bins = {
        "pos5_lt_1": int(np.sum(P < 1.0)),
        "pos5_1_to_1.57": int(np.sum((P >= 1.0) & (P < 1.57))),
        "pos5_1.57_to_2.42": int(np.sum((P >= 1.57) & (P <= 2.42))),
        "pos5_2.42_to_24": int(np.sum((P > 2.42) & (P < 24.0))),
        "pos5_24_to_26": int(np.sum((P >= 24.0) & (P <= 26.0))),
        "pos5_gt_26": int(np.sum(P > 26.0)),
    }
    closed_cmd = G <= 1.0
    empty_mask = closed_cmd & (P >= 0.0) & (P <= 1.0)
    held_mask = closed_cmd & (P >= SUMMARY_HELD[0]) & (P <= SUMMARY_HELD[1])
    open_pos = np.abs(P - SUMMARY_OPEN) <= 0.5
    open_goal = np.abs(G - SUMMARY_OPEN) <= 0.5
    empty_s, held_s, open_s = summ(P[empty_mask]), summ(P[held_mask]), summ(P[open_pos])

    lo_q, hi_q = units.URDF_LIMITS["gripper"]
    ref_empty = empty_s["median"] if empty_s else SUMMARY_EMPTY_CLOSE
    laws = {}
    for law, (offset, n0) in units.GRIPPER_LAWS.items():
        n_hi = units.gripper_rad_to_norm(hi_q, law)
        n_lo = units.gripper_rad_to_norm(lo_q, law)
        q_empty = units.gripper_norm_to_rad(ref_empty, law)
        laws[law] = {
            "offset_rad": offset, "n0": n0,
            "n_at_urdf_upper_limit": n_hi, "n_at_urdf_lower_limit": n_lo,
            "logged_pos5_beyond_urdf_limits": int(np.sum((P > n_hi) | (P < n_lo))),
            "logged_goal5_beyond_urdf_limits": int(np.sum((G > n_hi) | (G < n_lo))),
            "empty_close_rad": q_empty,
            "empty_close_above_closed_stop_deg": math.degrees(q_empty - lo_q),
        }

    checks = [
        {"check": "empty close reading near 0.59 (commanded closed, pos5 in [0, 1])",
         "observed": empty_s,
         "pass": bool(empty_s is not None and abs(empty_s["median"] - SUMMARY_EMPTY_CLOSE) <= 0.05)},
        {"check": "held readings in 1.57 to 2.42 (commanded closed)", "observed": held_s,
         "pass": bool(held_s is not None),
         "flag": "[unclear]: inferred from the logged values only; not checked against the images"},
        {"check": "open readings near 25 (pos5)", "observed": open_s, "pass": bool(open_s is not None),
         "goal5_rows_near_25": int(open_goal.sum())},
    ]
    notes = [
        "The gripper is not in G1: kin.fk takes the five body joints only.",
        "Closed-stop test: a law is plausible if the empty-close reading sits near the URDF closed stop "
        "(-0.174533 rad). closed_at_empty_reading is on the stop by construction, so it is not evidence.",
        "No jaw-gap measurement was made (no hardware or Studio access). The law choice stays open for Opus.",
    ]
    doc = {
        "utc": utc_now(), "files_read": len(files), "episode_dirs": len(all_dirs),
        "episode_dirs_without_joints_csv": missing, "rows": int(len(P)),
        "pos5": {"min": float(P.min()), "p1": float(np.percentile(P, 1)), "median": float(np.median(P)),
                 "p99": float(np.percentile(P, 99)), "max": float(P.max())},
        "goal5": {"min": float(G.min()), "median": float(np.median(G)), "max": float(G.max()),
                  "distinct_values": int(len(np.unique(G)))},
        "bins_pos5": bins, "closed_command_rows": int(closed_cmd.sum()),
        "empty_close_pos5": empty_s, "held_pos5_when_commanded_closed": held_s, "open_pos5": open_s,
        "open_goal5_rows": int(open_goal.sum()),
        "laws": laws, "checks": checks, "per_episode": per, "notes": notes,
    }
    with open(os.path.join(E1, "gripper_check.json"), "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print("rows", doc["rows"], "episodes", len(files), "dirs without joints.csv", len(missing))
    print("empty close", empty_s, "held", held_s, "open", open_s)
    for law, v in laws.items():
        print(law, "n_upper", round(v["n_at_urdf_upper_limit"], 2), "n_lower", round(v["n_at_urdf_lower_limit"], 2),
              "empty_close_above_stop_deg", round(v["empty_close_above_closed_stop_deg"], 2),
              "pos5_beyond", v["logged_pos5_beyond_urdf_limits"])


if __name__ == "__main__":
    main()
