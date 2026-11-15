#!/usr/bin/env python3
"""Measure every episode of a LeRobot v3 dataset from its state parquet (read-only).

Usage: python3 -I analyze.py <dataset_dir> <episodes_csv> <out_json>
The episodes_csv is the live episode table written by the dataset watcher (gripper events, release frame, labels).
"""
import csv
import glob
import json
import sys

import numpy as np
import pyarrow.parquet as pq

HOME = np.array([-6.6, -98.1, 99.4, 54.3, -53.1])  # arm joints at rest, live probe 14:13Z (shared/FACTS-LIVE.md)
MOVE_THR = 0.08  # max joint change per frame (5-frame mean) that counts as motion
REST_TOL = 12.0  # max joint distance from HOME (units) that counts as back at rest
PAD = 15  # 0.5 s at 30 fps kept around the motion


def load_states(ds):
    rows_by_ep = {}
    for path in sorted(glob.glob(ds + "/data/chunk-*/file-*.parquet")):
        cols = pq.read_table(path, columns=["episode_index", "frame_index", "observation.state"]).to_pydict()
        for ep, fi, st in zip(cols["episode_index"], cols["frame_index"], cols["observation.state"]):
            rows_by_ep.setdefault(int(ep), []).append((int(fi), st))
    states = {}
    for ep, rows in rows_by_ep.items():
        rows.sort(key=lambda r: r[0])
        states[ep] = np.array([r[1] for r in rows], dtype=float)
    return states


def destination(pan):
    if pan > 0:
        return "BLUE"
    if pan < -30:
        return "GREEN"
    return "AMBIG"


def main():
    ds, csv_path, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
    states = load_states(ds)
    table = {int(r["episode_index"]): r for r in csv.DictReader(open(csv_path, encoding="utf-8"))}
    result = []
    for ep in sorted(states):
        S = states[ep]
        n = len(S)
        arm = S[:, :5]
        d_home = np.abs(arm - HOME).max(axis=1)
        vel = np.concatenate([[0.0], np.abs(np.diff(arm, axis=0)).max(axis=1)])
        sm = np.convolve(vel, np.ones(5) / 5, mode="same")
        moving = np.where(sm > MOVE_THR)[0]
        ms, me = (int(moving[0]), int(moving[-1])) if len(moving) else (0, n - 1)
        t = table.get(ep, {})
        rel_raw = t.get("release", "")
        rec = {
            "episode": ep,
            "frames": n,
            "start_dist": round(float(d_home[0]), 2),
            "end_dist": round(float(d_home[-1]), 2),
            "settled_end": bool(sm[-15:].max() < MOVE_THR),
            "motion_start": ms,
            "motion_end": me,
            "cut_start": max(0, ms - PAD),
            "cut_end": min(n, me + PAD + 1),
            "table_n_grasps": t.get("n_grasps", ""),
            "table_closed_val": t.get("closed_val", ""),
            "table_release_label": t.get("release_label", ""),
            "table_type": t.get("type", ""),
            "release_frame": int(rel_raw) if rel_raw else None,
        }
        if rec["release_frame"] is not None and rec["release_frame"] < n:
            r = rec["release_frame"]
            rec["pan_at_release"] = round(float(arm[r, 0]), 2)
            rec["lift_at_release"] = round(float(arm[r, 1]), 2)
            rec["elbow_at_release"] = round(float(arm[r, 2]), 2)
            rec["grip_at_release"] = round(float(S[r, 5]), 2)
            rec["destination_by_pan"] = destination(float(arm[r, 0]))
            rec["min_dist_after_release"] = round(float(d_home[r:].min()), 2)
        result.append(rec)
        print(
            f"ep{ep:02d} n={n} start={rec['start_dist']} end={rec['end_dist']} settled={rec['settled_end']} "
            f"move={ms}-{me} cut={rec['cut_start']}-{rec['cut_end']} rel={rec['release_frame']} "
            f"pan_rel={rec.get('pan_at_release')} dest={rec.get('destination_by_pan')} "
            f"minD_after={rec.get('min_dist_after_release')} grasps={rec['table_n_grasps']} "
            f"closed={rec['table_closed_val']} label={rec['table_release_label']} type={rec['table_type']}"
        )
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print("wrote", out_json, len(result), "episodes")


if __name__ == "__main__":
    main()
