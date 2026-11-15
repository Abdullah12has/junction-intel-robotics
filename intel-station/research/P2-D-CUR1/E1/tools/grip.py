#!/usr/bin/env python3
"""Verify grasp, hold and release of every episode from the gripper and arm state (read-only).

Usage: python3 -I grip.py <dataset_dir> <out_json>
Closure = gripper falls below 5 after it was open (above 8). Hold = until the gripper opens above 8 again
(release). Brick grasped = hold median in [1.9, 2.9] and hold longer than 1 s. Destination = pan sign at release
(pan > 0 BLUE, pan < -30 GREEN). Rest = distance to the data home pose (median of the clean starts).
"""
import glob
import json
import sys

import numpy as np
import pyarrow.parquet as pq

LIVE_HOME = np.array([-6.6, -98.1, 99.4, 54.3, -53.1])  # live probe 14:13Z (shared/FACTS-LIVE.md)


def load(ds):
    rows = {}
    for path in sorted(glob.glob(ds + "/data/chunk-*/file-*.parquet")):
        cols = pq.read_table(path, columns=["episode_index", "frame_index", "observation.state"]).to_pydict()
        for ep, fi, st in zip(cols["episode_index"], cols["frame_index"], cols["observation.state"]):
            rows.setdefault(int(ep), []).append((int(fi), st))
    out = {}
    for ep, r in rows.items():
        r.sort(key=lambda x: x[0])
        out[ep] = np.array([x[1] for x in r], dtype=float)
    return out


def analyse(S, home):
    g = S[:, 5]
    arm = S[:, :5]
    n = len(S)
    res = {
        "frames": n,
        "start_d": round(float(np.abs(arm[0] - home).max()), 2),
        "end_d": round(float(np.abs(arm[-1] - home).max()), 2),
        "closures": 0,
        "grasp": None,
        "release": None,
        "hold_median": None,
        "hold_s": 0.0,
        "carry_units": None,
        "pan_release": None,
        "dest": None,
        "closed_on_brick": False,
        "grip_min": round(float(g.min()), 2),
    }
    state = None
    events = []
    for i in range(n):
        if g[i] > 8:
            state = "open"
        elif g[i] < 5 and state == "open":
            events.append(i)
            state = "closed"
    res["closures"] = len(events)
    if not events:
        return res
    gi = events[0]
    rel = next((j for j in range(gi + 1, n) if g[j] > 8), None)
    res["grasp"] = gi
    res["release"] = rel
    hold_end = rel if rel is not None else n
    hold = g[gi + 10:hold_end] if hold_end - gi > 12 else g[gi:hold_end]
    if len(hold):
        res["hold_median"] = round(float(np.median(hold)), 2)
    res["hold_s"] = round((hold_end - gi) / 30.0, 2)
    res["carry_units"] = round(float(np.abs(arm[hold_end - 1] - arm[gi]).max()), 2)
    res["closed_on_brick"] = bool(
        res["hold_median"] is not None and 1.9 <= res["hold_median"] <= 2.9 and res["hold_s"] >= 1.0
    )
    if rel is not None:
        pan = float(arm[rel, 0])
        res["pan_release"] = round(pan, 2)
        res["dest"] = "BLUE" if pan > 0 else ("GREEN" if pan < -30 else "AMBIG")
    return res


def main():
    ds, out_json = sys.argv[1], sys.argv[2]
    states = load(ds)
    clean = [ep for ep, S in states.items() if np.abs(S[0, :5] - LIVE_HOME).max() < 10]
    home = np.median(np.stack([states[ep][0, :5] for ep in clean]), axis=0)
    results = {}
    for ep in sorted(states):
        results[ep] = analyse(states[ep], home)
        r = results[ep]
        print(
            f"ep{ep:02d} closures={r['closures']} grasp={r['grasp']} release={r['release']} hold={r['hold_median']}"
            f" hold_s={r['hold_s']} on_brick={r['closed_on_brick']} carry={r['carry_units']} pan_rel={r['pan_release']}"
            f" dest={r['dest']} start_d={r['start_d']} end_d={r['end_d']} grip_min={r['grip_min']}"
        )
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump({"home": np.round(home, 2).tolist(), "episodes": results}, fh, indent=1)
    print("wrote", out_json)


if __name__ == "__main__":
    main()
