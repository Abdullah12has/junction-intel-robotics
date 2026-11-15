#!/usr/bin/env python3
"""Rest-pose check for LeRobot v3 episodes (read-only).

Home = median start pose of the clean starts (episodes that start within 10 units of the live home pose).
Usage: python3 -I tails.py <dataset_dir>
"""
import glob
import sys

import numpy as np
import pyarrow.parquet as pq

LIVE_HOME = np.array([-6.6, -98.1, 99.4, 54.3, -53.1])  # live probe 14:13Z (shared/FACTS-LIVE.md)


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


def main():
    states = load_states(sys.argv[1])
    clean = [ep for ep, S in states.items() if np.abs(S[0, :5] - LIVE_HOME).max() < 10]
    home = np.median(np.stack([states[ep][0, :5] for ep in clean]), axis=0)
    print("clean starts", sorted(clean), "data home", np.round(home, 2).tolist())
    for ep in sorted(states):
        S = states[ep]
        arm = S[:, :5]
        d = np.abs(arm - home).max(axis=1)
        steps = [round(float(d[-1 - 15 * k]), 1) for k in range(8)]
        vel = np.abs(np.diff(arm, axis=0)).max(axis=1)
        print(
            f"ep{ep:02d} start={d[0]:.2f} end={d[-1]:.2f} dist_last_3.5s_newest_first={steps} "
            f"maxvel_last_1s={vel[-30:].max():.3f} grip_end={S[-1, 5]:.2f} grip_min={S[:, 5].min():.2f}"
        )


if __name__ == "__main__":
    main()
