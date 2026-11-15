#!/usr/bin/env python3
"""Pairwise alignment of the 41 dataset episodes' action trajectories to find exact, prefix and shifted duplicates.

For every pair of episodes (A, B) it searches the shift s where A[t] is compared with B[t - s] over the overlap
and keeps the lowest mean absolute error. Writes out/overlap_pairs.json (all pairs with overlap >= 60 frames).
"""
import glob
import itertools
import json
import os

import numpy as np
import pyarrow.parquet as pq

R = "/home/intel/intel-junction-2026"
SD = os.path.join(R, "data", "studio-datasets")
OUTD = os.path.join(R, "work", "P3-LABEL-STUDIO", "E1", "out")
MAXSHIFT = 1400
MINOV = 60


def load_actions(eps):
    need = {}
    for e in eps:
        need.setdefault(e["dataset"], set()).add(e["episode_index"])
    traj = {}
    for ds, epset in need.items():
        d = os.path.join(SD, ds)
        by_ep = {}
        for f in sorted(glob.glob(os.path.join(d, "data", "chunk-*", "*.parquet"))):
            t = pq.read_table(f, columns=["episode_index", "frame_index", "action"])
            ep_col = t.column("episode_index").to_pylist()
            fi_col = t.column("frame_index").to_pylist()
            act = t.column("action").to_pylist()
            for i in range(len(ep_col)):
                by_ep.setdefault(ep_col[i], []).append((fi_col[i], act[i]))
        for ep in epset:
            rows = sorted(by_ep.get(ep, []), key=lambda r: r[0])
            traj[(ds, ep)] = np.array([r[1] for r in rows], dtype=np.float64)
    return traj


def best_align(A, B):
    na, nb = len(A), len(B)
    best = None
    for s in range(-min(nb - 1, MAXSHIFT), min(na - 1, MAXSHIFT) + 1):
        a0 = max(0, s)
        a1 = min(na, nb + s)
        ov = a1 - a0
        if ov < MINOV:
            continue
        mae = float(np.abs(A[a0:a1] - B[a0 - s:a1 - s]).mean())
        if best is None or mae < best[1]:
            best = (s, mae, ov)
    return best


def main():
    eps = [json.loads(line) for line in open(os.path.join(OUTD, "episodes_census.jsonl"))]
    traj = load_actions(eps)
    keys = sorted(traj.keys())
    pairs = []
    for ka, kb in itertools.combinations(keys, 2):
        A, B = traj[ka], traj[kb]
        if len(A) == 0 or len(B) == 0:
            continue
        res = best_align(A, B)
        if res is None:
            continue
        s, mae, ov = res
        pairs.append({"a": "%s:%d" % (ka[0][:8], ka[1]), "b": "%s:%d" % (kb[0][:8], kb[1]),
                      "na": len(A), "nb": len(B), "shift_frames_A_minus_B": s,
                      "overlap": ov, "overlap_frac_of_shorter": round(ov / min(len(A), len(B)), 3),
                      "mae_action": round(mae, 4)})
    pairs = [p for p in pairs if p["mae_action"] < 2.0]
    with open(os.path.join(OUTD, "overlap_pairs.json"), "w") as fh:
        json.dump(pairs, fh, indent=1)
    print("entries=%d pairs_with_mae<2=%d" % (len(keys), len(pairs)))
    for p in sorted(pairs, key=lambda p: (p["mae_action"], -p["overlap"])):
        print("%s ~ %s na=%d nb=%d shift(A-B)=%d ov=%d frac=%.3f mae=%.4f" % (
            p["a"], p["b"], p["na"], p["nb"], p["shift_frames_A_minus_B"], p["overlap"],
            p["overlap_frac_of_shorter"], p["mae_action"]))


if __name__ == "__main__":
    main()
