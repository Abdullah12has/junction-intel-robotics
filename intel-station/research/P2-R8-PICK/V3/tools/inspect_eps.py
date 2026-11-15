import glob
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = "/home/intel/intel-junction-2026/data/studio-datasets"


def load(ds, pattern):
    files = sorted(glob.glob(os.path.join(ROOT, ds, pattern)))
    return pa.concat_tables([pq.read_table(f) for f in files], promote_options="default")


def main():
    ds = sys.argv[1]
    eps = load(ds, "meta/episodes/chunk-000/*.parquet").to_pylist()
    data = load(ds, "data/chunk-000/*.parquet").to_pylist()
    print("EP_COLS", sorted(eps[0].keys()))
    print("ROWS", len(data), "EPISODES", len(eps))
    by_ep = {}
    for r in data:
        by_ep.setdefault(r["episode_index"], []).append(r)
    for e in eps:
        ei = e["episode_index"]
        rows = sorted(by_ep[ei], key=lambda r: r["frame_index"])
        S = np.array([r["observation.state"] for r in rows], dtype=float)
        g = S[:, 5]
        n = len(rows)
        grasp = None
        for t in range(n - 15):
            if g[t] < 4.0 and np.all(g[t:t + 15] < 4.0):
                grasp = t
                break
        rel = None
        if grasp is not None:
            for t in range(grasp, n):
                if g[t] > 8.0:
                    rel = t
                    break
        task = e.get("tasks")
        task = task[0] if isinstance(task, list) and task else task
        tops = {k: e[k] for k in e if "243622061421" in k and ("from" in k or "file" in k or "chunk" in k)}
        print("EP", ei, "n", n, "task", task, "grip_min", round(float(g.min()), 1))
        print("  start", np.round(S[0], 1).tolist())
        if grasp is not None:
            print("  grasp_t", grasp, "grip", round(float(g[grasp]), 2), "state", np.round(S[grasp], 2).tolist())
        else:
            print("  grasp_t None")
        print("  release_t", rel, "vid", tops)


if __name__ == "__main__":
    main()
