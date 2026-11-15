import glob
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BASE = "/home/intel/intel-junction-2026/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af"
OUT = "/home/intel/intel-junction-2026/work/P2-R8-PICK/V3/out"


def load(pattern):
    files = sorted(glob.glob(os.path.join(BASE, pattern)))
    return pa.concat_tables([pq.read_table(f) for f in files], promote_options="default")


def main():
    eps = {e["episode_index"]: e for e in load("meta/episodes/chunk-000/*.parquet").to_pylist()}
    data = load("data/chunk-000/*.parquet").to_pylist()
    by_ep = {}
    for r in data:
        by_ep.setdefault(r["episode_index"], []).append(r)
    out = {}
    for ei in sorted(eps):
        rows = sorted(by_ep[ei], key=lambda r: r["frame_index"])
        S = np.array([r["observation.state"] for r in rows], dtype=float)
        g = S[:, 5]
        grasp = None
        for t in range(len(g) - 15):
            if g[t] < 4.0 and np.all(g[t:t + 15] < 4.0):
                grasp = t
                break
        rel = None
        if grasp is not None:
            for t in range(grasp, len(g)):
                if g[t] > 8.0:
                    rel = t
                    break
        task = eps[ei]["tasks"][0] if eps[ei].get("tasks") else None
        # state at release: take the frame where the gripper first exceeds 8 after the grasp
        rec = {"task": task, "grasp_t": grasp, "release_t": rel,
               "grasp_state": None if grasp is None else np.round(S[grasp], 3).tolist(),
               "release_state": None if rel is None else np.round(S[rel], 3).tolist(),
               "end_state": np.round(S[-1], 3).tolist()}
        out[str(ei)] = rec
        print("EP", ei, grasp, rel, rec["release_state"], flush=True)
    with open(os.path.join(OUT, "release_states.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("WROTE", flush=True)


if __name__ == "__main__":
    main()
