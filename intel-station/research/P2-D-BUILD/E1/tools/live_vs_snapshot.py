import glob

import numpy as np
import pyarrow.parquet as pq

L = "/home/intel/intel-junction-2026"
W = L + "/work/P2-D-BUILD/E1"
DS = "6d6bdce3-b382-478d-9697-fb9e573b97af"
EPS = (3, 4, 5, 6)


def load(root):
    out = {}
    for path in sorted(glob.glob(root + "/data/chunk-*/file-*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "action", "observation.state"]).to_pydict()
        for e, a, s in zip(table["episode_index"], table["action"], table["observation.state"]):
            if e in EPS:
                out.setdefault(e, ([], []))
                out[e][0].append(np.asarray(a, dtype=np.float32))
                out[e][1].append(np.asarray(s, dtype=np.float32))
    return {e: (np.stack(v[0]), np.stack(v[1])) for e, v in out.items()}


snap = load(W + "/src/" + DS)
live = load(L + "/data/studio-datasets/" + DS)
for e in EPS:
    sa, ss = snap[e]
    la, ls = live[e]
    print(
        "episode", e,
        "snapshot_rows", len(sa), "live_rows", len(la),
        "action_equal", bool(sa.shape == la.shape and np.array_equal(sa, la)),
        "state_equal", bool(ss.shape == ls.shape and np.array_equal(ss, ls)),
    )
