import glob, json, os
import numpy as np
import pyarrow.parquet as pq

R = "/home/intel/intel-junction-2026"
SD = os.path.join(R, "data", "studio-datasets")
BASE = os.path.join(R, "work/P3-LABEL-STUDIO/E1/")
census = json.load(open(BASE + "out/datasets_census.json"))
by_pref = {d["uuid"][:8]: d for d in census}

print("CHECK 1: info.json vs meta/data row counts (independent re-read of the mirror)")
for d in census:
    base = os.path.join(SD, d["uuid"])
    info = json.load(open(os.path.join(base, "meta", "info.json")))
    meta_rows = sum(pq.read_metadata(f).num_rows
                    for f in glob.glob(os.path.join(base, "meta", "episodes", "chunk-*", "*.parquet")))
    data_rows = sum(pq.read_metadata(f).num_rows
                    for f in glob.glob(os.path.join(base, "data", "chunk-*", "*.parquet")))
    cams = sorted(k.split()[-1] for k in info.get("features", {}) if k.startswith("observation.images"))
    print(d["uuid"][:8], "fps", info.get("fps"), "info_total_episodes", info.get("total_episodes"),
          "info_total_frames", info.get("total_frames"), "meta_rows", meta_rows, "data_rows", data_rows,
          "census_data_rows", d["data_rows"], "cams", cams,
          "match", info.get("total_episodes") == meta_rows and info.get("total_frames") == data_rows)


def actions(uuid, ep):
    base = os.path.join(SD, uuid)
    rows = []
    for f in sorted(glob.glob(os.path.join(base, "data", "chunk-*", "*.parquet"))):
        t = pq.read_table(f, columns=["episode_index", "frame_index", "action"]).to_pydict()
        for e_, fi, a in zip(t["episode_index"], t["frame_index"], t["action"]):
            if e_ == ep:
                rows.append((fi, a))
    rows.sort(key=lambda x: x[0])
    return np.array([a for _, a in rows], dtype=np.float64)


print("CHECK 2: action rows of copies vs representative (raw parquet, not the census)")
PAIRS = [("18468a09", 0, "6d6bdce3", 3, 113), ("18468a09", 6, "6d6bdce3", 12, 0),
         ("a4ba6ede", 0, "6d6bdce3", 3, 0), ("a4ba6ede", 3, "6d6bdce3", 6, 0),
         ("51100357", 0, "6d6bdce3", 15, 0), ("800d7dc2", 1, "6d6bdce3", 12, 0),
         ("e2eae6c8", 2, "6d6bdce3", 12, 0), ("d2fd0a80", 0, "6d6bdce3", 11, 57),
         ("ba9eafba", 1, "6d6bdce3", 15, 0)]
for cp, ce, rp, re_, off in PAIRS:
    a = actions(by_pref[cp]["uuid"], ce)
    b = actions(by_pref[rp]["uuid"], re_)
    seg = b[off:off + len(a)]
    same_len = len(seg) == len(a)
    mad = float(np.abs(a - seg).max()) if same_len and len(a) else None
    print("%s:%d vs %s:%d offset %d  n_copy %d n_rep_window %d max_abs_diff %s" %
          (cp, ce, rp, re_, off, len(a), len(seg), mad))

print("CHECK 3: frozen joints in 7e99a9e4")
a = actions(by_pref["7e99a9e4"]["uuid"], 0)
print("n", len(a), "std per joint", np.round(a.std(axis=0), 4).tolist(),
      "gripper unique values", np.unique(a[:, 5]).tolist()[:5])
