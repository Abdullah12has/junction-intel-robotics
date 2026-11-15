import json, os, sys
import pyarrow.parquet as pq

R = "/home/intel/intel-junction-2026"

def info(ds):
    d = json.load(open(os.path.join(R, ds, "meta", "info.json")))
    keys = {k: d[k] for k in d if k != "features"}
    print("INFO", ds, json.dumps(keys)[:900])
    print("  features:", sorted(d.get("features", {}).keys()))

def pqs(ds):
    for root, _, files in os.walk(os.path.join(R, ds)):
        for f in sorted(files):
            if f.endswith(".parquet"):
                p = os.path.join(root, f)
                pf = pq.ParquetFile(p)
                print("PQ", os.path.relpath(p, R), "rows", pf.metadata.num_rows, "cols", pf.schema_arrow.names[:40])

def tasks(ds):
    t = pq.read_table(os.path.join(R, ds, "meta", "tasks.parquet")).to_pylist()
    print("TASKS", ds, len(t), json.dumps(t[:3])[:300])

for ds in sys.argv[1:]:
    print("=== DATASET", ds)
    info(ds)
    pqs(ds)
    tasks(ds)
