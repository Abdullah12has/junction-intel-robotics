import glob
import os

import pyarrow.parquet as pq

ROOT = "/home/intel/intel-junction-2026/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af"

t = pq.read_table(os.path.join(ROOT, "meta/tasks.parquet"))
print("TASKS cols", t.schema.names, "rows", t.num_rows)
for r in t.to_pylist():
    print("  ", r)

eps = []
for f in sorted(glob.glob(os.path.join(ROOT, "meta/episodes/chunk-*/file-*.parquet"))):
    tb = pq.read_table(f)
    print("EPFILE", os.path.basename(f), "rows", tb.num_rows)
    if not eps:
        print("  cols", tb.schema.names)
    eps += tb.to_pylist()
print("EP_COUNT", len(eps))
keep = [k for k in eps[0].keys() if not k.startswith("stats/")]
for r in eps:
    print("EP", {k: r[k] for k in keep})

for f in sorted(glob.glob(os.path.join(ROOT, "data/chunk-*/file-*.parquet"))):
    tb = pq.read_table(f)
    print("DATAFILE", os.path.basename(f), "rows", tb.num_rows, "cols", tb.schema.names)
    row0 = tb.slice(0, 1).to_pylist()[0]
    print("  row0", {k: (str(v)[:120]) for k, v in row0.items()})
