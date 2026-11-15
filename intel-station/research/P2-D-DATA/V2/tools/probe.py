import glob
import pyarrow.parquet as pq

D = "/home/intel/intel-junction-2026/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af"
for f in sorted(glob.glob(D + "/data/chunk-*/file-*.parquet")):
    t = pq.read_table(f)
    print("DATA", f.split("/data/")[1], t.num_rows, t.schema.names)
for f in sorted(glob.glob(D + "/meta/episodes/chunk-*/file-*.parquet")):
    t = pq.read_table(f)
    print("EPMETA", f.split("/episodes/")[1], t.num_rows)
    names = [n for n in t.schema.names if not n.startswith("stats/")]
    print("EPCOLS", names)
    row = t.slice(0, 1).to_pylist()[0]
    print("EPROW0", {k: (str(row[k])[:90]) for k in names})
t = pq.read_table(D + "/meta/tasks.parquet")
print("TASKS", t.num_rows, t.schema.names)
print("TASKROWS", t.to_pylist())
