import sys, glob, shutil, importlib
print("py", sys.version.split()[0], sys.executable)
for m in ["numpy", "pyarrow", "pandas", "cv2", "av"]:
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "MISSING", type(e).__name__)
print("ffmpeg", shutil.which("ffmpeg"))
import pyarrow.parquet as pq
R = "/home/intel/intel-junction-2026/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af"
for f in sorted(glob.glob(R + "/meta/episodes/**/*.parquet", recursive=True)):
    t = pq.read_table(f)
    print("EPMETA", f.split("/")[-1], t.num_rows)
    print([(n, str(t.schema.field(n).type)) for n in t.schema.names])
    d = t.to_pydict()
    print({k: d[k][:3] for k in list(d)[:60]})
for f in sorted(glob.glob(R + "/data/chunk-000/*.parquet")):
    t = pq.read_table(f)
    print("DATA", f.split("/")[-1], t.num_rows, t.schema.names)
print("TASKS", pq.read_table(R + "/meta/tasks.parquet").to_pydict())
