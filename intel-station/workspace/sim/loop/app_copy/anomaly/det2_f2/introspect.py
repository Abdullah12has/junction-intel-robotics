import glob, json, os, sys
import pyarrow.parquet as pq

root = sys.argv[1]
info = json.load(open(os.path.join(root, "meta/info.json")))
print({k: v for k, v in info.items() if k != "features"})
print("feature keys:", list(info["features"].keys()))
for pat in ["meta/episodes/**/*.parquet", "meta/tasks.parquet", "data/**/*.parquet"]:
    files = sorted(glob.glob(os.path.join(root, pat), recursive=True))
    print("PATTERN", pat, "files:", [os.path.relpath(f, root) for f in files][:8], "count", len(files))
    for f in files[:1]:
        t = pq.read_table(f)
        print("  rows", t.num_rows, "cols", t.schema.names)
        row = t.slice(0, 1).to_pylist()[0]
        print("  row0", {k: (v if not isinstance(v, list) or len(v) < 12 else str(v)[:120]) for k, v in row.items()})
vids = sorted(glob.glob(os.path.join(root, "videos/**/*.mp4"), recursive=True))
print("VIDEOS", len(vids))
for v in vids[:6]:
    print("  ", os.path.relpath(v, root), os.path.getsize(v))
