import json, os, sys
import pyarrow.parquet as pq
ROOT = sys.argv[1]
info = json.load(open(os.path.join(ROOT, "meta", "info.json")))
print("info keys:", list(info.keys()))
print("total_episodes", info.get("total_episodes"), "total_frames", info.get("total_frames"), "fps", info.get("fps"))
for k, v in info.get("features", {}).items():
    print("feature", k, v.get("dtype"), v.get("shape"), (v.get("names") if k in ("observation.state", "action") else ""))
for dp, dn, fn in os.walk(ROOT):
    for f in sorted(fn):
        p = os.path.join(dp, f)
        print("file", os.path.relpath(p, ROOT), os.path.getsize(p))
for sub in ("meta/episodes", "data", "meta"):
    for dp, dn, fn in os.walk(os.path.join(ROOT, sub)):
        for f in sorted(fn):
            if f.endswith(".parquet"):
                p = os.path.join(dp, f)
                t = pq.read_table(p)
                print("PARQUET", os.path.relpath(p, ROOT), "rows", t.num_rows)
                print(t.schema.to_string(show_schema_metadata=False)[:3000])
                r = t.slice(0, 1).to_pylist()[0]
                print("ROW0", {k: str(v)[:160] for k, v in r.items()})
