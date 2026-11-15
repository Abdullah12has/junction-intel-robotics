"""P4-FUSION E1 read-only summary of the P3-LABEL-STUDIO E1 label files (prints counts and one sample record)."""
import collections
import json
import os

R = "/home/intel/intel-junction-2026"
P = os.path.join(R, "work", "P3-LABEL-STUDIO", "E1")
recs = []
with open(os.path.join(P, "labels_episodes.jsonl"), encoding="utf-8") as fh:
    for line in fh:
        if line.strip():
            recs.append(json.loads(line))
print("n_records", len(recs))
keys = collections.Counter(k for r in recs for k in r.keys())
print("KEYS", dict(keys))
print("FIRST", json.dumps(recs[0], ensure_ascii=False)[:1800])
for k in sorted(keys):
    vals = collections.Counter(json.dumps(r.get(k), ensure_ascii=False)[:70] for r in recs)
    if len(vals) <= 20:
        print("COUNT", k, dict(vals))
    else:
        print("NVALS", k, len(vals))
with open(os.path.join(P, "datasets.json"), encoding="utf-8") as fh:
    ds = json.load(fh)
if isinstance(ds, dict):
    print("DATASETS dict keys", list(ds.keys())[:30])
    first = ds[list(ds.keys())[0]]
else:
    print("DATASETS list len", len(ds))
    first = ds[0]
print("DATASET_FIRST", json.dumps(first, ensure_ascii=False)[:1200])
dirs = collections.Counter()
for root, _dirs, files in os.walk(os.path.join(P, "frames")):
    if files:
        dirs[os.path.relpath(root, P)] = len(files)
print("FRAME_DIRS", len(dirs), list(dirs.items())[:25])
