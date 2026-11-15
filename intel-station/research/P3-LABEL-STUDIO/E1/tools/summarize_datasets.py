import json

BASE = "/home/intel/intel-junction-2026/work/P3-LABEL-STUDIO/E1/"
d = json.load(open(BASE + "datasets.json", encoding="utf-8"))
print("top-level type", type(d).__name__, "keys" if isinstance(d, dict) else "len", list(d.keys()) if isinstance(d, dict) else len(d))
items = d if isinstance(d, list) else d.get("datasets", d)
if isinstance(items, dict):
    for k, v in list(items.items())[:3]:
        print("KEY", k, "->", json.dumps(v)[:900])
else:
    for v in items[:2]:
        print("ITEM", json.dumps(v)[:1200])
for k in (d.keys() if isinstance(d, dict) else []):
    if not isinstance(d[k], (list, dict)):
        print("SCALAR", k, d[k])
    else:
        print("CONTAINER", k, type(d[k]).__name__, len(d[k]))
