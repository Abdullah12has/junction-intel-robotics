import json
import sys

d = json.load(open(sys.argv[1]))
for r in d["samples"]:
    if "error" in r:
        print(r["sample"], "ERROR", r["error"])
        continue
    print(r["sample"], "raw_refused", r["raw"]["refused"], "jpg_refused", r["jpg"]["refused"])
    for s in r["skin_top"]:
        if s["area"] >= 900 or s["in_mat"] > 0:
            print("   skin", s)
    for c in r["diffs"]["changes"]:
        print("   change", c)
    for h in r.get("close_hue", []):
        if h.get("merged") or h.get("shape") == "unknown" or h.get("share2", 0) > 0.1:
            print("   hue", h)
