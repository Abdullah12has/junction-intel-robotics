"""Per-frame comparison of E2 baseline, earlier V1 pass and current V1 run (analysis only). usage: python3 -I -B cmp_v1.py EVAL_DIR"""
import json
import os
import sys
from collections import Counter

EV = sys.argv[1]
F = {k: json.load(open(os.path.join(EV, f))) for k, f in
     (("e2", "baseline_e2.json"), ("p1", "pass1_v1.json"), ("now", sys.argv[2] if len(sys.argv) > 2 else "after_v1.json"))}
by = {k: {s["sample"]: s for s in v["samples"]} for k, v in F.items()}


def tally(rec):
    if rec is None or rec.get("refused"):
        return "REFUSED" if rec else "-"
    bs = rec.get("bricks", [])
    t = Counter((b.get("view", "?")[0], b.get("verdict", "?")[:4]) for b in bs)
    return " ".join("%s%s%d" % (vw, vd, n) for (vw, vd), n in sorted(t.items())) or "none"


names = sorted(by["now"].keys())
print("sample | variant | E2 | pass1 | now")
for n in names:
    for var in ("jpg", "raw"):
        cells = [tally((by[k].get(n) or {}).get(var)) for k in ("e2", "p1", "now")]
        if len(set(cells)) > 1 or var == "jpg":
            print(n, var, "|", " | ".join(cells))
print("--- now-vs-p1 brick detail, frames whose brick list changed (jpg)")
for n in names:
    a = by["p1"].get(n, {}).get("jpg") or {}
    b = by["now"].get(n, {}).get("jpg") or {}
    ka = sorted((x.get("view"), x.get("colour"), x.get("area")) for x in a.get("bricks", []))
    kb = sorted((x.get("view"), x.get("colour"), x.get("area")) for x in b.get("bricks", []))
    if ka != kb:
        print(n)
        for x in b.get("bricks", []):
            print("   now", x.get("view"), x.get("colour"), x.get("area"), x.get("shape"), "split" if x.get("split") else "",
                  "merged" if x.get("merged") else "", x.get("verdict"), x.get("tape_score"))
        for x in a.get("bricks", []):
            print("   p1 ", x.get("view"), x.get("colour"), x.get("area"), x.get("shape"), "split" if x.get("split") else "",
                  "merged" if x.get("merged") else "", x.get("verdict"), x.get("tape_score"))
