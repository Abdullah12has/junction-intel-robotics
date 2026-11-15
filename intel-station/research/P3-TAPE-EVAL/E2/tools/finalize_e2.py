#!/usr/bin/env python3
"""Write the spec deliverables eval_set.tsv, results.tsv, confusion.json from the base run; stage name tape-eval-v1."""
import sys, os, csv, json
E2 = sys.argv[1]
views = list(csv.DictReader(open(os.path.join(E2, "base", "views.tsv")), delimiter="\t"))
studio = list(csv.DictReader(open(os.path.join(E2, "base", "studio.tsv")), delimiter="\t"))
ev_cols = ["set", "file", "episode", "kind", "colour", "truth"]
with open(os.path.join(E2, "eval_set.tsv"), "w") as f:
    f.write("\t".join(ev_cols) + "\n")
    for r in views:
        f.write("\t".join([r["set"], r["file"], r["episode"], r["kind"], r["colour"], r["truth"]]) + "\n")
    for r in studio:
        if r["verdict"] == "NOT_LOCATED":
            continue
        f.write("\t".join(["studio", r["file"], r["entry"], r["stage"], r["colour"], r["truth"]]) + "\n")
res_cols = ["set", "file", "kind_or_stage", "colour", "truth", "verdict", "pred", "tape_score", "reasons"]
with open(os.path.join(E2, "results.tsv"), "w") as f:
    f.write("\t".join(res_cols) + "\n")
    for r in views:
        f.write("\t".join([r["set"], r["file"], r["kind"], r["colour"], r["truth"], r["verdict"], r["pred"], r["tape_score"], r["reasons"]]) + "\n")
    for r in studio:
        f.write("\t".join(["studio", r["file"], r["stage"], r["colour"], r["truth"], r["verdict"], r["pred"], r.get("tape_score", ""), r["reasons"]]) + "\n")
m = json.load(open(os.path.join(E2, "metrics.json")))
conf = {"stage": "tape-eval-v1", "note": "base run (R/app as is); see metrics.json for variants",
        "ours_close": json.load(open(os.path.join(E2, "base", "metrics.json")))["tape_ours_close"],
        "studio_close_per_brick": json.load(open(os.path.join(E2, "base", "metrics.json")))["tape_studio_close_per_brick"]}
json.dump(conf, open(os.path.join(E2, "confusion.json"), "w"), indent=1, default=str)
m["stage"] = "tape-eval-v1"
json.dump(m, open(os.path.join(E2, "metrics.json"), "w"), indent=1, default=str)
print("written", len(views), len(studio))
