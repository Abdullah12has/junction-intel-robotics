#!/usr/bin/env python3
# REPO-EXPORT E1 read-only dump (round 3). Writes dump.txt next to this script.
# Reads files only. Imports nothing from R/app. Writes nothing outside this folder.
import collections
import csv
import os
import re

R = "/home/intel/intel-junction-2026"
APP = os.path.join(R, "app")
HERE = os.path.dirname(os.path.abspath(__file__))
SKIP = ("/venv", "/.venv", "uvcache", "site-packages", "/pkg/", "node_modules", "__pycache__", "/.git", "/.cache")
out = []


def o(s=""):
    out.append(str(s))


def read(p):
    with open(p, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def block(title, p, limit=3500):
    size = os.path.getsize(p) if os.path.exists(p) else -1
    o("=== " + title + " :: " + os.path.relpath(p, R) + " bytes=" + str(size))
    if size >= 0:
        t = read(p)
        o(t[:limit])
        if len(t) > limit:
            o("...[truncated at " + str(limit) + " of " + str(len(t)) + " chars]")


# 1. last lines of probe.txt (not shown in the previous read)
o("=== 1 probe.txt lines 400 onward ===")
pl = read(os.path.join(HERE, "probe.txt")).split("\n")
o("\n".join(pl[399:]))

# 2. tape detector LOO metrics and evaluation set
block("2 tape metrics", os.path.join(R, "work", "P3-TAPE-EVAL", "E2", "metrics.json"), 5000)
block("2b tape eval set head", os.path.join(R, "work", "P3-TAPE-EVAL", "E2", "eval_set.tsv"), 900)
block("2c tape results head", os.path.join(R, "work", "P3-TAPE-EVAL", "E2", "results.tsv"), 1200)

# 3. OpenVINO benchmark outputs and the script that produced them
block("3 OVBENCH bench table", os.path.join(R, "work", "P4-OVBENCH", "E1", "bench.tsv"), 4200)
block("3b OVBENCH metrics", os.path.join(R, "work", "P4-OVBENCH", "E1", "metrics.json"), 4300)
block("3c OVBENCH run script", os.path.join(R, "work", "P4-OVBENCH", "E1", "tools", "run_ovbench.sh"), 1800)
for f in ("bench_ov_v2.py", "bench_ov.py", "aggregate_v2.py"):
    p = os.path.join(R, "work", "P4-OVBENCH", "E1", "tools", f)
    if os.path.exists(p):
        o("== head P4-OVBENCH tools/" + f + " lines=" + str(len(read(p).split("\n"))))
        for l in read(p).split("\n")[:10]:
            o("   " + l[:180])

# 4. hand detector API
hp = os.path.join(APP, "tools", "handdet.py")
o("=== 4 handdet.py definitions and head ===")
for i, l in enumerate(read(hp).split("\n")):
    if re.match(r"\s*(def |class |[A-Z_][A-Z0-9_]* *=)", l):
        o("L" + str(i + 1) + ": " + l[:170])
for l in read(hp).split("\n")[:10]:
    o("   " + l[:170])

# 5. hand label counts
o("=== 5 hand_labels.csv counts by view and label ===")
rows = list(csv.DictReader(open(os.path.join(APP, "tools", "tests", "fixtures", "hand_labels.csv"), newline="", encoding="utf-8")))
cnt = collections.Counter((r["view"], r["label"]) for r in rows)
for k, v in sorted(cnt.items()):
    o(str(k) + " " + str(v))
o("rows " + str(len(rows)) + " holding=1 " + str(sum(1 for r in rows if r["holding"] == "1")))

# 6. live hand watchdog log
hw = os.path.join(R, "logs", "monitor", "hand_watch.log")
o("=== 6 hand_watch.log ===")
if os.path.exists(hw):
    ls = read(hw).split("\n")
    o("lines " + str(len(ls)) + " bytes " + str(os.path.getsize(hw)))
    for l in ls[:3]:
        o("  " + l[:200])
    for l in ls[-3:]:
        o("  " + l[:200])
    o("lines with stop or STOP: " + str(sum(1 for l in ls if re.search(r"stop|STOP", l))))
else:
    o("MISSING")

# 7. dataset label quality metrics
block("7 label metrics", os.path.join(R, "work", "P3-LABEL-STUDIO", "E2", "metrics.json"), 2500)

# 8. locations of lint and helper tools named in the docs
o("=== 8 lint and helper files by name ===")
names = {"attrs.py", "undef.py", "smoke_offline.py", "extract_frames.py", "so101.py", "steps.tsv", "scan_pairs.json", "cycles.json"}
hits = []
for root in (os.path.join(R, "work"), APP):
    for dp, dn, fn in os.walk(root):
        if any(sk in dp for sk in SKIP):
            dn[:] = []
            continue
        for f in fn:
            if f in names:
                hits.append(os.path.relpath(os.path.join(dp, f), R))
o("\n".join(sorted(hits)[:60]))

# 9. telemetry scripts (first lines) and sampler output
o("=== 9 telemetry scripts (first 5 code lines each) ===")
for f in ("bench_ov.py", "ov_devices.py", "ir_inspect.py", "final_pack.py", "bench_summary2.py", "bench_compact.py"):
    p = os.path.join(APP, "telemetry", f)
    if os.path.exists(p):
        o("== telemetry/" + f + " lines=" + str(len(read(p).split("\n"))))
        for l in read(p).split("\n")[:5]:
            o("   " + l[:170])
block("9b run_final.sh", os.path.join(APP, "telemetry", "run_final.sh"), 1200)
tp = os.path.join(R, "logs", "telemetry", "20261114.jsonl")
if os.path.exists(tp):
    ls = read(tp).split("\n")
    o("=== 9c telemetry jsonl lines=" + str(len(ls)) + " first:")
    o(ls[0][:400])

with open(os.path.join(HERE, "dump.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(out) + "\n")
print("WROTE", os.path.join(HERE, "dump.txt"), len(out), "lines")
