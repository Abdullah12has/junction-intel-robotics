#!/usr/bin/env python3
# REPO-EXPORT E1 read-only probe (round 2). Writes probe.txt next to this script.
# Reads files only. Imports nothing from R/app. Writes nothing outside this folder.
import hashlib
import json
import os
import re
import subprocess
import time

R = "/home/intel/intel-junction-2026"
APP = os.path.join(R, "app")
HERE = os.path.dirname(os.path.abspath(__file__))
lines = []
SKIP = ("/venv", "/.venv", "uvcache", "site-packages", "/pkg/", "node_modules", "__pycache__", "/.git", "/.cache")


def o(s=""):
    lines.append(str(s))


def read(p):
    with open(p, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()[:12]


def skipped(dp):
    return any(sk in dp for sk in SKIP)


# 1. cycles.jsonl schema per run
o("=== 1 cycles.jsonl: records, union of keys with types, first record ===")
for run in ["any_1731", "any_1740", "any_2055", "any_2127"]:
    p = os.path.join(R, "logs", "sort", run, "cycles.jsonl")
    if not os.path.exists(p):
        o(run + " MISSING")
        continue
    recs, bad = [], 0
    with open(p, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                recs.append(json.loads(raw))
            except ValueError:
                bad += 1
    keys = {}
    for r in recs:
        if isinstance(r, dict):
            for k, v in r.items():
                keys.setdefault(k, type(v).__name__)
    o(run + ": records=" + str(len(recs)) + " bad=" + str(bad) + " mtime=" + time.strftime("%H:%MZ", time.gmtime(os.path.getmtime(p))))
    o("  keys: " + ", ".join(k + ":" + t for k, t in sorted(keys.items())))
    if recs:
        o("  first: " + json.dumps(recs[0], sort_keys=True)[:900])

# 2. OpenVINO benchmark folder P4-OVBENCH
base = os.path.join(R, "work", "P4-OVBENCH")
o("\n=== 2 P4-OVBENCH files (depth<=3) ===")
names = []
for dp, dn, fn in os.walk(base):
    if skipped(dp):
        dn[:] = []
        continue
    if dp[len(base):].count(os.sep) >= 3:
        dn[:] = []
    for f in fn:
        names.append(os.path.relpath(os.path.join(dp, f), base))
o("count " + str(len(names)))
o(", ".join(sorted(names))[:3000])

o("\n=== 2b P4-OVBENCH result heads and timing lines ===")
for f in ["s1_result.txt", "s2_result.txt", "s3_result.txt"]:
    p = os.path.join(base, "E1", "logs", f)
    if not os.path.exists(p):
        o(f + " MISSING")
        continue
    ls = read(p).split("\n")
    o("-- " + f + " lines=" + str(len(ls)))
    for l in ls[:8]:
        o("   " + l[:220])
    hits = [l for l in ls if re.search(r"p50|p95|median|latency|\bms\b", l, re.I)]
    o("   timing-lines=" + str(len(hits)))
    for l in hits[:14]:
        o("   > " + l[:220])

# 3. provenance by sha256 of .py files
o("\n=== 3 provenance: P4-OVBENCH .py files (sha12) and identical app files ===")
app_sha = {}
for dp, dn, fn in os.walk(APP):
    if skipped(dp):
        dn[:] = []
        continue
    for f in fn:
        if f.endswith(".py"):
            p = os.path.join(dp, f)
            app_sha.setdefault(sha(p), []).append(os.path.relpath(p, R))
for dp, dn, fn in os.walk(base):
    if skipped(dp):
        dn[:] = []
        continue
    for f in fn:
        if f.endswith(".py"):
            p = os.path.join(dp, f)
            h = sha(p)
            o(os.path.relpath(p, R) + " sha=" + h + " app_matches=" + str(app_sha.get(h, [])))

# 4. metric-like files under R/work
o("\n=== 4 metric-like files under R/work (name match), newest first ===")
pat = re.compile(r"(tape|det2|auroc|loo|hand|bench|latency|metric|eval|result|summary|verdict|triage|calib)", re.I)
wroot = os.path.join(R, "work")
found = []
for dp, dn, fn in os.walk(wroot):
    if skipped(dp) or dp[len(wroot):].count(os.sep) > 6:
        dn[:] = []
        continue
    for f in fn:
        if pat.search(f) and f.lower().endswith((".json", ".txt", ".md", ".csv", ".tsv", ".log")):
            p = os.path.join(dp, f)
            sz = os.path.getsize(p)
            if sz <= 400000:
                found.append((os.path.getmtime(p), p, sz))
found.sort(reverse=True)
o("candidates " + str(len(found)))
for mt, p, sz in found[:90]:
    keys = ""
    if p.endswith(".json") and sz < 200000:
        try:
            obj = json.loads(read(p))
            keys = ("keys=" + ",".join(list(obj.keys())[:12])) if isinstance(obj, dict) else "type=" + type(obj).__name__
        except ValueError:
            keys = "json-unreadable"
    o(os.path.relpath(p, R) + " bytes=" + str(sz) + " mtime=" + time.strftime("%m-%dT%H:%MZ", time.gmtime(mt)) + " " + keys)

# 5. AUROC / LOO / latency lines in those files
o("\n=== 5 lines mentioning AUROC, LOO, p50 or latency (first 2 per file, max 30 files) ===")
cnt = 0
for mt, p, sz in found:
    ls = [l for l in read(p).split("\n") if re.search(r"auroc|\bloo\b|p50|latency", l, re.I)]
    if ls:
        cnt += 1
        o(os.path.relpath(p, R))
        for l in ls[:2]:
            o("   " + l[:220])
        if cnt >= 30:
            break

# 6. specific files
o("\n=== 6 specific files ===")
checks = [
    "work/opus-verify/eval_top_roi.py", "work/opus-verify/top_roi", "work/opus-verify/label_xcheck.py",
    "logs/monitor", "logs/telemetry", "logs/ovrun", "app/anomaly/det2_f2/models/tape_thresholds.json",
    "app/tools/tests/fixtures/MANIFEST.tsv", "app/tools/tests/fixtures/hand_labels.csv",
    "app/motion/project_robots.json", "app/tools/jaw_calib.json", "app/tools/scan_calib.json",
]
for rel in checks:
    p = os.path.join(R, rel)
    if os.path.isdir(p):
        o(rel + "/ dir: " + ", ".join(sorted(os.listdir(p)))[:800])
    elif os.path.isfile(p):
        t = read(p)
        o(rel + " bytes=" + str(os.path.getsize(p)) + " head:")
        for l in t.split("\n")[:6]:
            o("   " + l[:200])
    else:
        o(rel + " MISSING")

# 7. tape bank files
o("\n=== 7 tape_bank and tape npz files ===")
for root in (APP, wroot, os.path.join(R, "logs")):
    for dp, dn, fn in os.walk(root):
        if skipped(dp):
            dn[:] = []
            continue
        for f in fn:
            if "tape_bank" in f or (f.endswith(".npz") and "tape" in f):
                p = os.path.join(dp, f)
                o(os.path.relpath(p, R) + " bytes=" + str(os.path.getsize(p)))

# 8. which detector and helpers the loop scripts use
o("\n=== 8 import and launch lines in loop scripts ===")
for rel in ("tools/sort_any.py", "tools/sort_loop.py", "tools/grasp_auto.py", "tools/hand_watch.py", "tools/tests/run_tests.py"):
    ls = read(os.path.join(APP, rel)).split("\n")
    o("-- " + rel)
    hit = [i for i, l in enumerate(ls) if re.search(r"det2|inspector|sys\.path|run_sequence|jaw_check|import sort_loop|conftest|rs_grab|mcmd", l)]
    for i in hit[:14]:
        o("   L" + str(i + 1) + ": " + ls[i].strip()[:200])

# 9. repos used by the code (HEAD) and URDF location and licence
o("\n=== 9 external repos, URDF and licence files ===")
for name in ("physicalai", "edge-developer-kit-reference-scripts", "anomalib", "model_api"):
    rp = os.path.join(R, "repos", name)
    if os.path.isdir(rp):
        r = subprocess.run(["git", "-C", rp, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=20)
        o("repo " + name + " HEAD=" + r.stdout.strip())
        for f in sorted(os.listdir(rp)):
            if f.upper().startswith(("LICENSE", "COPYING", "NOTICE")):
                o("   " + f + ": " + read(os.path.join(rp, f))[:160].replace("\n", " | "))
    else:
        o("repo " + name + " MISSING")
urdf = os.path.join(R, "repos", "edge-developer-kit-reference-scripts", "samples/ai/edge-ai-demo-studio/workers/services/robotics-ai/platforms/so101/urdf")
o("urdf dir exists=" + str(os.path.isdir(urdf)) + (" files=" + ",".join(sorted(os.listdir(urdf))) if os.path.isdir(urdf) else ""))

# 10. docs listing and file names referenced by markdown
o("\n=== 10 docs listing and file names referenced by markdown ===")
docs = os.path.join(R, "docs")
for f in sorted(os.listdir(docs)):
    p = os.path.join(docs, f)
    if os.path.isfile(p):
        o("docs/" + f + " bytes=" + str(os.path.getsize(p)))
for root in (docs, APP):
    for dp, dn, fn in os.walk(root):
        if skipped(dp):
            dn[:] = []
            continue
        for f in fn:
            if f.endswith(".md"):
                p = os.path.join(dp, f)
                names_ref = sorted(set(re.findall(r"[A-Za-z0-9_\-]+\.(?:py|sh|yaml|json|jsonl|csv|tsv)", read(p))))
                o(os.path.relpath(p, R) + " refs: " + ", ".join(names_ref)[:700])

# 11. logs and model folders (names and sizes only)
o("\n=== 11 logs subfolders and models folder (names only) ===")
ld = os.path.join(R, "logs")
for sub in ("telemetry", "ovrun", "detlive", "gm", "depth", "grasp_auto", "monitor", "sort"):
    d = os.path.join(ld, sub)
    if os.path.isdir(d):
        ns = sorted(os.listdir(d))
        o("logs/" + sub + "/ count=" + str(len(ns)) + " sample=" + ", ".join(ns[:10]))
md = os.path.join(R, "models")
if os.path.isdir(md):
    parts = []
    for f in sorted(os.listdir(md)):
        fp = os.path.join(md, f)
        parts.append(f + "(" + str(os.path.getsize(fp)) + ")" if os.path.isfile(fp) else f + "/")
    o("models/: " + ", ".join(parts)[:900])
dd = os.path.join(R, "discord")
if os.path.isdir(dd):
    o("discord/ count=" + str(len(os.listdir(dd))) + " names=" + ", ".join(sorted(os.listdir(dd)))[:300])

# 12. the start and resume scripts that exist (presence only)
o("\n=== 12 presence ===")
for rel in ("app/tools/start_all.sh", "app/tools/resume.sh", "app/tools/wrist_rec.sh", "app/motion/gm/spot_fast.yaml", "app/motion/gm/spot_grasp_v1.json", "app/tools/tests/fixtures/hand"):
    p = os.path.join(R, rel)
    o(rel + ": " + ("present" if os.path.exists(p) else "MISSING"))

with open(os.path.join(HERE, "probe.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print("WROTE", os.path.join(HERE, "probe.txt"), len(lines), "lines")
