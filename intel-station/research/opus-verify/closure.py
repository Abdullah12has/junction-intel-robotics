"""List the source files the live system imports (modulefinder from each entry point), the data files those files
name, and the lines in them that need a scrub. Read-only. Usage: python -I closure.py APP_DIR"""
import modulefinder
import os
import re
import sys

APP = os.path.abspath(sys.argv[1])
ENTRY = ["tools/sort_any.py", "tools/hand_watch.py", "tools/motion_server.py", "tools/jaw_check.py",
         "tools/run_report.py", "tools/cycle_peek.py", "tools/tests/run_tests.py"]
paths = [os.path.join(APP, d) for d in ("tools", "tools/tests", "anomaly", "motion", "pick_v2", ".")] + sys.path
mods = set()
for e in ENTRY:
    f = os.path.join(APP, e)
    if not os.path.exists(f):
        print("missing entry", e)
        continue
    mf = modulefinder.ModuleFinder(path=[os.path.dirname(f)] + paths)
    try:
        mf.run_script(f)
    except Exception as ex:  # noqa: BLE001
        print("modulefinder error", e, repr(ex)[:200])
    for name, m in mf.modules.items():
        if m.__file__ and os.path.abspath(m.__file__).startswith(APP + os.sep):
            mods.add(os.path.relpath(os.path.abspath(m.__file__), APP))
    mods.add(e)
for t in os.listdir(os.path.join(APP, "tools/tests")):
    if t.endswith(".py"):
        mods.add(f"tools/tests/{t}")
print("== CLOSURE", len(mods))
for m in sorted(mods):
    print(m)
lit = re.compile(r"[\"']([^\"'\s]*\.(?:json|jsonl|yaml|yml|png|tsv|csv|npz|npy|xml|bin|urdf))[\"']")
data = set()
for m in sorted(mods):
    for i, ln in enumerate(open(os.path.join(APP, m), errors="replace"), 1):
        for x in lit.findall(ln):
            data.add((m, x))
print("== DATA LITERALS", len(data))
for m, x in sorted(data):
    print(m, x)
ai = re.compile(r"\b(opus|claude|anthropic|haiku|chatgpt|llm)\b|opus[-_]|_opus", re.I)
print("== AI WORDS IN CLOSURE")
for m in sorted(mods):
    for i, ln in enumerate(open(os.path.join(APP, m), errors="replace"), 1):
        if ai.search(ln) or re.search(r"Hassan|hsn07pk", ln):
            print(f"{m}:{i}: {ln.strip()[:150]}")
for sh in ("tools/start_all.sh", "tools/stop_all.sh", "tools/monitor_cams.sh", "tools/wrist_rec.sh", "tools/resume.sh"):
    p = os.path.join(APP, sh)
    if os.path.exists(p):
        for i, ln in enumerate(open(p, errors="replace"), 1):
            if ai.search(ln) or re.search(r"Hassan|hsn07pk", ln):
                print(f"{sh}:{i}: {ln.strip()[:150]}")
