"""Opus: compare the hand labels (tests/fixtures/hand_labels.csv) with the P2-B-ANALYZE image classes (estops.csv)."""
import csv
import glob
import os

R = "/home/intel/intel-junction-2026"
mine = {os.path.basename(r["file"]): r for r in csv.DictReader(open(f"{R}/app/tools/tests/fixtures/hand_labels.csv"))}
theirs = {}
for r in csv.DictReader(open(f"{R}/work/P2-B-ANALYZE/E1/estops.csv")):
    if r["image_file"]:
        theirs.setdefault(r["image_file"], set()).add((r["image_class"], r["image_note"]))
files = sorted(os.path.basename(p) for p in glob.glob(f"{R}/logs/monitor/estop_*.jpg"))
MAP = {"hand": "REAL_HAND", "nohand": "FALSE_POSITIVE", "hand_partial": "REAL_HAND"}
agree = dis = 0
for f in files:
    m = mine.get(f)
    t = theirs.get(f, set())
    tc = {c for c, _ in t}
    ml = m["label"] if m else "-"
    ok = m is not None and MAP.get(ml) in tc and len(tc) == 1
    agree += ok
    dis += (not ok)
    if not ok:
        print(f"{f:32s} mine={ml:12s} holding={m['holding'] if m else '-'} theirs={sorted(tc) or '-'} "
              f"note={' | '.join(n for _, n in t)[:110]} mynote={(m or {}).get('note', '')[:60]}")
print("agree", agree, "disagree or missing", dis, "of", len(files))
