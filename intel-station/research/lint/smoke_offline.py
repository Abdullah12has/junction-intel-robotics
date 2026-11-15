"""Offline smoke checks of the 18:40-18:58Z changes (no robot, no cameras): handdet labels, sort_loop.hands_seen,
sort_loop.decide branches, sort_any import + candidates on a recorded scan frame. Exit code 1 on any failure."""
import csv
import os
import sys
import time

import cv2

R = "/home/intel/intel-junction-2026"
sys.path[:0] = [f"{R}/app/tools", f"{R}/app/anomaly/det2_f2"]
import handdet  # noqa: E402

fails = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


# 1) hand labels
for r in csv.DictReader(open(f"{R}/app/tools/tests/fixtures/hand_labels.csv")):
    if r["view"] == "wrist" or r["label"] == "hand_partial":
        continue
    img = cv2.imread(f"{R}/{r['file']}")
    got, f = handdet.is_hand(img, r["view"], r["holding"] == "1")
    check(got == (r["label"] == "hand"), f"handdet {r['view']} {os.path.basename(r['file'])} label={r['label']} got={got} area={f['area']}{f['edge']}")

# 2) hands_seen reads the shared log and the shared detector
import sort_loop as sl  # noqa: E402

logf = f"{R}/logs/monitor/hand_watch.log"
size0 = os.path.getsize(logf) if os.path.exists(logf) else 0
check(sl.hands_seen(5) is False, "hands_seen False with no recent log line and stale frames")
with open(logf, "a") as fh:
    fh.write(f"{time.strftime('%H:%M:%SZ', time.gmtime())} HAND? view=wrist blob=25000px edge=- hsv=12/90/180 active=False hits=1 (smoke test line)\n")
check(sl.hands_seen(5) is True, "hands_seen True on a fresh wrist line (wrist lines now count)")
with open(logf, "r+") as fh:  # remove the smoke line again
    fh.truncate(size0)

# 3) decide() branches
shape = ("2x2", "aspect 1.0")
p, lab, conf, why = sl.decide({"verdict": "UNCERTAIN"}, shape, [])
check((p, lab, conf) == ("blue", "GOOD", "no_evidence"), f"decide no views -> {p} {lab} {conf}")
p, lab, conf, why = sl.decide({"verdict": "CONFIDENT_DEFECT", "reasons": ["x"]}, shape, [])
check((p, lab) == ("green", "DEFECT"), f"decide pre defect -> {p} {lab}")
p, lab, conf, why = sl.decide({"verdict": "UNCERTAIN"}, ("not_2x2", "aspect 2.0"), [])
check((p, lab) == ("green", "DEFECT"), f"decide not_2x2 -> {p} {lab}")
p, lab, conf, why = sl.decide({"verdict": "UNCERTAIN"}, shape, [{"verdict": "CONFIDENT_GOOD"}])
check((p, lab, conf) == ("blue", "GOOD", "held_only"), f"decide clean station good -> {p} {lab} {conf}")

# 4) sort_any imports and finds the bricks in a recorded scan frame
import sort_any as S  # noqa: E402

S.RUN = f"{R}/work/lint/smoke_run"
os.makedirs(S.RUN, exist_ok=True)
S.fit_scan_corr()
c = S.scan_cal()
img = cv2.imread(f"{R}/logs/depth/servo/scan_now.jpg")
cands = S.candidates(img, c["q_scan"], c)
cols = sorted(b.get("colour") for b in cands)
check(len(cands) >= 2 and "red" in cols and any(x in cols for x in ("yellow", "white")),
      f"candidates on scan_now.jpg: {[(b.get('colour'), b['centre']) for b in cands]}")
print(f"{len(fails)} failures")
sys.exit(1 if fails else 0)
