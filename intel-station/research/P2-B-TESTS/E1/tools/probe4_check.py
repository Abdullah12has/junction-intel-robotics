import collections
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/intel/intel-junction-2026/app/tools")
sys.path.insert(0, "/home/intel/intel-junction-2026/app/anomaly/det2_f2")
import pick_any as pa  # noqa: E402

FIX = "/home/intel/intel-junction-2026/app/tools/tests/fixtures"
EP_ROOT = "/home/intel/intel-junction-2026/logs/sort/any_1740"

print("== F-1: blobs near the logged centre of servo_180002_red.png (log line 104: 422.7, 268.8, short 112.6)")
img = cv2.imread(os.path.join(FIX, "servo", "servo_180002_red.png"))
seed = np.array([422.7, 268.8])
bl = pa.blobs(img, min_area=pa.SERVO_MIN_AREA)
print("blob count at min_area", pa.SERVO_MIN_AREA, len(bl))
near = sorted(bl, key=lambda b: float(np.hypot(*(np.array(b["centre"]) - seed))))
for b in near[:6]:
    d = float(np.hypot(*(np.array(b["centre"]) - seed)))
    print(" ", b["colour"], [round(float(v), 1) for v in b["centre"]], "short", round(float(b["short"]), 1),
          "plate_frac", round(float(b["plate_frac"]), 2), "dist", round(d, 1))
hit = pa.find(bl, seed, 140, "red")
print("find(seed, 140, red):",
      None if hit is None else (hit["colour"], [round(float(v), 1) for v in hit["centre"]], round(float(hit["short"]), 1)))

print("== episode.json pre_close and hover_blob (logs/sort/any_1740)")
found = {}
for root, dirs, files in os.walk(EP_ROOT):
    if "episode.json" in files:
        found[os.path.basename(root)] = os.path.join(root, "episode.json")
print("episode.json files", len(found), sorted(found))
for name in sorted(found):
    with open(found[name], encoding="utf-8") as f:
        ep = json.load(f)
    if name.endswith("0004"):
        print("0004 top-level keys", sorted(ep.keys()), "has pre_close:", "pre_close" in ep)
    pc = ep.get("pre_close")
    hb = ep.get("hover_blob")
    if pc is None and hb is None:
        continue
    pcs = None if pc is None else {k: pc.get(k) for k in ("verdict", "reasons", "hue", "hue_diff", "colour")}
    hbs = None if hb is None else {k: hb.get(k) for k in ("centre", "short", "colour", "hue")}
    want = None
    if hb is not None and hb.get("hue") is not None:
        want = hb["hue"]
    elif pc is not None and pc.get("hue") is not None and pc.get("hue_diff") is not None:
        want = pc["hue"] + pc["hue_diff"]
    print(name, "pre_close", pcs, "hover_blob", hbs, "want_hue_from_episode", want)

print("== hand_labels.csv: counts")
with open(os.path.join(FIX, "hand_labels.csv"), encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
print("rows", len(rows))
print("view,label", sorted(collections.Counter((r["view"], r["label"]) for r in rows).items()))
print("wrist holding", sorted(collections.Counter(r["holding"] for r in rows if r["view"] == "wrist").items()))
