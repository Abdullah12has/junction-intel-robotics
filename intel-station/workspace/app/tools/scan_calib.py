#!/usr/bin/env python3
"""Opus I-60: with a brick under the hover, servo it to TARGET, align yaw, then raise SCAN_UP mm and record where it
appears from the scan height (target_scan) and the size ratio (scale of M at the scan height). Saves scan_calib.json.
Leaves the arm at the aligned hover above the brick."""
import json
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grasp_auto as ga  # noqa: E402
import pick_any as pa  # noqa: E402

st = ga.state()
q = st["present"][:5]
g = 25.0
b = pa.nearest(pa.blobs(pa.frame()), pa.target(), max_d=200)
print("start brick", b)
q, b, ok = pa.servo(q, g, b["centre"])
q, rot = pa.align_yaw(q, g, b)
if rot:
    b = pa.nearest(pa.blobs(pa.frame()), b["centre"], max_d=200)
    q, b, ok = pa.servo(q, g, b["centre"])
print("aligned", ok, q, b)
s_h = b["short"]
qs = pa.move_rel(q, g, [0, 0, pa.SCAN_UP], speed=10)
assert qs is not None, "scan ik failed"
time.sleep(0.3)
bs = pa.nearest(pa.blobs(pa.frame(), min_area=800), b["centre"], max_d=300)
print("scan view brick", bs)
out = {"t": time.strftime("%H:%M:%SZ", time.gmtime()), "q_hover": q, "q_scan": qs, "short_hover": s_h,
       "short_scan": bs["short"], "scale": round(bs["short"] / s_h, 4), "target_scan": bs["centre"],
       }
json.dump(out, open(pa.SCALF, "w"), indent=1)
print(json.dumps(out))
pa.move_rel(qs, g, [0, 0, -pa.SCAN_UP], speed=10)
