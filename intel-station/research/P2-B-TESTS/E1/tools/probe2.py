"""Scratch probe 2 of builder E1 (read only): open questions behind the test expectations."""
import ast
import json
import math
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, "/home/intel/intel-junction-2026/app/tools")
sys.path.insert(0, "/home/intel/intel-junction-2026/app/anomaly/det2_f2")
import pick_any as pa  # noqa: E402
import sort_any as sa  # noqa: E402
import sort_loop as sl  # noqa: E402
import kin  # noqa: E402

FIX = "/home/intel/intel-junction-2026/app/tools/tests/fixtures"
LOG = "/home/intel/intel-junction-2026/logs/sort/any_1740"
RAW = "/home/intel/intel-junction-2026/work/P2-B-TESTS/E1/raw"
sa.RUN = tempfile.mkdtemp(prefix="probe2_")
sl.RUN = sa.RUN

print("== (a) close verdict with want_hue = pre_close.hue + hue_diff (ep0008, ep0014)")
for ep in ["0008", "0014"]:
    pc = json.load(open(f"{LOG}/ep_{ep}/episode.json"))["pre_close"]
    want = (pc["hue"] + pc["hue_diff"]) % 180
    img = cv2.imread(f"{FIX}/close/ep{ep}_pre_close_full.jpg")
    v = sa.close_pre_verdict(img, want_hue=want)
    print("ep", ep, "want", want, "now", v["verdict"], v["reasons"], "verdict_raw", v.get("verdict_raw"),
          "| recorded", pc["verdict"], pc["reasons"], pc.get("verdict_raw"), "| hue now", v["hue"], "rec", pc["hue"])

print("== (b) scan blobs: reach and plate keep-out of every blob")
c = json.load(open(f"{FIX}/scan_calib.json"))
img = cv2.imread(f"{FIX}/scan_now.jpg")
p, _ = kin.fk(c["q_scan"])
for b in pa.blobs(img, min_area=700, max_area=40000):
    d = sa.brick_xy_from_scan(c["q_scan"], b, c)
    xy = (p[0] + d[0], p[1] + d[1])
    r = math.hypot(xy[0] - kin.PAN_XY_MM[0], xy[1] - kin.PAN_XY_MM[1])
    pk = min(math.hypot(xy[0] - v[0], xy[1] - v[1]) for v in sa.PLATES_XY.values())
    print("blob", b["colour"], b["centre"], "xy", np.round(xy, 1).tolist(), "r", round(r, 1),
          "plate_keepout", round(pk, 1), "plate_frac", b["plate_frac"], "border", b["border"])

print("== (c) ik_rel at q_hover for the two None points, default tol and relaxed tol")
qh = c["q_hover"]
for d in ([60.0, 30.0, 0.0], [60.0, 60.0, 0.0]):
    dflt = kin.ik_rel(qh, d)
    relax = kin.ik_rel(qh, d, tol_mm=5.0, tol_rot=0.2)
    print("d", d, "default", dflt, "relaxed(5mm,0.2rad)", relax)

print("== (d) hand_area via ast extraction (no module import), 41 estop frames")
src = open("/home/intel/intel-junction-2026/app/tools/hand_watch.py", encoding="utf-8").read()
tree = ast.parse(src)
keep = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == "hand_area":
        keep.append(node)
    elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("ROI", "MIN_AREA") for t in node.targets):
        keep.append(node)
mod = ast.Module(body=keep, type_ignores=[])
ast.fix_missing_locations(mod)
ns = {"cv2": cv2, "np": np}
exec(compile(mod, "hand_watch.py(extract)", "exec"), ns)
rows = []
for fn in sorted(os.listdir(f"{FIX}/hand")):
    view = fn.split("_")[1]
    a = ns["hand_area"](cv2.imread(f"{FIX}/hand/{fn}"), view)
    rows.append((fn, view, int(a), ns["MIN_AREA"][view]))
with open(f"{RAW}/L22_hand_areas.tsv", "w") as f:
    f.write("file\tview\tarea_px\tmin_area_px\tabove_min\n")
    for fn, view, a, m in rows:
        f.write(f"{fn}\t{view}\t{a}\t{m}\t{int(a >= m)}\n")
for view in ("close", "top", "wrist"):
    vals = [a for fn, v, a, m in rows if v == view]
    print(view, "n", len(vals), "min", min(vals), "max", max(vals), "above_min", sum(1 for fn, v, a, m in rows if v == view and a >= m))
