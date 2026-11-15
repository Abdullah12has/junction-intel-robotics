"""Scratch probe of builder E1: run the CURRENT functions on the fixtures and print results (read only)."""
import json
import os
import sys
import tempfile
import time

import cv2
import numpy as np

sys.path.insert(0, "/home/intel/intel-junction-2026/app/tools")
sys.path.insert(0, "/home/intel/intel-junction-2026/app/anomaly/det2_f2")
t0 = time.time()
import pick_any as pa  # noqa: E402
import sort_any as sa  # noqa: E402
import kin  # noqa: E402
print("import_s", round(time.time() - t0, 2), "target", pa.target().tolist())
FIX = "/home/intel/intel-junction-2026/app/tools/tests/fixtures"
LOG = "/home/intel/intel-junction-2026/logs/sort/any_1740"
sa.RUN = tempfile.mkdtemp(prefix="probe_run_")

print("== hover (fixture 1)")
for ep in ["0001", "0006", "0013", "0015", "0017"]:
    d = json.load(open(f"{LOG}/ep_{ep}/episode.json"))
    hb = d["hover_blob"]
    img = cv2.imread(f"{FIX}/hover/ep{ep}_wrist_hover.png")
    bl = pa.blobs(img, min_area=1500)
    b = pa.find(bl, pa.target(), 260, hb.get("colour"))
    print("ep", ep, "hover_blob keys", sorted(hb.keys()), "rec", hb.get("centre"), hb.get("short"), hb.get("colour"),
          "| found", None if b is None else (b["centre"], b["short"], b["colour"], b["area"], b["hue"]), "n_blobs", len(bl))

print("== servo (fixture 2)")
lines = open("/home/intel/intel-junction-2026/work/P2-B-TESTS/E1/raw/L19_extract_summary.tsv").read().strip().split("\n")[1:]
for ln in lines:
    f = ln.split("\t")
    if f[2] == "white" and f[3] == "" :
        continue
    img = cv2.imread(f"{FIX}/servo/{f[0]}")
    bl = pa.blobs(img, min_area=pa.SERVO_MIN_AREA)
    rec = np.array([float(f[3]), float(f[4])])
    b_seed = pa.find(bl, rec, 140, f[2])
    b_tgt = pa.find(bl, pa.target(), 140, f[2])
    b_any = pa.find(bl, pa.target(), 140, None)
    fmt = lambda b: None if b is None else (b["centre"], b["short"], b["colour"])
    print(f[0], "rec", f[3], f[4], f[5], "| seed", fmt(b_seed), "| target_want", fmt(b_tgt), "| target_any", fmt(b_any),
          "| n", len(bl))

print("== split (fixture 4)")
for fn in sorted(os.listdir(f"{FIX}/split")):
    img = cv2.imread(f"{FIX}/split/{fn}")
    bl = pa.blobs(img, min_area=1500)
    wh = [(b["centre"], b["short"], b["area"]) for b in bl if b["colour"] == "white"]
    print(fn, "n_white", len(wh), wh, "n_all", len(bl))

print("== scan (fixture 3)")
c = json.load(open(f"{FIX}/scan_calib.json"))
img = cv2.imread(f"{FIX}/scan_now.jpg")
allb = pa.blobs(img, min_area=700, max_area=40000)
print("all blobs", [(b["colour"], b["centre"], b["short"], b["area"], b["plate_frac"], b["border"]) for b in allb])
cands = sa.candidates(img, c["q_scan"], c)
print("candidates", [(b["colour"], b["centre"], b["short"], b["area"], b["plate_frac"], b["d_mm"], b["xy"]) for b in cands])

print("== close pre-verdict (fixture 5)")
for ep in ["0001", "0006", "0008", "0013", "0014", "0015", "0017"]:
    d = json.load(open(f"{LOG}/ep_{ep}/episode.json"))
    pc = d.get("pre_close")
    hb = d.get("hover_blob")
    wh = hb.get("hue") if isinstance(hb, dict) else None
    img = cv2.imread(f"{FIX}/close/ep{ep}_pre_close_full.jpg")
    v = sa.close_pre_verdict(img, want_hue=wh)
    v0 = sa.close_pre_verdict(img, want_hue=None)
    print("ep", ep, "pre_close_keys", sorted(pc.keys()) if isinstance(pc, dict) else pc, "| want_hue", wh,
          "| verdict", v.get("verdict") if v else None, v.get("reasons") if v else None,
          "| rec", pc.get("verdict"), pc.get("reasons"), "| no-hue verdict", v0.get("verdict") if v0 else None,
          "| hue", v.get("hue") if v else None, "dist", v.get("dist") if v else None)

print("== kin (fixture 7)")
q_scan = c["q_scan"]
q_hover = c["q_hover"]
p_s, _ = kin.fk(q_scan)
p_h, _ = kin.fk(q_hover)
print("fk q_scan", np.round(p_s, 2).tolist(), "fk q_hover", np.round(p_h, 2).tolist(),
      "MAT_Z+HOVER_DZ", round(kin.MAT_Z + kin.HOVER_DZ, 2))
errs = []
for dx in (-60, -30, 0, 30, 60):
    for dy in (-60, -30, 0, 30, 60):
        d = np.array([dx, dy, 0.0])
        qn = kin.ik_rel(q_hover, d)
        if qn is None:
            errs.append(("NONE", dx, dy))
            continue
        e = np.linalg.norm(kin.fk(qn)[0] - kin.fk(q_hover)[0] - d)
        errs.append((round(float(e), 3), dx, dy))
print("ik_rel round trip (err_mm, dx, dy)", errs)
print("max_err", max(e[0] for e in errs if e[0] != "NONE") if all(e[0] != "NONE" for e in errs) else "has NONE")
