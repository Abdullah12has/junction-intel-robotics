#!/usr/bin/env python3
"""P3-TWIN-CAMS E1 step 9a: read-only probe of the inputs for the camera fit (no camera or serial access).
Run: <R>/envs/anomalib-env/bin/python -I tools/s9a_probe.py
"""
import glob
import json
import os

import cv2
import numpy as np

R = "/home/intel/intel-junction-2026"
E1 = f"{R}/work/P3-TWIN-CAMS/E1"

g = json.load(open(f"{E1}/out/jaw_geometry.json"))
print("gripper_joint axis", g["gripper_joint"]["axis"], "limit_lower_upper_rad", g["gripper_joint"]["limit_lower_upper_rad"])
print("fixed tip gripper mm", g["fixed_jaw_tip_gripper_mm"], "| moving tip theta0 gripper mm", g["moving_jaw_tip_gripper_mm_at_theta0"], "| tcp mm", g["tcp_gripper_mm"])

for cam in ("top", "close"):
    rows = [r for r in json.load(open(f"{E1}/out/detect_{cam}.json")) if r.get("jaws")]
    eps = {}
    for r in rows:
        e = eps.setdefault(str(r["episode"]), [0, 0])
        e[0] += 1
        e[1] += int(len(r["jaws"]) == 2)
    print(cam, "windows with >=1 jaw", len(rows), "| episodes (windows, two-jaw):", eps)
    P = np.array([r["pose"] for r in rows], float)
    print(cam, "pose min", P.min(0).round(2).tolist(), "max", P.max(0).round(2).tolist())
    print(cam, "gripper (pos5) std across windows", round(float(P[:, 5].std()), 2))

for cam in ("top", "close"):
    im = cv2.imread(f"{E1}/frames/peek_{cam}_0.jpg")
    print(cam, "frame shape", None if im is None else im.shape)

wr = sorted(glob.glob(f"{R}/data/wrist/raw/w_*.mkv"))
tot = sum(os.path.getsize(p) for p in wr)
print("wrist segments", len(wr), "total MB", round(tot / 1e6, 1), "first", os.path.basename(wr[0]) if wr else None,
      "last", os.path.basename(wr[-1]) if wr else None)

for name in ("clock_offsets_stage1.json", "clock_offsets_stage2.json"):
    p = f"{E1}/out/{name}"
    if os.path.exists(p):
        d = json.load(open(p))
        print(name, "keys", list(d.keys())[:20])
        print(name, "summary", json.dumps({k: v for k, v in d.items() if not isinstance(v, list)})[:900])
    else:
        print(name, "MISSING")

ss = json.load(open(f"{E1}/out/still_windows.json"))
print("still_windows keys", list(ss.keys())[:10], "| segment keys", list(ss["segments"][0].keys()))
print("pts files", [os.path.exists(f"{E1}/out/pts_{c}.csv") for c in ("top", "close")])
ea = f"{R}/app/eyeinhand"
names = []
for root, _, files in os.walk(ea):
    for f in files:
        names.append(os.path.relpath(os.path.join(root, f), ea))
print("eyeinhand files", sorted(names)[:60])
