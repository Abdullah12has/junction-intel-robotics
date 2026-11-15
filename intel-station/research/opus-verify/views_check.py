"""views_here on recorded hover poses (no robot): how many views, which faces, how long the planning takes."""
import glob
import json
import os
import sys
import time

import cv2

R = "/home/intel/intel-junction-2026"
sys.path[:0] = [f"{R}/app/tools"]
import kin  # noqa: E402
import survey as sv  # noqa: E402

sp = sv.spec()
lib = sv.library()
crit = json.load(open(sv.LIB_F))["criteria"]
jm = cv2.imread(f"{R}/app/tools/wrist_jaw_mask.png", 0) > 0
for epj in sorted(glob.glob(f"{R}/logs/sort/any_2127/ep_*/episode.json"))[:12]:
    d = json.load(open(epj))
    q, bh = d.get("hover_q"), d.get("hover_blob")
    if not q or not bh:
        continue
    xy = [float(v) for v in sv.wc.pixel_to_mat(q, bh["centre"], kin.MAT_Z + sp["height_mm"])]
    t = time.time()
    v = sv.views_here(q, xy, kin.psi_of(q), sp, jm, lib, [], 10.0, crit["min_cos"], crit["max_jaw"], crit["max_occ"])
    dt = time.time() - t
    print(os.path.basename(os.path.dirname(epj)), f"xy {xy[0]:.0f},{xy[1]:.0f}", f"{len(v['views'])} views {dt:.2f} s",
          [vv["faces"] for vv in v["views"]], "missing", v["missing"])
