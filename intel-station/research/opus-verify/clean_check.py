"""The v2_test brick (clean by the operator's account): per face, the robust spreads and the largest anomalous region
for a range of cutoffs, plus an overlay sheet of every face polygon on its view (work/export/clean_check.jpg)."""
import json
import os
import sys

import cv2
import numpy as np

R = "/home/intel/intel-junction-2026"
sys.path[:0] = [f"{R}/app/tools"]
import faces as fc  # noqa: E402
import survey as sv  # noqa: E402

ep = f"{R}/logs/sort/v2_test/ep_0001"
d = json.load(open(f"{ep}/episode.json"))
sp = sv.spec()
area = {"top": sp["footprint_mm"] ** 2, **{f"side{i}": sp["footprint_mm"] * sp["height_mm"] for i in range(4)}}
tiles = []
for src in ("survey", "survey_spot"):
    rec = d.get(src) or {}
    print(src, "measure", {k: rec.get("measure", {}).get(k) for k in ("ok", "xy", "yaw", "size_mm", "sep")},
          "views", rec.get("n_views"), "missing", rec.get("missing"), "s", rec.get("survey_s"))
    for f, r in sorted((rec.get("faces") or {}).items()):
        img = cv2.imread(f"{ep}/survey/{r.get('view')}.jpg")
        if img is None or not r.get("uv"):
            print("  ", f, "no image/poly", r.get("view"), r.get("reason"))
            continue
        uv = np.array(r["uv"])
        m = fc.face_mask(img.shape, uv, 0.10)
        dv = fc.deviations(img, m)
        regs = {}
        for k in (4, 6, 8, 10):
            reg, frac, _ = fc.region_mm2(img, uv, area[f], dv, k, k, 1.0, 1.5) if dv else (None, None, None)
            regs[k] = None if reg is None else round(reg, 1)
        print("  ", f, r.get("view"), "cos", r.get("cos"), "align", r.get("align"), "sC", None if not dv else round(dv["sC"], 2),
              "sL", None if not dv else round(dv["sL"], 2), "base", None if not dv else [round(x) for x in dv["base_lab"]],
              "n", None if not dv else dv["n_valid"], "region_mm2 by k", regs)
        vis = img.copy()
        cv2.polylines(vis, [uv.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.putText(vis, f"{src[7:] or 'here'} {f} {r.get('view')}", (8, 24), 0, 0.7, (255, 255, 255), 2)
        tiles.append(cv2.resize(vis, (320, 240)))
if tiles:
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(f"{R}/work/export/clean_check.jpg", np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    print("sheet", len(tiles))
