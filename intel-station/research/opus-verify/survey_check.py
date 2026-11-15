"""Check the survey geometry on recorded hover frames (no robot): for every episode with hover_q and
crops/wrist_hover.png, predict the brick from the camera model (blob centre at the hover -> brick-top plane), measure
it with survey.measure_top, project the measured box and align it. Prints per-episode numbers and a summary; writes a
contact sheet of overlays (work/export/survey_check.jpg)."""
import glob
import json
import math
import os
import sys

import cv2
import numpy as np

R = "/home/intel/intel-junction-2026"
sys.path[:0] = [f"{R}/app/tools"]
import kin  # noqa: E402
import survey as sv  # noqa: E402

sp = sv.spec()
rows, tiles = [], []
for epj in sorted(glob.glob(f"{R}/logs/sort/any_*/ep_*/episode.json")):
    d = json.load(open(epj))
    ep = os.path.dirname(epj)
    img = cv2.imread(f"{ep}/crops/wrist_hover.png")
    q, bh = d.get("hover_q"), d.get("hover_blob")
    if img is None or not q or not bh:
        continue
    z_top = kin.MAT_Z + sp["height_mm"]
    xy0 = sv.wc.pixel_to_mat(q, bh["centre"], z_top)
    yaw0 = kin.psi_of(q)
    m = sv.measure_top(img, q, xy0, yaw0, sp)
    if not m.get("ok"):
        rows.append((ep, None))
        print(os.path.relpath(ep, f"{R}/logs/sort"), "MEASURE FAILED", m)
        continue
    faces = sv.brick_faces(m["xy"], m["yaw"], (sp["footprint_mm"],) * 2, sp["height_mm"], kin.MAT_Z)
    proj = sv.project_faces(q, faces)
    al = sv.align_shift(img, proj, 24)
    dxy = float(np.hypot(m["xy"][0] - xy0[0], m["xy"][1] - xy0[1]))
    rows.append((ep, (dxy, m["size_mm"], m["sep"], al.get("iou"), al.get("shift"), d.get("shape"), d.get("target_colour"))))
    print(os.path.relpath(ep, f"{R}/logs/sort"), f"d_prior {dxy:.1f} mm size {m['size_mm']} sep {m['sep']} "
          f"align iou {al.get('iou')} shift {al.get('shift')} shape {d.get('shape')} colour {d.get('target_colour')}",
          {f: (round(r['cos'], 2), round(r['area_px'])) for f, r in proj.items() if r['cos'] > 0})
    if len(tiles) < 12:
        vis = img.copy()
        for f, r in proj.items():
            if r["cos"] > 0:
                uv = (r["uv"] + np.array(al.get("shift", [0, 0]))).astype(np.int32)
                cv2.polylines(vis, [uv], True, (0, 255, 0) if f == "top" else (255, 0, 255), 2)
        cv2.putText(vis, os.path.relpath(ep, f"{R}/logs/sort"), (8, 24), 0, 0.7, (255, 255, 255), 2)
        tiles.append(cv2.resize(vis, (320, 240)))
ok = [r[1] for r in rows if r[1]]
print("SUMMARY", len(rows), "frames", len(ok), "measured;",
      "median d_prior", round(float(np.median([o[0] for o in ok])), 1) if ok else None,
      "median iou", round(float(np.median([o[3] for o in ok if o[3] is not None])), 3) if ok else None,
      "sizes", sorted(round(o[1][0]) for o in ok))
if tiles:
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(f"{R}/work/export/survey_check.jpg", sheet)
