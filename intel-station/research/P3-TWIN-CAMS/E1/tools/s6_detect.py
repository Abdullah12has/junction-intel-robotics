#!/usr/bin/env python3
"""P3-TWIN-CAMS E1 step 6: red jaw detection (2D) with overlays. Read-only on recordings.

Red mask: H<=8 or H>=170, V>=50 (as close_pre_verdict in app/tools/sort_any.py), and a per-camera saturation floor:
top S>=150, close S>=165 (HSV medians measured in s8_hsv.py: jaws S 181-237 in close, 211-215 in top; the pink
2x2 brick in the close view has S 148 and is excluded by the close floor).
Components with area >= MIN_AREA are candidates. Per component: bbox, area, centroid, tip = (mean x of the bottom
TIP_ROWS rows, bottom-most row y), image coordinates (y down).
Close camera: candidates must touch the top image row (jaws enter from the top, as in close_pre_verdict).
Top camera: the two largest candidates are the jaws; the rest are logged as extra.
Modes:
  peek  : frames/peek_<cam>_<k>.jpg -> overlays/peek_<cam>_<k>_ov.jpg, out/detect_peek.json
  still : one frame per still window with dur >= 0.5 s, video time = joint centre + clock offset,
          -> out/detect_<cam>.json and overlays/still_<cam>_<win>.jpg
Run: envs/data-env/bin/python -I tools/s6_detect.py peek|still
"""
import json
import sys

import cv2
import numpy as np

R = "/home/intel/intel-junction-2026"
E1 = f"{R}/work/P3-TWIN-CAMS/E1"
RED_H, RED_V = 8, 50
RED_S = {"top": 150, "close": 165}
MIN_AREA = 150
TIP_ROWS = 4
VID = {"top": f"{R}/logs/monitor/top_20261114T170351Z.mkv",
       "close": f"{R}/logs/monitor/close_20261114T170351Z.mkv"}
OFFSET_S = {"top": 0.0333, "close": 0.0167}   # video_time = joint_time + offset (s5_offsets2.py medians)


def red_components(img, cam):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (((H <= RED_H) | (H >= 170)) & (S >= RED_S[cam]) & (V >= RED_V)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = [int(v) for v in st[i]]
        if area < MIN_AREA:
            continue
        ys, xs = np.nonzero(lab == i)
        yb = int(ys.max())
        tip_x = float(xs[ys >= yb - TIP_ROWS + 1].mean())
        comps.append({"area": area, "bbox": [x, y, w, h],
                      "centroid": [round(float(cen[i][0]), 2), round(float(cen[i][1]), 2)],
                      "tip": [round(tip_x, 2), float(yb)], "touch_top": bool(y <= 3)})
    return comps


def select(comps, cam):
    if cam == "close":
        cand = sorted([c for c in comps if c["touch_top"] and c["area"] > 800], key=lambda c: -c["area"])
    else:
        cand = sorted(comps, key=lambda c: -c["area"])
    return cand[:2], [c for c in comps if not any(c is s for s in cand[:2])]


def overlay(img, sel, extra):
    out = img.copy()
    for c in extra:
        x, y, w, h = c["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 0), 1)
    for j, c in enumerate(sel):
        x, y, w, h = c["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.circle(out, (int(round(c["tip"][0])), int(round(c["tip"][1]))), 5, (0, 255, 255), -1)
        cv2.putText(out, str(j), (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.drawMarker(out, (int(round(c["centroid"][0])), int(round(c["centroid"][1]))), (0, 255, 0),
                       cv2.MARKER_CROSS, 12, 2)
    return out


def peek():
    res = {}
    for cam in ("top", "close"):
        for k in range(6):
            img = cv2.imread(f"{E1}/frames/peek_{cam}_{k}.jpg")
            sel, extra = select(red_components(img, cam), cam)
            cv2.imwrite(f"{E1}/overlays/peek_{cam}_{k}_ov.jpg", overlay(img, sel, extra))
            res[f"{cam}_{k}"] = {"selected": sel, "extra_count": len(extra)}
            print(cam, k, "selected", [(c["area"], c["tip"]) for c in sel], "extra", len(extra))
    with open(f"{E1}/out/detect_peek.json", "w") as fh:
        json.dump(res, fh, indent=1)


def still():
    with open(f"{E1}/out/still_windows.json") as fh:
        segs = json.load(fh)["segments"]
    pts = {}
    for cam in ("top", "close"):
        with open(f"{E1}/out/pts_{cam}.csv") as fh:
            pts[cam] = np.sort(np.array([float(x.split(",")[0]) for x in fh if x.strip()]))
    for cam in ("top", "close"):
        cap = cv2.VideoCapture(VID[cam])
        rows = []
        for idx, s in enumerate(segs):
            if s["dur"] < 0.5:
                continue
            tc = 0.5 * (s["t0"] + s["t1"]) + OFFSET_S[cam]
            fi = int(np.argmin(np.abs(pts[cam] - tc)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, img = cap.read()
            if not ok:
                rows.append({"win": idx, "error": "read"})
                continue
            sel, extra = select(red_components(img, cam), cam)
            cv2.imwrite(f"{E1}/overlays/still_{cam}_{idx:03d}.jpg", overlay(img, sel, extra))
            rows.append({"win": idx, "episode": s["episode"], "t_joint_centre": round(0.5 * (s["t0"] + s["t1"]), 4),
                         "t_video": round(float(pts[cam][fi]), 4), "pose": s["pose_median"],
                         "jaws": [{"area": c["area"], "tip": c["tip"], "centroid": c["centroid"],
                                   "bbox": c["bbox"], "touch_top": c["touch_top"]} for c in sel],
                         "extra": len(extra)})
        with open(f"{E1}/out/detect_{cam}.json", "w") as fh:
            json.dump(rows, fh, indent=1)
        counts = {n: sum(1 for r in rows if len(r.get("jaws", [])) == n) for n in (0, 1, 2)}
        print(cam, "windows", len(rows), "jaws_per_frame", counts)


if __name__ == "__main__":
    if sys.argv[1] == "peek":
        peek()
    else:
        still()
