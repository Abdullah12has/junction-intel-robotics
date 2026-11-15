"""Verifier V2: numbered blob sheets (start frames) and grasp-frame sheets for visual labelling. Usage: python3 -I label_sheet.py HERE"""
import json
import os
import sys

import cv2
import numpy as np

HERE = sys.argv[1]
eps = json.load(open(os.path.join(HERE, "episodes.json")))
usable = [k for k in sorted(eps, key=int) if eps[k]["status"] == "ok"]


def find_plate(img, hlo, hhi, smin=90, vmin=90):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    m = ((h >= hlo) & (h <= hhi) & (s >= smin) & (v >= vmin)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    pts = np.column_stack(np.where(lab == i))[:, ::-1].astype(np.float32)
    (cx, cy), r = cv2.minEnclosingCircle(pts)
    return (float(cx), float(cy), float(r))


def blobs(img, plates):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    m = (((s > 110) & (v > 110)) | ((v > 175) & (s < 60))).astype(np.uint8) * 255
    m[330:, :] = 0
    m[:, :30] = 0
    m[:, 610:] = 0
    for (cx, cy, r) in plates:
        cv2.circle(m, (int(round(cx)), int(round(cy))), int(round(r * 1.08)), 0, -1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, ce = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        x, y, w, hh, a = [int(t) for t in st[i]]
        if 120 <= a <= 4000:
            out.append({"u": round(float(ce[i][0]), 1), "v": round(float(ce[i][1]), 1), "x": x, "y": y,
                        "w": w, "h": hh, "area": a})
    return out


ids = {}
start_imgs, grasp_imgs = [], []
for k in usable:
    img = cv2.imread(os.path.join(HERE, eps[k]["frame_start"]))
    pl = [find_plate(img, 88, 104), find_plate(img, 60, 86)]
    bs = blobs(img, pl)
    ids[k] = bs
    for i, b in enumerate(bs):
        cv2.rectangle(img, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), (0, 255, 0), 1)
        cv2.putText(img, str(i), (b["x"] + b["w"] + 2, b["y"] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(img, "ep%s START" % k, (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    start_imgs.append(img)
    g = cv2.imread(os.path.join(HERE, eps[k]["frame_grasp"]))
    cv2.putText(g, "ep%s GRASP (frame %s)" % (k, eps[k]["grasp_frame"]), (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 255), 2)
    grasp_imgs.append(g)
json.dump(ids, open(os.path.join(HERE, "blob_ids_start.json"), "w"), indent=0)


def sheet(imgs, prefix):
    for s in range(0, len(imgs), 4):
        grp = list(imgs[s:s + 4])
        while len(grp) < 4:
            grp.append(np.zeros((480, 640, 3), np.uint8))
        out = np.vstack([np.hstack(grp[0:2]), np.hstack(grp[2:4])])
        cv2.imwrite(os.path.join(HERE, "%s_%d.png" % (prefix, s // 4)), out)


sheet(start_imgs, "label_start")
sheet(grasp_imgs, "label_grasp")
print("usable", usable, "blob counts", {k: len(v) for k, v in ids.items()})
