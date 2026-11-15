#!/usr/bin/env python3
"""Opus I-59: brick pose in the wrist camera relative to the jaws.
Blue (or any saturated non-red) brick blob -> minAreaRect: centre, size, yaw of its edges vs the image axes
(the jaws close along the image x axis in this camera). Red jaw blobs at the bottom give the jaw gap centre.
usage: wrist_brick.py IMAGE [--color blue|yellow|white]"""
import json
import sys

import cv2
import numpy as np

img = cv2.imread(sys.argv[1])
color = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--color" else "blue"
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
if color == "blue":
    m = (H >= 95) & (H <= 125) & (S >= 120) & (V >= 80)
elif color == "yellow":
    m = (H >= 18) & (H <= 35) & (S >= 100) & (V >= 120)
else:
    m = (S <= 40) & (V >= 170)
m = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
n, lab, st, cen = cv2.connectedComponentsWithStats(m)
out = {}
if n > 1:
    i = 1 + int(np.argmax(st[1:, 4]))
    ys, xs = np.nonzero(lab == i)
    (cx, cy), (w, h), ang = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
    yaw = ang % 90
    if yaw > 45:
        yaw -= 90
    out["brick"] = {"area": int(st[i, 4]), "centre": [round(cx, 1), round(cy, 1)], "size": [round(w, 1), round(h, 1)],
                    "edge_yaw_deg": round(float(yaw), 1)}
red = ((H <= 8) | (H >= 170)) & (S >= 120) & (V >= 70)
red = cv2.morphologyEx(red.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
n, lab, st, cen = cv2.connectedComponentsWithStats(red)
jaws = sorted([(int(st[i, 4]), i) for i in range(1, n) if st[i, 4] > 1500], reverse=True)[:2]
if len(jaws) == 2:
    tips = []
    for _, i in jaws:
        ys, xs = np.nonzero(lab == i)
        top = ys.min()
        sel = ys <= top + 8
        tips.append([float(xs[sel].mean()), float(top)])
    tips.sort()
    out["jaw_tips_px"] = [[round(t[0], 1), round(t[1], 1)] for t in tips]
    out["jaw_gap_centre_x"] = round((tips[0][0] + tips[1][0]) / 2, 1)
    if "brick" in out:
        out["brick_minus_gap_x"] = round(out["brick"]["centre"][0] - out["jaw_gap_centre_x"], 1)
print(json.dumps(out))
