#!/usr/bin/env python3
"""Opus I-60: re-measure the jaws in the wrist camera (gripper open 25, any arm pose) and rebuild the static jaw
mask + grasp target pixel. Target = gap centre x + 3 px (16:46Z verified grasp had the brick 3 px right of the
gap centre), y = 268 (hover height). Writes app/tools/wrist_jaw_mask.png and app/tools/jaw_calib.json.
usage: jaw_calib.py [--image PATH]   (default: the live wrist frame)"""
import argparse
import json
import time

import cv2
import numpy as np

T = "/home/intel/intel-junction-2026/app/tools"
ap = argparse.ArgumentParser()
ap.add_argument("--image", default="/home/intel/intel-junction-2026/run/live_wrist.jpg")
a = ap.parse_args()
img = cv2.imread(a.image)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
red = (((H >= 165) | (H <= 3)) & (S >= 90) & (V >= 60)).astype(np.uint8)
red[:200] = 0
n, lab, st, _ = cv2.connectedComponentsWithStats(red)
jaws = [i for i in range(1, n) if st[i, 4] > 1500 and st[i, 1] + st[i, 3] >= img.shape[0] - 3]
m = np.zeros_like(red)
tips = []
for i in jaws:
    m[lab == i] = 1
    ys, xs = np.nonzero(lab == i)
    top = ys.min()
    sel = ys <= top + 8
    tips.append([round(float(xs[sel].mean()), 1), int(top)])
tips.sort()
m = cv2.dilate(cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)), np.ones((19, 19), np.uint8))
out = {"t": time.strftime("%H:%M:%SZ", time.gmtime()), "image": a.image, "jaw_tips": tips, "mask_px": int(m.sum())}
if len(tips) == 2:
    gx = (tips[0][0] + tips[1][0]) / 2
    out.update({"gap_centre_x": round(gx, 1), "target_px": [round(gx + 3, 1), 268.0], "gap_px": round(tips[1][0] - tips[0][0], 1)})
    cv2.imwrite(f"{T}/wrist_jaw_mask.png", m * 255)
    json.dump(out, open(f"{T}/jaw_calib.json", "w"), indent=1)
    vis = img.copy()
    vis[m > 0] = (0, 255, 0)
    cv2.circle(vis, (int(gx + 3), 268), 8, (255, 0, 255), 2)
    cv2.imwrite("/home/intel/intel-junction-2026/logs/depth/servo/jaw_calib.jpg", vis)
else:
    out["error"] = f"found {len(tips)} jaws; mask not written"
print(json.dumps(out))
