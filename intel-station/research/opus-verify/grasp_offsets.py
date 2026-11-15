"""Where was the brick relative to the jaws at the moment the gripper started to close? For every episode with a
grasp_open wrist frame: the brick blob nearest the jaw gap, its centre minus the jaw gap centre (jaw tips from the
jaw mask in use at the time: the static mask file is used for all, tips found in the frame itself when possible),
and whether the grasp held. Successful centred grasps give the reference offset; recent misses show the error."""
import glob
import json
import os
import sys

import cv2
import numpy as np

R = "/home/intel/intel-junction-2026"
sys.path.insert(0, f"{R}/app/tools")
import pick_any as pa  # noqa: E402


def tips_in(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    r = (((H >= 165) | (H <= 4)) & (S >= 85) & (V >= 55)).astype(np.uint8)
    r[:180] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(r)
    tips = []
    for i in range(1, n):
        if st[i, 4] > 1500 and st[i, 1] + st[i, 3] >= img.shape[0] - 3:
            ys, xs = np.nonzero(lab == i)
            top = ys.min()
            tips.append([float(xs[ys <= top + 8].mean()), int(top)])
    return sorted(tips)


rows = []
for run in sys.argv[1:]:
    for ep in sorted(glob.glob(f"{R}/logs/sort/{run}/ep_*")):
        for tag in ("", "re_"):
            f = f"{ep}/{tag}grasp_open_wrist.jpg"
            if not os.path.exists(f):
                continue
            img = cv2.imread(f)
            try:
                d = json.load(open(f"{ep}/episode.json"))
            except (OSError, ValueError):
                d = {}
            tips = tips_in(img)
            bl = pa.blobs(img, min_area=3000, max_area=200000)
            b = pa.nearest(bl, pa.target(), max_d=400)
            if len(tips) != 2 or b is None:
                rows.append((run, os.path.basename(ep), tag, None, None, d.get(tag + "grip_closed"), d.get("success")))
                continue
            gx = (tips[0][0] + tips[1][0]) / 2
            ty = max(tips[0][1], tips[1][1])
            rows.append((run, os.path.basename(ep), tag, round(b["centre"][0] - gx, 1), round(b["centre"][1] - ty, 1),
                         d.get(tag + "grip_closed"), d.get("success"), round(b["short"]), [round(t[0]) for t in tips]))
for r in rows:
    print(r)
