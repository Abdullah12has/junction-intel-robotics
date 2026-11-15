#!/usr/bin/env python3
"""Opus I-60 jaw check + static jaw mask (robust to bricks under the jaws).
At the hover with the gripper open (25): frames at two wrist rolls; the jaws stay fixed in the wrist image while the
background rotates, so red pixels present in BOTH frames are the jaws. Compares the jaw tips with the reference
(16:42Z: left 235.7,320  right 544.6,329). Writes wrist_jaw_mask.png and jaw_calib.json when the jaws are within
tolerance, else exits 3 (jaw moved: loose fixed jaw or bent).
usage: jaw_check.py [--tol 18] [--no-move]"""
import argparse
import os
import json
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grasp_auto as ga  # noqa: E402

T = os.path.dirname(os.path.abspath(__file__))
REF = [[235.7, 320], [544.6, 329]]


def redmask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    r = (((H >= 165) | (H <= 4)) & (S >= 85) & (V >= 55)).astype(np.uint8)
    r[:180] = 0
    return r


def jaws(m, h):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    tips = []
    for i in range(1, n):
        if st[i, 4] > 1500 and st[i, 1] + st[i, 3] >= h - 3:
            keep[lab == i] = 1
            ys, xs = np.nonzero(lab == i)
            top = ys.min()
            tips.append([round(float(xs[ys <= top + 8].mean()), 1), int(top)])
    return keep, sorted(tips)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=18.0)
    ap.add_argument("--no-move", action="store_true")
    a = ap.parse_args()
    st = ga.state()
    q = list(st["target"])
    q[5] = 25.0
    frames = []
    if not a.no_move:
        ga.move(q, speed=8, comp=True)
        for dr in (0.0, 30.0):
            qq = list(q)
            qq[4] = round(q[4] + dr, 2)
            ga.move(qq, speed=15, comp=False if dr else True)
            time.sleep(0.4)
            frames.append(ga.fresh("wrist", time.time() + 0.2))
        ga.move(q, speed=15, comp=True)
    else:
        frames.append(ga.fresh("wrist", time.time()))
    m = frames[0] if len(frames) == 1 else None
    r = redmask(frames[0])
    for f in frames[1:]:
        r = r & redmask(f)
    r = cv2.morphologyEx(r, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    keep, tips = jaws(r, frames[0].shape[0])
    out = {"t": time.strftime("%H:%M:%SZ", time.gmtime()), "jaw_tips": tips, "ref": REF}
    if len(tips) != 2:
        out["error"] = f"found {len(tips)} jaws"
        print(json.dumps(out))
        return 4
    d = [float(np.hypot(tips[i][0] - REF[i][0], tips[i][1] - REF[i][1])) for i in range(2)]
    gx = (tips[0][0] + tips[1][0]) / 2
    out.update({"shift_px": [round(x, 1) for x in d], "gap_centre_x": round(gx, 1), "target_px": [round(gx + 3, 1), 268.0]})
    mask = cv2.dilate(keep, np.ones((17, 17), np.uint8))
    vis = frames[0].copy()
    vis[mask > 0] = (0, 255, 0)
    cv2.circle(vis, (int(gx + 3), 268), 8, (255, 0, 255), 2)
    os.makedirs(os.path.join(ga.LOG_ROOT, "depth", "servo"), exist_ok=True)
    cv2.imwrite(os.path.join(ga.LOG_ROOT, "depth", "servo", "jaw_check.jpg"), vis)
    if a.no_move:  # check only: never rewrite the mask/target from a single frame
        out["check_only"] = True
        out["ok"] = d[0] <= a.tol
        print(json.dumps(out))
        return 0 if out["ok"] else 3
    if d[0] > a.tol:
        out["error"] = f"fixed jaw moved {d[0]:.0f}px from the reference: check/tighten the jaw"
        print(json.dumps(out))
        return 3
    cv2.imwrite(f"{T}/wrist_jaw_mask.png", mask * 255)
    json.dump(out, open(f"{T}/jaw_calib.json", "w"), indent=1)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
