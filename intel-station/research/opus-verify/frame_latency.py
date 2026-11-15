"""Measure how long after a move completes the live wrist JPEG shows the new view: roll the wrist by 15 deg, then
read run/live_wrist.jpg every 50 ms (by mtime) for 2 s and compare each with the settled view taken at the end.
Prints per frame: write time after the move returned, mean abs difference to the settled view."""
import os
import time

import cv2
import numpy as np

import grasp_auto as ga

P = f"{ga.RUN_ROOT}/live_wrist.jpg"
st = ga.state()
q = list(st["target"])
res = []
for sign in (1, -1):
    qq = list(q)
    qq[4] = round(q[4] + sign * 15.0, 2)
    ga.move(qq, speed=15, comp=False)
    t_done = time.time()
    seen, frames = set(), []
    while time.time() - t_done < 2.0:
        try:
            m = os.path.getmtime(P)
        except OSError:
            m = None
        if m and m not in seen:
            seen.add(m)
            img = cv2.imread(P)
            if img is not None:
                frames.append((m - t_done, img))
        time.sleep(0.02)
    settled = frames[-1][1].astype(np.int16)
    for dt, img in frames:
        res.append((sign, round(dt, 3), round(float(np.abs(img.astype(np.int16) - settled).mean()), 2)))
    q = qq
ga.move(list(st["target"]), speed=15, comp=True)
for r in res:
    print(r)
