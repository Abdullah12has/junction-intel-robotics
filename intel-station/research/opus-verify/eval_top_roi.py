"""Top-view hand ROI: y0 = 30 (current handdet) vs y0 = 0 (what the live watchdog used at 17:38Z, per the logged
areas). Runs both on every frame in work/handdet/frames_top (one frame every 2 s of top_20261114T170351Z.mkv) and
writes contact sheets of the frames where they disagree, for review by eye."""
import glob
import os
import sys

import cv2
import numpy as np

R = "/home/intel/intel-junction-2026"
OUT = f"{R}/work/opus-verify/top_roi"
sys.path.insert(0, f"{R}/app/tools")
import handdet as hd  # noqa: E402

os.makedirs(OUT, exist_ok=True)
files = sorted(glob.glob(f"{R}/work/handdet/frames_top/*.jpg"))
pos = {30: 0, 0: 0}
only0, only30 = [], []
for f in files:
    img = cv2.imread(f)
    if img is None:
        continue
    res = {}
    for y0 in (30, 0):
        hd.ROI["top"] = (175, y0, 640, 345)
        res[y0] = hd.is_hand(img, "top")
        pos[y0] += int(res[y0][0])
    if res[0][0] != res[30][0]:
        t = cv2.resize(img, (320, 240))
        cv2.rectangle(t, (87, 0), (319, 15), (0, 255, 255), 1)
        cv2.putText(t, f"{os.path.basename(f)} y0=0:{res[0][1]['area']}{res[0][1]['edge']} y0=30:{res[30][1]['area']}",
                    (3, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)
        (only0 if res[0][0] else only30).append(t)
print("frames", len(files), "positives y0=30:", pos[30], "y0=0:", pos[0], "only y0=0:", len(only0), "only y0=30:", len(only30))
for name, tiles in (("only_y0_0", only0), ("only_y0_30", only30)):
    for k in range(0, len(tiles), 12):
        chunk = tiles[k:k + 12]
        while len(chunk) % 3:
            chunk.append(np.zeros_like(chunk[0]))
        rows = [np.hstack(chunk[i:i + 3]) for i in range(0, len(chunk), 3)]
        cv2.imwrite(f"{OUT}/{name}_{k // 12}.jpg", np.vstack(rows))
