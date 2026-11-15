"""Zoom sheet of selected close-view bricks at 256 px for strict visual label checks (writes one JPEG).

usage: python3 -I zoom.py <out.jpg> <sample_id>:<k> [<sample_id>:<k> ...]
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLOSE, WORK  # noqa: E402

bricks = json.load(open(os.path.join(WORK, "seg", "bricks.json")))["bricks"]
samples = json.load(open(os.path.join(WORK, "samples", "samples.json")))
out = sys.argv[1]
want = [a for a in sys.argv[2:]]
tiles = []
for w in want:
    sid, k = w.rsplit(":", 1)
    b = [x for x in bricks if x["view"] == "close" and x["sample"] == sid and x["k"] == int(k)]
    if not b:
        print("missing", w)
        continue
    b = b[0]
    img = cv2.imread(samples[sid]["views"][CLOSE]["png"])
    x, y, w_, h_ = b["bbox"]
    pad = 12
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img.shape[1], x + w_ + pad), min(img.shape[0], y + h_ + pad)
    crop = img[y0:y1, x0:x1]
    s = 256 / max(crop.shape[:2])
    crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((256 + 30, 256, 3), np.uint8)
    canvas[:crop.shape[0], :crop.shape[1]] = crop
    cv2.putText(canvas, f"{sid[9:]} k{k}", (3, 256 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                cv2.LINE_AA)
    cv2.putText(canvas, f"{b['colour']} a{b['aspect']:.2f} off{b['tape_features']['off_frac']:.2f}",
                (3, 256 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(canvas)
cols = 4
rows = (len(tiles) + cols - 1) // cols
grid = np.zeros((rows * 286, cols * 256, 3), np.uint8)
for i, t in enumerate(tiles):
    r, c = divmod(i, cols)
    grid[r * 286:r * 286 + t.shape[0], c * 256:(c + 1) * 256] = t
cv2.imwrite(out, grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
print("wrote", out, len(tiles))
