"""Draw a reproducible random sample of top-view bricks (seeded) at 4x zoom for strict shape checks.

Writes sheets/zoom_top_shape_check.jpg and labels/top_shape_check_sample.tsv (brick_id, shape_auto, aspect, area).
usage: python3 -I zoom_top_sample.py
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TOP, WORK  # noqa: E402

d = json.load(open(os.path.join(WORK, "seg", "bricks.json")))
samples = json.load(open(os.path.join(WORK, "samples", "samples.json")))
tops = [b for b in d["bricks"] if b["view"] == "top"]
rng = np.random.default_rng(1)
pick = []
for shape, n in (("2x2", 6), ("not_2x2", 6), ("unknown", 4)):
    pool = [b for b in tops if b["shape"] == shape]
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    pick += [pool[i] for i in sorted(idx)]
tiles, rows = [], []
for b in pick:
    img = cv2.imread(samples[b["sample"]]["views"][TOP]["png"])
    x, y, w, h = b["bbox"]
    pad = 8
    crop = img[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
    s = 200 / max(crop.shape[:2])
    crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((200 + 30, 200, 3), np.uint8)
    canvas[:crop.shape[0], :crop.shape[1]] = crop
    cv2.putText(canvas, f"{b['sample'][9:]} k{b['k']}", (3, 212), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                cv2.LINE_AA)
    cv2.putText(canvas, f"{b['shape']} a{b['aspect']:.2f} A{b['area']}", (3, 226), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(canvas)
    rows.append([b["id"].replace(b["view"], "top"), b["shape"], str(b["aspect"]), str(b["area"]), b["colour"]])
cols = 8
rws = (len(tiles) + cols - 1) // cols
grid = np.zeros((rws * 230, cols * 200, 3), np.uint8)
for i, t in enumerate(tiles):
    r, c = divmod(i, cols)
    grid[r * 230:r * 230 + t.shape[0], c * 200:(c + 1) * 200] = t
os.makedirs(os.path.join(WORK, "labels"), exist_ok=True)
cv2.imwrite(os.path.join(WORK, "sheets", "zoom_top_shape_check.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
with open(os.path.join(WORK, "labels", "top_shape_check_sample.tsv"), "w") as f:
    f.write("brick_id\tshape_auto\taspect\tarea\tcolour_auto\n")
    for r in rows:
        f.write("\t".join(r) + "\n")
print("wrote", len(tiles), "tiles")
