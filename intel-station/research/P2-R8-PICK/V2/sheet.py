"""Verifier V2: labelled contact sheet of top-view frames. Usage: python3 -I sheet.py OUT.png KIND
KIND is start, grasp or end; reads episodes.json in this folder."""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT, KIND = sys.argv[1], sys.argv[2]
eps = json.load(open(os.path.join(HERE, "episodes.json")))
cell_w, cell_h, cols = 320, 240, 4
keys = sorted(eps, key=lambda k: int(k))
rows = (len(keys) + cols - 1) // cols
sheet = np.full((rows * cell_h, cols * cell_w, 3), 40, np.uint8)
for i, k in enumerate(keys):
    p = os.path.join(HERE, eps[k].get("frame_" + KIND, ""))
    img = cv2.imread(p) if p and os.path.exists(p) else None
    r, c = divmod(i, cols)
    if img is None:
        img = np.zeros((cell_h, cell_w, 3), np.uint8)
    img = cv2.resize(img, (cell_w, cell_h))
    cv2.putText(img, "ep%s %s" % (k, KIND), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    sheet[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = img
cv2.imwrite(OUT, sheet)
print("WROTE", OUT, sheet.shape)
