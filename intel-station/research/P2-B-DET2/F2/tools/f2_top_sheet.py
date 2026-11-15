"""Contact sheet of the raw top-view CONFIDENT_DEFECT calls of a run (crops from the PNG frames; analysis only)."""
import json
import sys

import cv2
import numpy as np

FINAL = json.load(open(sys.argv[1]))
SAMP = json.load(open(sys.argv[2]))
OUT = sys.argv[3]
TOP = "realsense 243622061421"
tiles = []
for key, res in sorted(FINAL["frames"].items()):
    for b in res.get("raw", {}).get("bricks", []):
        if b["view"] != "top" or b["verdict"] != "CONFIDENT_DEFECT":
            continue
        img = cv2.imread(SAMP[key]["views"][TOP]["png"])
        cx, cy = b["centre_px"]
        x0, y0 = max(0, int(cx) - 60), max(0, int(cy) - 60)
        crop = cv2.resize(img[y0:y0 + 120, x0:x0 + 120], (240, 240), interpolation=cv2.INTER_NEAREST)
        cv2.putText(crop, key[9:] + " " + b["id"], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(crop)
sheet = np.hstack(tiles) if tiles else np.zeros((240, 240, 3), np.uint8)
cv2.imwrite(OUT, sheet)
print(len(tiles), sheet.shape)
