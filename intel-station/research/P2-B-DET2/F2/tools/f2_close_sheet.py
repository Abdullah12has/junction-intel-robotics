"""Contact sheet of the raw close-view CONFIDENT_DEFECT calls of a run, with tape score (crops from the PNG; analysis only)."""
import json
import sys

import cv2
import numpy as np

FINAL = json.load(open(sys.argv[1]))
SAMP = json.load(open(sys.argv[2]))
OUT = sys.argv[3]
CLOSE = "realsense 243722064316"
tiles = []
for key, res in sorted(FINAL["frames"].items()):
    for b in res.get("raw", {}).get("bricks", []):
        if b["view"] != "close" or b["verdict"] != "CONFIDENT_DEFECT":
            continue
        img = cv2.imread(SAMP[key]["views"][CLOSE]["png"])
        cx, cy = b["centre_px"]
        x0, y0 = max(0, int(cx) - 70), max(0, int(cy) - 70)
        crop = cv2.resize(img[y0:y0 + 140, x0:x0 + 140], (200, 200), interpolation=cv2.INTER_AREA)
        cv2.putText(crop, key[9:] + " " + b["colour"], (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(crop, "tape %.3f" % b["tape_score"], (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        tiles.append(crop)
sheet = np.hstack(tiles) if tiles else np.zeros((200, 200, 3), np.uint8)
cv2.imwrite(OUT, sheet)
print(len(tiles), sheet.shape)
