"""Raw versus JPEG q80 crops of the JPEG-only close CONFIDENT_DEFECT calls (analysis only)."""
import json
import sys

import cv2
import numpy as np

R = json.load(open(sys.argv[1]))
S = json.load(open(sys.argv[2]))
OUT = sys.argv[3]
CLOSE = "realsense 243722064316"


def q80(x):
    ok, b = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return cv2.imdecode(b, cv2.IMREAD_COLOR)


tiles = []
for key, bid in [("6d6bdce3_e11_start", "b07_close"), ("6d6bdce3_e13_start", "b14_close")]:
    img = cv2.imread(S[key]["views"][CLOSE]["png"])
    jb = [b for b in R["frames"][key]["jpg"]["bricks"] if b["id"] == bid][0]
    cx, cy = jb["centre_px"]
    x0, y0 = max(0, int(cx) - 70), max(0, int(cy) - 70)
    for name, src in (("raw", img), ("jpg80", q80(img))):
        crop = cv2.resize(src[y0:y0 + 140, x0:x0 + 140], (200, 200), interpolation=cv2.INTER_AREA)
        cv2.putText(crop, key[9:] + " " + name, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(crop, "%s jpg tape %s" % (jb["colour"], jb["tape_score"]), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        tiles.append(crop)
cv2.imwrite(OUT, np.hstack(tiles))
print(len(tiles))
