"""Contact sheets of every hand-watch estop image with the current skin mask overlaid (Opus I-61 offline work)."""
import glob
import os
import time

import cv2
import numpy as np

R = "/home/intel/intel-junction-2026"
OUT = f"{R}/work/handdet"
ROI = {"top": (175, 30, 640, 345), "close": (0, 0, 640, 480), "wrist": (0, 0, 640, 480)}


def skin(img):
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((ycc[..., 1] >= 138) & (ycc[..., 1] <= 170) & (ycc[..., 2] >= 90) & (ycc[..., 2] <= 125)
         & (h >= 5) & (h <= 20) & (s >= 45) & (s <= 150) & (v >= 90)).astype(np.uint8) * 255
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


rows = []
for view in ("top", "close", "wrist"):
    files = sorted(glob.glob(f"{R}/logs/monitor/estop_{view}_*.jpg"))
    tiles = []
    for i, f in enumerate(files):
        img = cv2.imread(f)
        if img is None:
            continue
        m = skin(img)
        x0, y0, x1, y1 = ROI[view]
        roi = np.zeros_like(m)
        roi[y0:y1, x0:x1] = m[y0:y1, x0:x1]
        n, lab, st, _ = cv2.connectedComponentsWithStats(roi)
        big = int(st[1:, 4].max()) if n > 1 else 0
        ov = img.copy()
        ov[roi > 0] = (0.4 * ov[roi > 0] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
        cv2.rectangle(ov, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 0), 2)
        ep = int(os.path.basename(f).split("_")[-1].split(".")[0])
        t = time.strftime("%H:%M:%S", time.gmtime(ep))
        tile = cv2.resize(ov, (320, 240))
        cv2.putText(tile, f"#{i} {view} {t}Z {big}px", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        tiles.append(tile)
        rows.append((view, i, os.path.basename(f), t, big))
    for k in range(0, len(tiles), 9):
        chunk = tiles[k:k + 9]
        while len(chunk) % 3:
            chunk.append(np.zeros((240, 320, 3), np.uint8))
        sheet = np.vstack([np.hstack(chunk[j:j + 3]) for j in range(0, len(chunk), 3)])
        cv2.imwrite(f"{OUT}/sheet_{view}_{k // 9}.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
with open(f"{OUT}/estops.tsv", "w") as fh:
    fh.write("view\tidx\tfile\tutc\tmax_blob_px\n")
    for r in rows:
        fh.write("\t".join(map(str, r)) + "\n")
print(len(rows), "images;", sorted(glob.glob(f"{OUT}/sheet_*.jpg")))
