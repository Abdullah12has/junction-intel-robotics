"""Zoom sheet of the close bricks where Opus and E2 labels disagree, plus the e06_start brick (Opus k1 unmatched).

Crops come from the close frames samples/<sample>_243722064316.png; boxes come from seg/bricks.json.
Writes sheets/zoom_disputed_v2b.jpg and prints the bbox and area of each brick for the log.
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

TILE = 320
IDS = [
    ("e03_end k0: Opus GOOD, E2 X", "6d6bdce3_e03_end_243722064316_0"),
    ("e06_end k0: Opus GOOD, E2 X", "6d6bdce3_e06_end_243722064316_0"),
    ("e09_end k1: Opus GOOD, E2 X", "6d6bdce3_e09_end_243722064316_1"),
    ("e10_end k1: Opus GOOD, E2 X", "6d6bdce3_e10_end_243722064316_1"),
    ("e10_end k0: Opus DEFECT_NOT2X2, E2 X", "6d6bdce3_e10_end_243722064316_0"),
    ("e06_start k0: E2 G, Opus k1 unmatched", "6d6bdce3_e06_start_243722064316_0"),
]


def main():
    work = C.WORK
    seg = json.load(open(os.path.join(work, "seg", "bricks.json")))
    by_id = {b["id"]: b for b in seg["bricks"]}
    tiles = []
    for label, bid in IDS:
        b = by_id.get(bid)
        if b is None:
            print("MISSING", bid)
            continue
        parts = bid.split("_")
        sample, serial = "_".join(parts[:3]), parts[3]
        img = cv2.imread(os.path.join(work, "samples", f"{sample}_{serial}.png"))
        H, W = img.shape[:2]
        x, y, w, h = b["bbox"]
        print(bid, "bbox", b["bbox"], "area", b["area"], "aspect", b["aspect"], "colour", b["colour"],
              "merged", b.get("merged"), b.get("merged_reason"), "tape_frac", b.get("tape_score"))
        pad = 30
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
        crop = img[y0:y1, x0:x1].copy()
        s = min(TILE / crop.shape[1], TILE / crop.shape[0])
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(crop, (int((x - x0) * s), int((y - y0) * s)),
                      (int((x + w - x0) * s), int((y + h - y0) * s)), (0, 255, 255), 1)
        tile = np.zeros((TILE, TILE, 3), np.uint8)
        oy, ox = (TILE - crop.shape[0]) // 2, (TILE - crop.shape[1]) // 2
        tile[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
        cv2.rectangle(tile, (0, 0), (TILE - 1, 22), (0, 0, 0), -1)
        cv2.putText(tile, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    # e06_start: list all close bricks of that sample so the Opus k1 numbering can be checked
    print("e06_start bricks in seg:", [(b["id"], b["bbox"], b["colour"]) for b in seg["bricks"]
                                       if b["sample"] == "6d6bdce3_e06_start" and b["view"] == "close"])
    while len(tiles) % 3:
        tiles.append(np.zeros((TILE, TILE, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    sheet = np.vstack(rows)
    out = os.path.join(work, "sheets", "zoom_disputed_v2b.jpg")
    cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", out, sheet.shape)


if __name__ == "__main__":
    main()
