"""Contact sheets of the multi crops (scratch review). Reads manifest.tsv and the crop PNGs; writes PNG sheets only.

One sheet per view and usable episode: status ok and ok_hue_mismatch crops, in manifest order. Tile text: index, label,
scan flag, status, and for close crops the close verdict and hue difference. A red frame marks ok_hue_mismatch crops.
"""
import csv
import os

import cv2
import numpy as np

OUT = "/home/intel/intel-junction-2026/work/P2-B-ANOM3/E1/multi"
SHEETS = os.path.join(OUT, "sheets")
TILE = 256
HEAD = 40
COLS = 6


def tile(path, lines, mismatch):
    img = cv2.imread(path, cv2.IMREAD_COLOR) if path else None
    if img is None:
        img = np.full((TILE, TILE, 3), 128, np.uint8)
    img = cv2.resize(img, (TILE, TILE), interpolation=cv2.INTER_AREA)
    canvas = np.full((TILE + HEAD, TILE, 3), 30, np.uint8)
    canvas[HEAD:, :] = img
    for j, text in enumerate(lines):
        cv2.putText(canvas, text, (4, 15 + 16 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    if mismatch:
        cv2.rectangle(canvas, (1, 1), (TILE - 2, TILE + HEAD - 2), (0, 0, 255), 4)
    return canvas


def sheet(tiles):
    rows = (len(tiles) + COLS - 1) // COLS
    canvas = np.full((rows * (TILE + HEAD), COLS * TILE, 3), 60, np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, COLS)
        canvas[r * (TILE + HEAD):(r + 1) * (TILE + HEAD), c * TILE:(c + 1) * TILE] = t
    return canvas


def main():
    with open(os.path.join(OUT, "manifest.tsv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    os.makedirs(SHEETS, exist_ok=True)
    groups = {}
    for r in rows:
        if r["status"] in ("ok", "ok_hue_mismatch") and r["crop_path"]:
            groups.setdefault((r["view"], r["key"]), []).append(r)
    for (view, key), rs in groups.items():
        tiles = []
        for r in rs:
            mism = r["status"] == "ok_hue_mismatch"
            if view == "wrist_multi":
                l1 = "#%s %s %s" % (r["idx"], r["label"], "scan" if r["scan"] == "1" else "hover")
                l2 = "blob %s h%s d%s" % (r["blob_colour"], r["blob_hue"], r["blob_dist"])
            else:
                l1 = "#%s %s %s" % (r["idx"], r["label"], r["close_verdict"])
                l2 = "hue diff %s %s" % (r["close_hue_diff"] or "-", "MISMATCH" if mism else "ok")
            tiles.append(tile(r["crop_path"], [l1, l2], mism))
        name = "%s__%s.png" % (view, key.replace("/", "_"))
        cv2.imwrite(os.path.join(SHEETS, name), sheet(tiles))
        print("wrote", name, len(tiles), "tiles", flush=True)


main()
