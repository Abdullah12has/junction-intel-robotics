"""Scratch tool of builder E1: labelled contact sheet of fixture images, for visual checks.
usage: montage.py <glob under fixtures, e.g. hover/*> <output png> [columns] [tile width]"""
import glob
import os
import sys

import cv2
import numpy as np

FIX = "/home/intel/intel-junction-2026/app/tools/tests/fixtures"


def sheet(paths, out, cols=2, tw=400):
    th = int(tw * 0.75)
    tiles = []
    for p in paths:
        img = cv2.imread(p)
        t = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.rectangle(t, (0, 0), (tw, 20), (0, 0, 0), -1)
        cv2.putText(t, os.path.basename(p)[:52], (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        tiles.append(t)
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    cv2.imwrite(out, np.vstack(rows))
    return out


if __name__ == "__main__":
    paths = sorted(glob.glob(os.path.join(FIX, sys.argv[1])))
    cols = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    tw = int(sys.argv[4]) if len(sys.argv) > 4 else 400
    print(sheet(paths, sys.argv[2], cols=cols, tw=tw), len(paths), "tiles")
