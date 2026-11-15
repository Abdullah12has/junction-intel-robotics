"""Composite of the live top / close / wrist frames for a quick visual check (Opus monitoring)."""
import sys

import cv2
import numpy as np

import os  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import APP, R  # noqa: E402  (roots from this file's location)
out = sys.argv[1] if len(sys.argv) > 1 else f"{R}/logs/depth/servo/peek.jpg"
tiles = []
for cam in ("top", "close", "wrist"):
    im = cv2.imread(f"{R}/run/live_{cam}.jpg")
    if im is None:
        im = np.zeros((320, 426, 3), np.uint8)
    tiles.append(cv2.resize(im, (426, 320)))
cv2.imwrite(out, np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 80])
print(out)
