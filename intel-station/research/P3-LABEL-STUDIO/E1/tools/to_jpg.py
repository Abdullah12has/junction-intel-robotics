#!/usr/bin/env python3
"""Convert contact sheets (PNG) to JPEG copies for faster viewing; the PNG originals are kept."""
import glob
import os

from PIL import Image

WD = "/home/intel/intel-junction-2026/work/P3-LABEL-STUDIO/E1"
SRC = sorted(glob.glob(os.path.join(WD, "sheets", "*.png")))
for p in SRC:
    out = p[:-4] + ".jpg"
    im = Image.open(p).convert("RGB")
    im.save(out, "JPEG", quality=88, optimize=True)
    print(os.path.basename(out), im.size, os.path.getsize(out))
