"""Opus quick look at the continuous wrist recording: sample 1 fps, keep active moments, write a contact sheet."""
import glob
import os
import sys

import cv2
import numpy as np

RAW = "/home/intel/intel-junction-2026/data/wrist/raw"
OUT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 48
segs = sorted(glob.glob(os.path.join(RAW, "w_*.mkv")))[:-1]
samples = []
prev = None
for seg in segs:
    cap = cv2.VideoCapture(seg)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % int(round(fps)) == 0:
            small = cv2.resize(fr, (160, 120))
            diff = float(np.mean(cv2.absdiff(small, prev))) if prev is not None else 0.0
            prev = small
            samples.append((os.path.basename(seg), i / fps, diff, fr))
        i += 1
    cap.release()
active = [s for s in samples if s[2] > 6.0]
print("segments", len(segs), "samples", len(samples), "active", len(active))
pick = active if len(active) <= N else [active[int(k * len(active) / N)] for k in range(N)]
tiles = []
for name, t, diff, fr in pick:
    im = cv2.resize(fr, (240, 180))
    cv2.putText(im, f"{name[2:17]} +{t:.0f}s", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
    tiles.append(im)
while len(tiles) % 8:
    tiles.append(np.zeros((180, 240, 3), np.uint8))
rows = [np.hstack(tiles[k:k + 8]) for k in range(0, len(tiles), 8)]
cv2.imwrite(OUT, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
print("wrote", OUT, len(pick))
