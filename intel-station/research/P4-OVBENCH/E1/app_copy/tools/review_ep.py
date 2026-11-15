#!/usr/bin/env python3
"""Opus I-60: one contact sheet per episode for visual review. usage: review_ep.py EPDIR [OUT.jpg]
Rows: phases (pre, yaw/centre, stage1/2, grasp_open, closed, lifted, released); columns: top, close, wrist.
Held multi-view close crops are appended as a last row. Title row: decision and timings from episode.json."""
import glob
import json
import os
import sys

import cv2
import numpy as np

ep = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ep, "review.jpg")
W, Hh = 320, 240
rows = []
phases = ["pre", "yaw0", "centre0", "stage1", "stage2", "grasp_open", "closed", "lifted", "released"]


def tile(p, label):
    im = cv2.imread(p) if p and os.path.exists(p) else None
    im = np.zeros((Hh, W, 3), np.uint8) if im is None else cv2.resize(im, (W, Hh))
    cv2.putText(im, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return im


for ph in phases:
    if not glob.glob(os.path.join(ep, f"{ph}_*.jpg")):
        continue
    rows.append(np.hstack([tile(os.path.join(ep, f"{ph}_{v}.jpg"), f"{ph} {v}") for v in ("top", "close", "wrist")]))
held = sorted(glob.glob(os.path.join(ep, "crops", "held_close_r*_full.jpg")))
if held:
    t = [tile(p, os.path.basename(p)[:-9]) for p in held[:3]]
    while len(t) < 3:
        t.append(np.zeros((Hh, W, 3), np.uint8))
    rows.append(np.hstack(t))
try:
    e = json.load(open(os.path.join(ep, "episode.json")))
    title = (f"ep {e.get('episode')} {e.get('colour')} {e.get('shape')} -> {e.get('label')} {e.get('plate')} "
             f"({e.get('decision_conf')}) ok={e.get('success')} {str(e.get('failure') or '')[:60]}")
    title2 = f"{str(e.get('decision_reasons'))[:110]}  t={e.get('timings')}"
except Exception:  # noqa: BLE001
    title, title2 = os.path.basename(ep), ""
head = np.zeros((44, 3 * W, 3), np.uint8)
cv2.putText(head, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.putText(head, title2[:150], (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
sheet = np.vstack([head] + rows) if rows else head
cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
print(out, sheet.shape)
