#!/usr/bin/env python3
"""P2-B-ANOM-REAL phase A2 (OD-11): contact sheets for held-brick samples.

Run isolated: python3 -I held_sheets.py
Reads R/data/anomaly/real/held/samples_held.csv and crops/raw; writes
  R/data/anomaly/real/label_sheets/held_sheet_NN.jpg   24 raw crops per sheet, captioned with sample_id
  R/data/anomaly/real/label_sheets/held_labels_template.tsv   sample_id, label (blank), suggested
  R/data/anomaly/real/held/held_stats.json   counts per camera and per episode
Sheets are rebuilt from the full CSV each run, so ids and sheet order stay sorted.
"""
import csv
import json
import os
from collections import Counter

import cv2
import numpy as np

BASE = "/home/intel/intel-junction-2026/data/anomaly/real"
HELD = f"{BASE}/held"
SHEETS = f"{BASE}/label_sheets"
CAP_H = 30
T = 256
PER = 24
COLS = 6


def caption(img, text):
    band = np.zeros((CAP_H, img.shape[1], 3), np.uint8)
    cv2.putText(band, text, (3, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return np.vstack([band, img])


def main():
    os.makedirs(SHEETS, exist_ok=True)
    rows = sorted(csv.DictReader(open(f"{HELD}/samples_held.csv")), key=lambda r: r["sample_id"])
    for old in os.listdir(SHEETS):
        if old.startswith("held_sheet_"):
            os.remove(os.path.join(SHEETS, old))
    n_sheets = 0
    for s in range(0, len(rows), PER):
        chunk = rows[s:s + PER]
        n_sheets += 1
        rws = (len(chunk) + COLS - 1) // COLS
        canvas = np.zeros((rws * (T + CAP_H), COLS * T, 3), np.uint8)
        for k, r in enumerate(chunk):
            img = cv2.imread(os.path.join(HELD, "crops", "raw", f"{r['sample_id']}.png"))
            rr, cc = divmod(k, COLS)
            canvas[rr * (T + CAP_H):(rr + 1) * (T + CAP_H), cc * T:(cc + 1) * T] = caption(img, r["sample_id"])
        cv2.imwrite(os.path.join(SHEETS, f"held_sheet_{n_sheets:02d}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(f"{SHEETS}/held_labels_template.tsv", "w", newline="") as fh:
        fh.write("sample_id\tlabel\tsuggested\n")
        for r in rows:
            sugg = r["suggested"] + (" TAPE?" if r.get("flag") == "tape?" else "")
            fh.write(f"{r['sample_id']}\t\t{sugg}\n")
    per_cam = Counter(r["serial"] for r in rows)
    per_ep = Counter(f"{r['dataset'][:8]}_e{int(r['episode']):02d}" for r in rows)
    stats = {"samples": len(rows), "sheets": n_sheets, "per_camera": dict(per_cam), "per_episode": dict(per_ep),
             "suggested": dict(Counter(r["suggested"] for r in rows)), "tape_flagged": sum(1 for r in rows if r.get("flag") == "tape?")}
    json.dump(stats, open(f"{HELD}/held_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
