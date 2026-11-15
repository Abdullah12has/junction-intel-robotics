#!/usr/bin/env python3
"""P2-B-ANOM-REAL phase A (OD-9): label contact sheets and per-camera spot statistics.

Run isolated: python3 -I spot_sheets.py
Reads R/data/anomaly/real/spot/samples.csv and crops; writes
  R/data/anomaly/real/label_sheets/sheet_01.jpg       raw crops, up to 24 per sheet, captioned with sample_id
  R/data/anomaly/real/label_sheets/sheet_02_colour.jpg  raw | gray-world | mat-normalised per sample
  R/data/anomaly/real/spot/spot_stats.json            per-camera spot centre and bbox statistics
  R/data/anomaly/real/label_sheets/labels_template.tsv  sample_id, label (blank), suggested
"""
import csv
import json
import os

import cv2
import numpy as np

BASE = "/home/intel/intel-junction-2026/data/anomaly/real"
SPOT = f"{BASE}/spot"
SHEETS = f"{BASE}/label_sheets"
CAP_H = 30
T = 256


def load_rows():
    with open(os.path.join(SPOT, "samples.csv")) as fh:
        return [r for r in csv.DictReader(fh) if r.get("usable") == "1"]


def caption(img, text):
    band = np.zeros((CAP_H, img.shape[1], 3), np.uint8)
    cv2.putText(band, text, (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
    return np.vstack([band, img])


def sheet_raw(rows):
    per, n_sheets = 24, 0
    for s in range(0, len(rows), per):
        chunk = rows[s:s + per]
        n_sheets += 1
        cols = 6
        rws = (len(chunk) + cols - 1) // cols
        canvas = np.zeros((rws * (T + CAP_H), cols * T, 3), np.uint8)
        for k, r in enumerate(chunk):
            img = cv2.imread(os.path.join(SPOT, "crops", "raw", f"{r['sample_id']}.png"))
            tile = caption(img, r["sample_id"])
            rr, cc = divmod(k, cols)
            canvas[rr * (T + CAP_H):(rr + 1) * (T + CAP_H), cc * T:(cc + 1) * T] = tile
        cv2.imwrite(os.path.join(SHEETS, f"sheet_{n_sheets:02d}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return n_sheets


def sheet_colour(rows):
    canvas_rows = []
    for r in rows:
        tiles = []
        for var, lab in (("raw", "raw"), ("gw", "gray-world"), ("mat", "mat-normalised")):
            img = cv2.imread(os.path.join(SPOT, "crops", var, f"{r['sample_id']}.png"))
            tiles.append(caption(img, f"{r['sample_id']} {lab}"))
        canvas_rows.append(np.hstack(tiles))
    cv2.imwrite(os.path.join(SHEETS, "sheet_02_colour.jpg"), np.vstack(canvas_rows), [cv2.IMWRITE_JPEG_QUALITY, 92])


def stats(rows):
    out = {}
    for cam in ("close", "top"):
        sub = [r for r in rows if r["camera"] == cam]
        if not sub:
            out[cam] = {"n": 0}
            continue
        cx = np.array([float(r["cx"]) for r in sub])
        cy = np.array([float(r["cy"]) for r in sub])
        bw = np.array([float(r["bbox_w"]) for r in sub])
        bh = np.array([float(r["bbox_h"]) for r in sub])
        out[cam] = {"n": len(sub), "cx_mean": round(float(cx.mean()), 1), "cx_std": round(float(cx.std()), 1),
                    "cy_mean": round(float(cy.mean()), 1), "cy_std": round(float(cy.std()), 1),
                    "bbox_w_median": float(np.median(bw)), "bbox_h_median": float(np.median(bh)),
                    "episodes": sorted({int(r["episode"]) for r in sub})}
    return out


def main():
    os.makedirs(SHEETS, exist_ok=True)
    rows = sorted(load_rows(), key=lambda r: r["sample_id"])
    n = sheet_raw(rows)
    sheet_colour(rows)
    st = stats(rows)
    with open(os.path.join(SPOT, "spot_stats.json"), "w") as fh:
        json.dump(st, fh, indent=1)
    with open(os.path.join(SHEETS, "labels_template.tsv"), "w", newline="") as fh:
        fh.write("sample_id\tlabel\tsuggested\n")
        for r in rows:
            fh.write(f"{r['sample_id']}\t\t{r['suggested']}\n")
    print("sheets", n, "samples", len(rows))
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
