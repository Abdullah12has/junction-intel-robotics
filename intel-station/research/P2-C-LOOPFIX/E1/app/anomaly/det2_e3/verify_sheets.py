"""Large verification sheets (E3): every brick crop of one view with a big tile index (#i) and an index table.
Labels are assigned by tile index, so they stay valid for the current bricks_build.json order.
Run: envs/data-env/bin/python -I -B app/anomaly/det2_e3/verify_sheets.py"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import det2  # noqa: E402

WORK = "/home/intel/intel-junction-2026/work/P2-B-DET2/E3"


def index_sheet(recs, cal, out, size, cols):
    tiles = []
    for i, r in enumerate(recs):
        im = cv2.imread(r["png"])
        x, y, w, h = r["bbox"]
        p = 6
        cr = im[max(0, y - p):min(im.shape[0], y + h + p), max(0, x - p):min(im.shape[1], x + w + p)]
        cr = cv2.resize(cr, (size, size), interpolation=cv2.INTER_AREA).copy()
        cv2.putText(cr, f"#{i}", (3, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(cr, f"#{i}", (3, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        bar = np.zeros((18, size, 3), np.uint8)
        sh = r.get("shape", "")
        cv2.putText(bar, f"{r['colour'][:5]} {sh}", (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        tiles.append(np.vstack([cr, bar]))
    th, tw = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for i, t in enumerate(tiles):
        canvas[(i // cols) * th:(i // cols + 1) * th, (i % cols) * tw:(i % cols + 1) * tw] = t
    cv2.imwrite(out, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    with open(out.replace(".jpg", "_index.tsv"), "w") as fh:
        fh.write("index\tid\tview\tcolour\tshape\tcentre_px\tarea\ttape_dark_grey\n")
        for i, r in enumerate(recs):
            f = r["tape_feats"]
            fh.write(f"{i}\t{r['id']}\t{r['view']}\t{r['colour']}\t{r.get('shape', '-')}\t{r['centre_px']}\t{r['area']}\t{f['dark_grey']}\n")
    return canvas.shape


def main():
    cal = det2.load_cal()
    allrecs = json.load(open(f"{WORK}/out/bricks_build.json"))
    for view, size, cols in (("close", 160, 10), ("top", 128, 10)):
        rv = [r for r in allrecs if r["view"] == view]
        out = f"{WORK}/sheets/verify_idx_{view}.jpg"
        print(out, index_sheet(rv, cal, out, size, cols), len(rv))


if __name__ == "__main__":
    main()
