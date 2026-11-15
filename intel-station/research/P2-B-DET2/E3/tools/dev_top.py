"""P2-B-DET2 E3 dev: top-view background subtraction + blob table + annotated grid. Run with data-env python -I -B."""
import json
import sys

import cv2
import numpy as np

W = "/home/intel/intel-junction-2026/work/P2-B-DET2/E3"
THR = float(sys.argv[1]) if len(sys.argv) > 1 else 22.0
MINA = 250


def lab(b):
    return cv2.cvtColor(b, cv2.COLOR_BGR2LAB).astype(np.float32)


def main():
    idx = json.load(open(f"{W}/out/samples_index.json"))
    recs = [r for r in idx if r["ds"] == "6d6bdce3" and r["cam"] == "top"]
    ims = [cv2.imread(r["png"]) for r in recs]
    bg = np.median(np.stack([lab(i) for i in ims], 0), axis=0).astype(np.float32)
    cv2.imwrite(f"{W}/out/dev_top_bg.png", cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_LAB2BGR))
    tiles = []
    for r, im in zip(recs, ims):
        d = np.linalg.norm(lab(im) - bg, axis=2)
        m = (d > THR).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lbl, st, cen = cv2.connectedComponentsWithStats(m, 8)
        vis = im.copy()
        rows = []
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if a < MINA:
                continue
            comp = (lbl == i).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = max(cnts, key=cv2.contourArea)
            (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
            long_, short_ = max(rw, rh), max(min(rw, rh), 1e-6)
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            px = hsv[comp.astype(bool)]
            med = np.median(px, axis=0)
            k = len(rows)
            rows.append(f"{k} a={a} rect={rw:.0f}x{rh:.0f} asp={long_/short_:.2f} hsv=({med[0]:.0f},{med[1]:.0f},{med[2]:.0f}) xy=({x},{y})")
            box = cv2.boxPoints(((cx, cy), (rw, rh), ang)).astype(np.int32)
            cv2.drawContours(vis, [box], 0, (0, 255, 0), 1)
            cv2.putText(vis, str(k), (int(cx) - 6, int(cy) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        print(r["id"], "THR", THR, "n", len(rows))
        for row in rows:
            print("   ", row)
        cv2.putText(vis, r["id"][-22:], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        tiles.append(cv2.resize(vis, (320, 240)))
    cols = 4
    rows_n = (len(tiles) + cols - 1) // cols
    canvas = np.zeros((rows_n * 240, cols * 320, 3), np.uint8)
    for i, t in enumerate(tiles):
        canvas[(i // cols) * 240:(i // cols + 1) * 240, (i % cols) * 320:(i % cols + 1) * 320] = t
    cv2.imwrite(f"{W}/out/dev_top_sheet.jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("SHEET", canvas.shape)


if __name__ == "__main__":
    main()
