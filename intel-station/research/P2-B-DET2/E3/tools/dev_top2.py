"""P2-B-DET2 E3 dev2: mat-hull + static-mask segmentation, top view. Run data-env python -I -B."""
import json
import sys

import cv2
import numpy as np

W = "/home/intel/intel-junction-2026/work/P2-B-DET2/E3"
CAM = sys.argv[1] if len(sys.argv) > 1 else "243622061421"
DS = sys.argv[2] if len(sys.argv) > 2 else "6d6bdce3"
MINA, MAXA = 300, 5000


def mat_hull(hsv):
    dark = (hsv[..., 2] < 70).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(dark, 8)
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    m = (lbl == big).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    hm = np.zeros(m.shape, np.uint8)
    cv2.fillPoly(hm, [hull], 1)
    return hm.astype(bool), float(np.median(hsv[..., 2][m.astype(bool)]))


def cand(hsv):
    return ((hsv[..., 1] > 70) & (hsv[..., 2] > 90)) | (hsv[..., 2] > 150)


def main():
    idx = json.load(open(f"{W}/out/samples_index.json"))
    recs = [r for r in idx if r["ds"] == DS and r["serial"] == CAM]
    ims = [cv2.imread(r["png"]) for r in recs]
    hsvs = [cv2.cvtColor(i, cv2.COLOR_BGR2HSV) for i in ims]
    hulls, vmed = zip(*[mat_hull(h) for h in hsvs])
    freq = np.mean(np.stack([cand(h) & hu for h, hu in zip(hsvs, hulls)], 0), axis=0)
    static = freq >= 0.9
    cv2.imwrite(f"{W}/out/dev2_static_{CAM}.png", (static * 255).astype(np.uint8))
    print("frames", len(recs), "mat V median", [round(v) for v in vmed][:4], "static px", int(static.sum()))
    tiles = []
    for r, im, h, hu in zip(recs, ims, hsvs, hulls):
        fg = (cand(h) & hu & ~static).astype(np.uint8) * 255
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, lbl, st, _ = cv2.connectedComponentsWithStats(fg, 8)
        vis = im.copy()
        rows = []
        for i in range(1, n):
            x, y, w, hh, a = st[i]
            if a < MINA or a > MAXA:
                continue
            comp = (lbl == i).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = max(cnts, key=cv2.contourArea)
            (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
            med = np.median(h[comp.astype(bool)], axis=0)
            k = len(rows)
            rows.append(f"{k} a={a} rect={rw:.0f}x{rh:.0f} asp={max(rw, rh) / max(min(rw, rh), 1e-6):.2f} hsv=({med[0]:.0f},{med[1]:.0f},{med[2]:.0f}) xy=({x},{y})")
            box = cv2.boxPoints(((cx, cy), (rw, rh), ang)).astype(np.int32)
            cv2.drawContours(vis, [box], 0, (0, 255, 0), 1)
            cv2.putText(vis, str(k), (int(cx) - 6, int(cy) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        print(r["id"], "n", len(rows))
        for row in rows:
            print("   ", row)
        cv2.putText(vis, r["id"][-22:], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        tiles.append(cv2.resize(vis, (320, 240)))
    cols = 4
    rn = (len(tiles) + cols - 1) // cols
    canvas = np.zeros((rn * 240, cols * 320, 3), np.uint8)
    for i, t in enumerate(tiles):
        canvas[(i // cols) * 240:(i // cols + 1) * 240, (i % cols) * 320:(i % cols + 1) * 320] = t
    cv2.imwrite(f"{W}/out/dev2_sheet_{CAM}.jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("SHEET", canvas.shape)


if __name__ == "__main__":
    main()
