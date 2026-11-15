"""Opus: show the pieces of each bridged blob (kept = green, dropped = magenta) for one wrist frame and colour."""
import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/intel/intel-junction-2026/app/tools")
import grasp_auto as ga  # noqa: E402
import pick_any as pa  # noqa: E402

path, cname, out = sys.argv[1], sys.argv[2], sys.argv[3]
img = cv2.imread(path)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
fg8 = cv2.morphologyEx((pa.COLOUR_CLASSES[cname](H, S, V) & ~ga.JAW_MASK).astype(np.uint8) * 255, cv2.MORPH_OPEN,
                       np.ones((7, 7), np.uint8))
fg8 = cv2.morphologyEx(fg8, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
n0, lab0, st0, c0 = cv2.connectedComponentsWithStats(fg8)
n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(fg8, np.ones((15, 15), np.uint8)))
vis = img.copy()
for i in range(1, n):
    pieces = np.unique(lab0[lab == i])
    pieces = pieces[pieces > 0]
    if len(pieces) == 0 or st0[pieces, 4].sum() < 700:
        continue
    amax = st0[pieces, 4].max()
    for j in pieces:
        keep = st0[j, 4] >= pa.FRAG_FRAC * amax
        col = np.array((0, 255, 0) if keep else (255, 0, 255))
        vis[lab0 == j] = (0.35 * vis[lab0 == j] + 0.65 * col).astype(np.uint8)
        print("blob", i, "piece", j, "area", int(st0[j, 4]), "centroid", c0[j].round(1).tolist(), "kept" if keep else "DROPPED")
for b in pa.blobs(img, min_area=700):
    if b["colour"] == cname:
        print("blob", b["centre"], "short", b["short"], "area", b["area"])
        cv2.drawMarker(vis, tuple(map(int, b["centre"])), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
cv2.imwrite(out, vis)
