"""Opus check of F-1: which red pixels the 15 px bridge joins to the brick in servo_180002_red.png."""
import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/intel/intel-junction-2026/app/tools")
import grasp_auto as ga  # noqa: E402
import pick_any as pa  # noqa: E402

img = cv2.imread("/home/intel/intel-junction-2026/app/tools/tests/fixtures/servo/servo_180002_red.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
fg = pa.COLOUR_CLASSES["red"](H, S, V) & ~ga.JAW_MASK
fg8 = cv2.morphologyEx(fg.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
fg8 = cv2.morphologyEx(fg8, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
n0, lab0, st0, cen0 = cv2.connectedComponentsWithStats(fg8)
print("no bridge components (area >= 200):")
for i in range(1, n0):
    if st0[i, 4] >= 200:
        print("  ", i, "area", int(st0[i, 4]), "centre", cen0[i].round(1).tolist(), "bbox", st0[i, :4].tolist())
bridge = cv2.dilate(fg8, np.ones((15, 15), np.uint8))
n, lab, st, cen = cv2.connectedComponentsWithStats(bridge)
big = 1 + int(np.argmax(st[1:, 4]))
m = (lab == big) & (fg8 > 0)
vis = img.copy()
for i in range(1, n0):
    if st0[i, 4] < 200:
        continue
    comp = lab0 == i
    col = (0, 255, 0) if (comp & m).any() else (128, 128, 128)
    vis[comp] = (0.4 * vis[comp] + 0.6 * np.array(col)).astype(np.uint8)
vis[ga.JAW_MASK] = (vis[ga.JAW_MASK] * 0.3).astype(np.uint8)
cv2.imwrite("/home/intel/intel-junction-2026/work/opus-verify/f1_mask.jpg", vis)
print("bridged blob area", int(m.sum()))
