#!/usr/bin/env python3
"""P3-TWIN-CAMS E1 step 8: HSV check of red jaw pixels versus other red-ish objects in the peek frames (read-only).

Prints the median H, S, V over a 5x5 patch at chosen pixels, to set per-camera saturation thresholds.
Run: envs/data-env/bin/python -I tools/s8_hsv.py
"""
import cv2
import numpy as np

E1 = "/home/intel/intel-junction-2026/work/P3-TWIN-CAMS/E1"
PTS = {
    "close_2": {"jaw_left_body": (375, 100), "jaw_right_body": (528, 60), "pink_brick": (320, 95)},
    "top_1": {"jaw_left": (362, 150), "jaw_right": (446, 165), "orange_mount": (405, 170)},
}


def patch(hsv, x, y):
    p = hsv[max(0, y - 2):y + 3, max(0, x - 2):x + 3].reshape(-1, 3)
    return [int(np.median(p[:, k])) for k in range(3)]


def main():
    for name, pts in PTS.items():
        img = cv2.imread(f"{E1}/frames/peek_{name.split("_")[0]}_{name.split("_")[1]}.jpg")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for label, (x, y) in pts.items():
            print(name, label, "HSV_median", patch(hsv, x, y))


if __name__ == "__main__":
    main()
