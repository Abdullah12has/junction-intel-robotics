import glob
import json
import os
import sys

import cv2
import numpy as np

folder = sys.argv[1]


def centred(hue):
    return ((hue.astype(np.int16) + 90) % 180) - 90


rows = []
for path in sorted(glob.glob(os.path.join(folder, "*.png"))):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        rows.append({"file": os.path.basename(path), "error": "unreadable"})
        continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    lum = rgb.astype(np.float32).mean(axis=2)
    thr = np.percentile(lum, 30)
    mat = rgb[lum <= thr].astype(np.float32)
    mat_med = np.median(mat, axis=0)
    gains = mat_med.mean() / np.maximum(mat_med, 1e-6)
    roi = np.ascontiguousarray(rgb[140:480, 120:520])
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    hc = centred(hsv[..., 0])
    r = roi[..., 0].astype(np.int32)
    g = roi[..., 1].astype(np.int32)
    b = roi[..., 2].astype(np.int32)
    jaw = (r > 90) & (r > 1.8 * g) & (r > 1.8 * b)
    skin = (hc >= 3) & (hc <= 20) & (hsv[..., 1] >= 40) & (hsv[..., 1] < 200) & (hsv[..., 2] >= 110)
    row = {
        "file": os.path.basename(path),
        "frame_mean_rgb": [round(float(x), 1) for x in rgb.reshape(-1, 3).mean(axis=0)],
        "mat_rgb_median": [round(float(x), 1) for x in mat_med],
        "mat_gains_to_neutral": [round(float(x), 3) for x in gains],
        "jaw_pixels_roi": int(jaw.sum()),
        "skin_like_pixels_roi": int(skin.sum()),
    }
    if jaw.sum() > 200:
        row["jaw_rgb_median"] = [round(float(x), 1) for x in np.median(roi[jaw], axis=0)]
        row["jaw_hue_centred_median"] = round(float(np.median(hc[jaw])), 2)
        row["jaw_sat_median"] = round(float(np.median(hsv[..., 1][jaw])), 1)
    rows.append(row)

print(json.dumps({"folder": folder, "n": len(rows), "frames": rows}, indent=1))
