#!/usr/bin/env python3
"""V1 colour, lighting and ROI check on the recorded frames (read-only inputs; cv2).
Own ROI slices (exclusive end). Writes outputs/colour_check.json, outputs/roi_overlay_<cam>.png and outputs/roi_crops.png."""
import json

import cv2
import numpy as np

G = "/home/intel/intel-junction-2026/logs/depth/g1_162903"
SRC = {"top": G + "/top_color.jpg", "close": G + "/close_color.jpg",
       "pre": "/home/intel/intel-junction-2026/logs/sort/any_1740/ep_0001/pre_top.jpg"}
OUT = "/home/intel/intel-junction-2026/work/P3-TWIN-SCENE/V1/outputs"
# name, image key, x0, y0, x1, y1 (x1, y1 exclusive), statistic, builder value (R, G, B)
ROIS = [
    ("close_blue_2x2_top", "close", 345, 205, 430, 240, "mean", (38.5, 119.2, 162.7)),
    ("close_blue_2x2_dark_tape", "close", 318, 240, 356, 300, "mean", (59.4, 82.5, 104.4)),
    ("close_white_2x2", "close", 520, 330, 600, 380, "mean", (220.3, 219.4, 220.2)),
    ("close_mat_centre", "close", 180, 250, 300, 330, "mean", (40.1, 39.2, 45.3)),
    ("close_mat_lower_left", "close", 0, 300, 200, 480, "mean", (36.1, 44.0, 46.1)),
    ("top_blue_plate_interior", "top", 205, 115, 275, 175, "mean", (109.0, 252.9, 253.2)),
    ("top_green_plate_interior", "top", 530, 140, 600, 210, "mean", (105.2, 211.9, 173.8)),
    ("top_mat_centre", "top", 230, 300, 330, 340, "mean", (37.6, 35.8, 40.4)),
    ("pre_orange_on_blue_plate", "pre", 239, 133, 251, 145, "median", (233.5, 126.0, 56.0)),
    ("pre_blue_on_green_plate", "pre", 587, 174, 599, 186, "median", (3.0, 194.5, 251.0)),
    ("pre_orange_on_mat", "pre", 304, 240, 316, 252, "median", (253.0, 196.0, 109.0)),
    ("pre_yellow_on_mat", "pre", 468, 249, 480, 261, "median", (254.0, 255.0, 188.0)),
    ("pre_white_on_mat", "pre", 406, 310, 418, 322, "median", (254.0, 255.0, 255.0)),
    ("pre_black_mat", "pre", 240, 300, 260, 320, "median", (38.0, 36.0, 35.0)),
    ("pre_blue_plate_interior", "pre", 185, 160, 205, 180, "median", (87.0, 255.0, 253.0)),
    ("pre_green_plate_interior", "pre", 525, 130, 545, 150, "median", (83.0, 203.0, 158.0)),
]
COLOURS = {"top": (0, 0, 255), "close": (0, 255, 0), "pre": (255, 0, 0)}


def load_rgb(key):
    img_bgr = cv2.imread(SRC[key], cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise SystemExit("cannot read " + SRC[key])
    return img_bgr, img_bgr[:, :, ::-1].copy()


def main():
    imgs, shapes, lighting = {}, {}, {}
    for key in SRC:
        bgr, rgb = load_rgb(key)
        imgs[key] = (bgr, rgb)
        shapes[key] = list(rgb.shape)
        luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        lighting[key] = {"full_frame_luma_mean": round(float(luma.mean()), 2)}
    rows = []
    for name, key, x0, y0, x1, y1, stat, bval in ROIS:
        bgr, rgb = imgs[key]
        crop = rgb[y0:y1, x0:x1].reshape(-1, 3).astype(float)
        mean = crop.mean(axis=0)
        med = np.median(crop, axis=0)
        std = crop.std(axis=0)
        clipped = float((crop >= 250).any(axis=1).mean())
        hsv = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2HSV).reshape(-1, 3)
        rec = {"name": name, "image": SRC[key], "box_xyxy": [x0, y0, x1, y1], "n_px": int(len(crop)),
               "mean_rgb": [round(float(v), 1) for v in mean], "median_rgb": [round(float(v), 1) for v in med],
               "std_rgb": [round(float(v), 1) for v in std], "frac_any_channel_ge250": round(clipped, 3),
               "V_median_0_255": int(np.median(hsv[:, 2])), "builder_value": list(bval), "builder_stat": stat}
        used = mean if stat == "mean" else med
        rec["diff_vs_builder_rgb"] = [round(float(used[i] - bval[i]), 1) for i in range(3)]
        rows.append(rec)
    mat_v = {}
    for key in ("top", "close"):
        bgr, rgb = imgs[key]
        if key == "top":
            box = rgb[300:340, 230:330]
        else:
            box = rgb[250:330, 180:300]
        hsv = cv2.cvtColor(np.ascontiguousarray(box), cv2.COLOR_RGB2HSV)
        mat_v[key] = {"mat_centre_V_p50": int(np.percentile(hsv[..., 2], 50))}
    res = {"stage": "twin-scene-v1", "worker": "V1", "images": SRC, "shapes_hw3": shapes,
           "lighting": lighting, "mat_V": mat_v, "rois": rows}
    with open(OUT + "/colour_check.json", "w") as f:
        json.dump(res, f, indent=1)
    for key in ("top", "close", "pre"):
        bgr, _ = imgs[key]
        ov = bgr.copy()
        for name, k2, x0, y0, x1, y1, stat, bval in ROIS:
            if k2 != key:
                continue
            cv2.rectangle(ov, (x0, y0), (x1 - 1, y1 - 1), COLOURS[key], 2)
            cv2.putText(ov, name.split("_", 1)[1][:22], (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        COLOURS[key], 1)
        cv2.imwrite(OUT + "/roi_overlay_" + key + ".png", ov)
    cells = []
    for name, key, x0, y0, x1, y1, stat, bval in ROIS:
        bgr, _ = imgs[key]
        pad = 8
        xa, ya = max(0, x0 - pad), max(0, y0 - pad)
        xb, yb = min(bgr.shape[1], x1 + pad), min(bgr.shape[0], y1 + pad)
        crop = bgr[ya:yb, xa:xb].copy()
        cv2.rectangle(crop, (x0 - xa, y0 - ya), (x1 - xa - 1, y1 - ya - 1), (0, 255, 255), 1)
        scale = min(300.0 / crop.shape[1], 150.0 / crop.shape[0])
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cell = np.full((170, 310, 3), 40, np.uint8)
        cell[20:20 + crop.shape[0], 5:5 + crop.shape[1]] = crop
        cv2.putText(cell, name[:40], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cells.append(cell)
    while len(cells) % 4:
        cells.append(np.full((170, 310, 3), 40, np.uint8))
    grid = [np.hstack(cells[i:i + 4]) for i in range(0, len(cells), 4)]
    cv2.imwrite(OUT + "/roi_crops.png", np.vstack(grid))
    print(json.dumps({"lighting": lighting, "mat_V": mat_v}))
    for r in rows:
        print(r["name"], r["n_px"], "mean", r["mean_rgb"], "median", r["median_rgb"], "clip", r["frac_any_channel_ge250"],
              "builder", r["builder_value"], "diff", r["diff_vs_builder_rgb"], "V", r["V_median_0_255"])


if __name__ == "__main__":
    main()
