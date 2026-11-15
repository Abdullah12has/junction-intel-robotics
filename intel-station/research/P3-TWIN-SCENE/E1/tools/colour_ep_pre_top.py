"""E1 colour check: median colour of brick, plate and mat ROIs in the episode snapshot
logs/sort/any_1740/ep_0001/pre_top.jpg (read-only input). ROI boxes were placed by eye on
the 640x480 image; each box is 12x12 px (or 20x20 px for plates and mat) inside the
object's top face. Writes nothing; prints JSON to stdout."""
import json

import cv2
import numpy as np

SRC = "/home/intel/intel-junction-2026/logs/sort/any_1740/ep_0001/pre_top.jpg"
ROIS = {
    "orange_2x2_on_mat": (304, 240, 316, 252),
    "orange_2x2_on_blue_plate": (239, 133, 251, 145),
    "blue_2x2_on_green_plate": (587, 174, 599, 186),
    "yellow_2x2_on_mat": (468, 249, 480, 261),
    "white_2x2_on_mat": (406, 310, 418, 322),
    "black_mat_lower_centre": (240, 300, 260, 320),
    "blue_plate_interior": (185, 160, 205, 180),
    "green_plate_interior": (525, 130, 545, 150),
}


def main():
    im = cv2.imread(SRC, cv2.IMREAD_COLOR)
    if im is None:
        raise SystemExit("cannot read " + SRC)
    rgb = im[:, :, ::-1].astype(np.float64)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    out = {"source": SRC, "shape_hw": list(im.shape[:2]), "rois": {}}
    for name, (x0, y0, x1, y1) in ROIS.items():
        p = rgb[y0:y1, x0:x1].reshape(-1, 3)
        h = hsv[y0:y1, x0:x1].reshape(-1, 3).astype(np.float64)
        out["rois"][name] = {
            "box_xyxy": [x0, y0, x1, y1],
            "rgb_median": np.median(p, axis=0).round(1).tolist(),
            "rgb_p10": np.percentile(p, 10, axis=0).round(1).tolist(),
            "rgb_p90": np.percentile(p, 90, axis=0).round(1).tolist(),
            "hsv_median_opencv": np.median(h, axis=0).round(1).tolist(),
        }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
