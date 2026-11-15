#!/usr/bin/env python3
"""P2-D-SIM2 W3: scale-free camera comparison of sim frames vs real RealSense frames (read-only inputs).
Inputs: work/P2-D-SIM2/W3/raw/*.png (sim frame 0 extracted by ffmpeg), data/inspection/views/rs_video8.jpg (real top),
rs_video14.jpg (real close). Output: work/P2-D-SIM2/W3/numbers_cams.json and a side-by-side sheet."""
import json
import math

import cv2
import numpy as np
from PIL import Image

W = "/home/intel/intel-junction-2026/work/P2-D-SIM2/W3"
VIEWS = "/home/intel/intel-junction-2026/data/inspection/views"
PLATE_D_M = 0.128            # sim plate radius 0.064 m (station_scene.py); real plates assumed same size (unverified)
SEP_M = 0.3412               # sim centre-to-centre blue-green (BOWL_CENTERS)


def plate_stats(bgr, lo, hi):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    area = float(st[k, cv2.CC_STAT_AREA])
    return {"cx": round(float(cen[k][0]), 1), "cy": round(float(cen[k][1]), 1), "area_px": int(area),
            "diam_px": round(2 * math.sqrt(area / math.pi), 1)}


def frame_stats(path):
    bgr = cv2.imread(path)
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(float)
    s = hsv[..., 1].astype(float)
    b = {"size": [w, h], "mean_bgr": [round(float(x), 1) for x in bgr.reshape(-1, 3).mean(0)],
         "mean_sat": round(float(s.mean()), 1), "mean_val": round(float(v.mean()), 1),
         "frac_dark_v<60": round(float((v < 60).mean()), 3)}
    # blue plate: cyan-blue hue; green plate: green hue (OpenCV hue 0-180)
    b["blue_plate"] = plate_stats(bgr, (85, 120, 60), (110, 255, 255))
    b["green_plate"] = plate_stats(bgr, (40, 90, 40), (85, 255, 255))
    return b


if __name__ == "__main__":
    sets = {"sim_overhead_f0": f"{W}/raw/sim_overhead_f0.png", "real_top_rs8": f"{W}/raw/real_top_ds.png",
            "sim_close_f0": f"{W}/raw/sim_close_f0.png", "real_close_rs14": f"{W}/raw/real_close_ds.png"}
    res = {k: frame_stats(p) for k, p in sets.items()}
    for k in ("sim_overhead_f0", "real_top_rs8"):
        bp, gp = res[k]["blue_plate"], res[k]["green_plate"]
        if bp and gp:
            sep = math.dist((bp["cx"], bp["cy"]), (gp["cx"], gp["cy"]))
            res[k]["plate_sep_px"] = round(sep, 1)
            res[k]["plate_sep_over_diam"] = round(sep / ((bp["diam_px"] + gp["diam_px"]) / 2), 3)
    if res["sim_overhead_f0"].get("plate_sep_over_diam") and res["real_top_rs8"].get("plate_sep_over_diam"):
        res["sep_over_diam_sim_minus_real"] = round(res["sim_overhead_f0"]["plate_sep_over_diam"]
                                                   - res["real_top_rs8"]["plate_sep_over_diam"], 3)
    res["sim_expected_sep_over_diam"] = round(SEP_M / PLATE_D_M, 3)
    json.dump(res, open(f"{W}/numbers_cams.json", "w"), indent=1)
    # side-by-side sheet for the record (top pair, close pair)
    ims = [cv2.imread(p) for p in sets.values()]
    ims = [cv2.resize(im, (480, 360)) for im in ims]
    sheet = np.vstack([np.hstack(ims[:2]), np.hstack(ims[2:])])
    cv2.imwrite(f"{W}/raw/cams_sheet_sim_vs_real.jpg", sheet)
    print(json.dumps(res))
