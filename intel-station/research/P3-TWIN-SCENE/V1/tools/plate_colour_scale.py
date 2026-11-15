#!/usr/bin/env python3
"""V1 plate-spacing check on the top colour frame that does not need the colour camera's field of view.
The blue and green plate outlines are segmented by hue inside fixed regions; an ellipse is fitted to each outline
(the green plate is cut by the right image edge, so only its arc is used). The centre separation is expressed in
plate diameters (K), then multiplied by the plate outer diameters measured from the depth data (177.3-190.3 mm).
Also solves the diameter that would make PLATES_XY (304.49 mm) true at this K.
Horizontal pixels scale with mm/px = D/(2 a_x); vertical pixels are foreshortened by cos(tilt).
Writes outputs/plate_colour_scale.json and outputs/plate_colour_overlay.png."""
import json
import math

import cv2
import numpy as np

COL = "/home/intel/intel-junction-2026/logs/depth/g1_162903/top_color.jpg"
OUT = "/home/intel/intel-junction-2026/work/P3-TWIN-SCENE/V1/outputs"
TILT_DEG = 20.32
COS_T = math.cos(math.radians(TILT_DEG))
DEPTH_D = {"ridge_blue_v1_179.5": 179.5, "outline_green_v4_177.34": 177.34, "ridge_green_v1_190.26": 190.26}
PLATES_XY_SEP = 304.49
BLUE_ROI = (100, 30, 370, 265)
GREEN_ROI = (470, 60, 640, 290)


def comp_in_roi(mask, roi):
    x0, y0, x1, y1 = roi
    m = np.zeros_like(mask)
    m[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return (lab == i).astype(np.uint8), int(st[i, cv2.CC_STAT_AREA])


def boundary(comp, margin=3):
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)[:, 0, :].astype(np.float32)
    h, w = comp.shape
    keep = ((c[:, 0] > margin) & (c[:, 0] < w - 1 - margin) &
            (c[:, 1] > margin) & (c[:, 1] < h - 1 - margin))
    return c[keep], c


def main():
    bgr = cv2.imread(COL, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    Hh, Ss, Vv = (hsv[..., k].astype(int) for k in range(3))
    cyan = ((Hh >= 84) & (Hh <= 110) & (Ss >= 90) & (Vv >= 90)).astype(np.uint8)
    green = ((Hh >= 50) & (Hh <= 88) & (Ss >= 60) & (Vv >= 60)).astype(np.uint8)
    cb, area_b = comp_in_roi(cyan, BLUE_ROI)
    cg, area_g = comp_in_roi(green, GREEN_ROI)
    pb, cb_all = boundary(cb)
    pg, cg_all = boundary(cg)
    eb = cv2.fitEllipse(pb)
    eg = cv2.fitEllipse(pg)
    bx0, bx1 = float(pb[:, 0].min()), float(pb[:, 0].max())
    by0, by1 = float(pb[:, 1].min()), float(pb[:, 1].max())
    a_x = 0.5 * (bx1 - bx0)
    b_y = 0.5 * (by1 - by0)
    cbx, cby = eb[0]
    cgx, cgy = eg[0]
    du = cgx - cbx
    dv = cgy - cby
    radial_eff = math.hypot(du, dv / COS_T)
    K = radial_eff / (2.0 * a_x)
    res = {"stage": "twin-scene-v1", "worker": "V1", "image": COL, "shape_hw": list(bgr.shape[:2]),
           "tilt_deg_used": TILT_DEG, "blue": {"area_px": area_b, "bbox_x": [bx0, bx1], "bbox_y": [by0, by1],
                                              "ellipse_centre_px": [round(float(cbx), 2), round(float(cby), 2)],
                                              "ellipse_axes_px": [round(float(v), 2) for v in eb[1]],
                                              "horizontal_semi_axis_px": round(a_x, 2),
                                              "vertical_semi_axis_px": round(b_y, 2),
                                              "n_boundary_pts": int(len(pb))},
           "green": {"area_px": area_g, "ellipse_centre_px": [round(float(cgx), 2), round(float(cgy), 2)],
                     "ellipse_axes_px": [round(float(v), 2) for v in eg[1]],
                     "boundary_x_max_px": round(float(pg[:, 0].max()), 1),
                     "n_boundary_pts": int(len(pg)), "n_contour_pts": int(len(cg_all))},
           "delta_px": [round(float(du), 2), round(float(dv), 2)],
           "separation_in_plate_diameters_K": round(K, 4),
           "PLATES_XY_separation_mm": PLATES_XY_SEP,
           "diameter_needed_for_PLATES_XY_mm": round(PLATES_XY_SEP / K, 2),
           "implied_separation_mm_by_depth_diameter": {},
           "radial_offset_mm_mat_y": None}
    for k, D in DEPTH_D.items():
        res["implied_separation_mm_by_depth_diameter"][k] = round(K * D, 2)
    res["radial_offset_mm_mat_y"] = round(abs(dv) / COS_T * (DEPTH_D["ridge_blue_v1_179.5"] / (2.0 * a_x)), 2)
    res["reference_depth_separations_mm"] = {"builder_PLATES_XY": 304.49, "builder_ridge_v1": 341.31,
                                             "builder_outline_v4_pair_uses_rejected_blue": 431.73}
    with open(OUT + "/plate_colour_scale.json", "w") as fh:
        json.dump(res, fh, indent=1)
    ov = bgr.copy()
    cv2.ellipse(ov, eb, (0, 0, 255), 2)
    cv2.ellipse(ov, eg, (0, 255, 0), 2)
    cv2.circle(ov, (int(cbx), int(cby)), 4, (0, 0, 255), -1)
    cv2.circle(ov, (int(cgx), int(cgy)), 4, (0, 255, 0), -1)
    cv2.line(ov, (int(cbx), int(cby)), (int(cgx), int(cgy)), (255, 255, 255), 1)
    cv2.imwrite(OUT + "/plate_colour_overlay.png", ov)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
