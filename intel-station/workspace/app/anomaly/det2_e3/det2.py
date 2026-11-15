"""P2-B-DET2 (E3) core: brick segmentation on the black mat, footprint shape, colour and tape features.
CPU only (OpenCV + NumPy). Top RealSense view gives position, footprint shape and colour; close view gives tape.
Calibration (static background, mat ROI, 2x2 reference area) is written by build_model.py into models/.
D-068 changes: static mask also covers grey/bright non-brick parts (rail, plate rim); shape bands are tight and
anything outside them is 'unknown' (merged blobs, side-lying or partial bricks); tape = dark grey fraction inside
the brick silhouette (convex hull), because cand() rejects dark grey tape pixels and the mask has holes there."""
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
VIEW_CFG = {"top": {"area": (500, 3500), "minside": 18, "robot": [(296, 0, 420, 140)]},
            "close": {"area": (1500, 60000), "minside": 30, "robot": [(0, 0, 75, 60), (100, 0, 200, 55)]}}
MIN_FILL = {"top": 0.72, "close": 0.45}
TAPE_FEATS = ("dark_grey",)
GREY_V = 100                              # static background: any pixel brighter than this counts as brick-like
SHAPE_2X2 = ((0.85, 1.18), (0.85, 1.20))  # (aspect, area ratio to a_ref) band for 2x2
SHAPE_2X4 = ((1.80, 2.30), (1.60, 2.50))  # band for a clear 2x4; everything else is 'unknown'


def hsv_of(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def cand(hsv):
    """Brick-like pixels: saturated and bright, or very bright (white bricks)."""
    return ((hsv[..., 1] > 70) & (hsv[..., 2] > 90)) | (hsv[..., 2] > 150)


def mat_hull(hsv, pad=6):
    """Convex hull of the largest dark component (the black mat), eroded by pad px."""
    dark = cv2.morphologyEx((hsv[..., 2] < 70).astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(dark, 8)
    m = (lbl == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hm = np.zeros(m.shape, np.uint8)
    cv2.fillPoly(hm, [cv2.convexHull(max(cnts, key=cv2.contourArea))], 1)
    return cv2.erode(hm, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)).astype(bool)


def region(bgr, view):
    return mat_hull(hsv_of(bgr)) if view == "top" else np.ones(bgr.shape[:2], bool)


def build_static(bgrs, view):
    """Static mask = pixels that are brick-like or bright (V > GREY_V) in >= 90% of reference frames:
    robot, plates, grey rail and plate rim. Returns (static, roi)."""
    rois, hits = [], []
    for b in bgrs:
        hsv = hsv_of(b)
        r = region(b, view)
        rois.append(r)
        hits.append(((cand(hsv) | (hsv[..., 2] > GREY_V)) & r).astype(np.float32))
    freq = np.mean(np.stack(hits, 0), axis=0)
    static = cv2.dilate((freq >= 0.9).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    return static, np.mean(np.stack(rois, 0), axis=0) >= 0.5


def find_bricks(bgr, view, static, roi):
    hsv = hsv_of(bgr)
    fg = (cand(hsv) & roi & ~static).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(fg, 8)
    cfg, H, W = VIEW_CFG[view], fg.shape[0], fg.shape[1]
    out = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if not (cfg["area"][0] <= a <= cfg["area"][1]) or min(w, h) < cfg["minside"]:
            continue
        comp = lbl[y:y + h, x:x + w] == i
        cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea) + np.array([x, y], np.int32)
        rect = cv2.minAreaRect(c)
        if a / max(rect[1][0] * rect[1][1], 1.0) < MIN_FILL[view]:
            continue
        cx, cy = rect[0]
        if any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in cfg["robot"]):
            continue
        out.append({"bbox": [x, y, w, h], "area": a, "rect": rect, "mask": comp,
                    "touch": bool(x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2)})
    return out


def _margin(pairs):
    """Margin-based confidence in [0, 1]: distance to the nearest band edge, scaled by 0.1."""
    return float(np.clip(min(min(v - lo, hi - v) for v, lo, hi in pairs) / 0.1, 0.0, 1.0))


def shape_of(b, a_ref):
    """Footprint shape from minAreaRect aspect and area ratio to a calibrated 2x2 area.
    Touching the frame edge, merged blobs (ratio > band) and side-lying or partial bricks (ratio < band) are unknown."""
    (_, _), (rw, rh), _ = b["rect"]
    asp = max(rw, rh) / max(min(rw, rh), 1e-6)
    ratio = b["area"] / a_ref
    if b["touch"]:
        return "unknown", 0.0, asp, ratio
    (a0, a1), (r0, r1) = SHAPE_2X2
    if a0 <= asp <= a1 and r0 <= ratio <= r1:
        return "2x2", _margin([(asp, a0, a1), (ratio, r0, r1)]), asp, ratio
    (b0, b1), (q0, q1) = SHAPE_2X4
    if b0 <= asp <= b1 and q0 <= ratio <= q1:
        return "not_2x2", _margin([(asp, b0, b1), (ratio, q0, q1)]), asp, ratio
    return "unknown", 0.0, asp, ratio


def colour_of(hsv_px):
    h, s, v = (float(np.median(hsv_px[:, k])) for k in range(3))
    if s < 60:
        name = "white" if v > 160 else "grey"
    elif h < 8 or h >= 160:
        name = "red"
    elif 12 <= h < 40 and 45 <= s < 120 and v > 150:
        name = "cream"
    elif h < 22:
        name = "orange"
    elif h < 40:
        name = "yellow"
    elif 95 <= h < 130:
        name = "blue"
    elif 40 <= h < 95:
        name = "green"
    else:
        name = "other"
    return name, [round(h, 1), round(s, 1), round(v, 1)]


def core(mask):
    er = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return er if er.sum() >= 60 else mask


def hull_of(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    h = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(h, [cv2.convexHull(max(cnts, key=cv2.contourArea))], 1)
    return h.astype(bool)


def tape_feats(bgr_c, hsv_c, mask):
    """Dark grey tape fraction: pixels with S < 60 and 50 <= V <= 130 inside the brick silhouette (convex hull of
    the mask, eroded 2 px). Saturated print (the '2' on a red brick) and the black mat (V < 50) are excluded."""
    hull = hull_of(mask)
    inner = cv2.erode(hull.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if inner.sum() < 60:
        inner = hull
    s, v = hsv_c[..., 1], hsv_c[..., 2]
    grey = (s < 60) & (v >= 50) & (v <= 130) & inner
    return {"dark_grey": round(float(grey.sum()) / max(float(inner.sum()), 1.0), 4)}


def mixed_colour(hsv_px):
    """True when one blob holds two materials (saturated and white/grey parts, or two hues): a touching pair of
    bricks, which must not be scored as one 2x4. Green plate pixels are static and removed before this."""
    frac_white = float(np.mean((hsv_px[:, 1] < 60) & (hsv_px[:, 2] > 150)))
    if 0.3 <= frac_white <= 0.7:
        return True
    sat = hsv_px[hsv_px[:, 1] > 60]
    if len(sat) < 30:
        return False
    hues = sat[:, 0].astype(np.float32)
    return bool(np.percentile(hues, 90) - np.percentile(hues, 10) >= 25)


def analyse(bgr, view, model, a_ref=None):
    """All bricks of one view: list of dicts (sorted left to right, top to bottom)."""
    hsv = hsv_of(bgr)
    bricks = find_bricks(bgr, view, model["static"], model["roi"])
    bricks.sort(key=lambda b: (b["bbox"][0], b["bbox"][1]))
    out = []
    for b in bricks:
        x, y, w, h = b["bbox"]
        er = core(b["mask"])
        colour, hsv_med = colour_of(hsv[y:y + h, x:x + w][er])
        (_, _), (rw0, rh0), _ = b["rect"]
        if colour == "red" and b["area"] / max(rw0 * rh0, 1.0) < 0.6:
            continue  # gripper jaw: thin red triangle
        (cx, cy), (rw, rh), ang = b["rect"]
        rec = {"k": len(out), "view": view, "bbox": b["bbox"], "centre_px": [round(cx, 1), round(cy, 1)],
               "angle_deg": round(float(ang), 1), "footprint_px": [round(rw, 1), round(rh, 1)], "area": b["area"],
               "touch_edge": b["touch"], "colour": colour, "hsv_median": hsv_med,
               "tape_feats": tape_feats(bgr[y:y + h, x:x + w], hsv[y:y + h, x:x + w], b["mask"])}
        rec["mixed_colour"] = mixed_colour(hsv[y:y + h, x:x + w][er])
        if view == "top" and a_ref:
            rec["shape"], rec["shape_conf"], rec["aspect"], rec["area_ratio"] = shape_of(b, a_ref)
            if rec["shape"] == "not_2x2" and rec["mixed_colour"]:
                rec["shape"], rec["shape_conf"] = "unknown", 0.0
        out.append(rec)
    return out


def load_model(view):
    z = np.load(os.path.join(MODEL_DIR, f"{view}.npz"))
    return {"static": z["static"], "roi": z["roi"]}


def load_cal():
    with open(os.path.join(MODEL_DIR, "cal.json")) as fh:
        return json.load(fh)
