"""Top-view cube detector for the marker (RealSense colour node, 640x480, black mat)."""
import cv2
import numpy as np

CLASSES = {
    "red": lambda H, S, V: ((H <= 8) | (H >= 165)) & (S >= 80) & (V >= 120),
    "orange": lambda H, S, V: (H >= 9) & (H <= 16) & (S >= 120) & (V >= 100),
    "light": lambda H, S, V: (((H >= 17) & (H <= 38) & (S >= 30)) | (S < 45)) & (V >= 150),
    "green": lambda H, S, V: (H >= 39) & (H <= 85) & (S >= 150) & (V >= 80),
    "blue": lambda H, S, V: (H >= 86) & (H <= 130) & (S >= 170) & (V >= 80),
}
PLATE = lambda H, S, V: (H >= 70) & (H <= 105) & (S >= 60) & (S < 170) & (V >= 150)


def poly_mask(shape, polys):
    m = np.zeros(shape[:2], np.uint8)
    for poly in polys:
        cv2.fillPoly(m, [np.asarray(poly, np.int32)], 1)
    return m.astype(bool)


def detect(img, workspace, keepout=(), area=(450, 3600), max_aspect=1.8, min_fill=0.55):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    ws = poly_mask(V.shape, workspace)
    plate = PLATE(H, S, V).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(plate)
    plates = np.zeros(V.shape, np.uint8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= 6000:
            plates[lab == i] = 1
    plates = cv2.dilate(plates, np.ones((21, 21), np.uint8)).astype(bool)
    ko = poly_mask(V.shape, keepout)
    out = []
    for name, fn in CLASSES.items():
        fg = (fn(H, S, V) & ws).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        k, lab, st, cen = cv2.connectedComponentsWithStats(fg)
        for i in range(1, k):
            a = int(st[i, cv2.CC_STAT_AREA])
            if not area[0] <= a <= area[1]:
                continue
            m = lab == i
            ys, xs = np.nonzero(m)
            (cx, cy), (w, h), ang = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
            aspect = max(w, h) / max(1.0, min(w, h))
            fill = a / max(1.0, w * h)
            cxi, cyi = int(round(cx)), int(round(cy))
            reason = None
            if aspect > max_aspect:
                reason = "aspect"
            elif fill < min_fill:
                reason = "fill"
            elif plates[cyi, cxi] or plates[m].mean() > 0.3:
                reason = "plate"
            elif ko[cyi, cxi]:
                reason = "keepout"
            out.append({"u": round(float(cx), 1), "v": round(float(cy), 1), "area": a, "colour": name,
                        "size": [round(float(w), 1), round(float(h), 1)], "aspect": round(float(aspect), 2),
                        "fill": round(float(fill), 2), "ok": reason is None, "reject": reason})
    return out
