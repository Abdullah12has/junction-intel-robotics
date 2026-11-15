#!/usr/bin/env python3
"""Opus I-60 pick-anywhere: wrist-camera visual servoing + differential IK (kin.ik_rel, small moves from the current
configuration, so absolute FK offsets cancel) + brick-size height check. Never lets the jaws reach the mat
(TIP_FINAL 8 mm; Hassan 17:24Z: touching the mat loosens the fixed jaw).

Frames: base frame from the URDF FK (mm). Wrist image Jacobian M (px per mm of TCP motion) from calib_M.py,
rotated with the tool yaw. TARGET = brick centre pixel when the jaws are centred at the hover (jaw_check.py).
"""
import json
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/home/intel/intel-junction-2026/app/tools")
import grasp_auto as ga  # noqa: E402
import kin  # noqa: E402

R = ga.R
HOVER_DZ = 57.7
TIP_FINAL = 8.0
SIZE_K, SIZE_C = 9886.0, 33.0          # tip height above the mat (mm) = SIZE_K / short_side_px - SIZE_C (hover tilt)
SCAN_UP = 70.0
SERVO_SPEED = 12.0
SCALF = f"{R}/app/tools/scan_calib.json"
PLATE_HUES = ((85, 112), (38, 85))     # blue, green plates (OpenCV hue)


def target():
    return np.array(ga.TARGET_PX, dtype=float)


COLOUR_CLASSES = {  # hue classes so touching bricks of different colours become separate blobs (17:54Z pink+cream)
    "red": lambda H, S, V: ((H <= 8) | (H >= 165)) & (S >= 90) & (V >= 60),
    "orange": lambda H, S, V: (H >= 9) & (H <= 16) & (S >= 90) & (V >= 70),
    "yellow": lambda H, S, V: (H >= 17) & (H <= 38) & (S >= 45) & (V >= 90),
    "green": lambda H, S, V: (H >= 39) & (H <= 85) & (S >= 70) & (V >= 60),
    "blue": lambda H, S, V: (H >= 86) & (H <= 130) & (S >= 90) & (V >= 60),
    "white": lambda H, S, V: (S < 45) & (V >= 150),
}


def blobs(img, min_area=2500, max_area=60000):
    """Brick-like blobs in a wrist frame (jaws masked), one connected component per colour class, with
    plate-surround flag."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    jaw = ga.JAW_MASK if ga.JAW_MASK is not None else np.zeros(H.shape, bool)
    plate = np.zeros(H.shape, bool)
    for lo, hi in PLATE_HUES:
        plate |= (H >= lo) & (H < hi) & (S >= 70) & (V >= 70)
    out = []
    for cname, fn in COLOUR_CLASSES.items():
        fg = fn(H, S, V) & ~jaw
        fg8 = cv2.morphologyEx(fg.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        fg8 = cv2.morphologyEx(fg8, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        # 18:16Z: one white brick split into top/side pieces and the servo alternated between them; label on a
        # 15 px bridge so the pieces of one brick stay one blob, measure on the real pixels only.
        bridge = cv2.dilate(fg8, np.ones((15, 15), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(bridge)
        for i in range(1, n):
            if int(st[i, 4]) < min_area:
                continue
            m = (lab == i) & (fg8 > 0)
            a = int(m.sum())
            if a < min_area or a > max_area:
                continue
            ys, xs = np.nonzero(m)
            rect = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
            (cx, cy), (sw, sh), ang = rect
            yaw = ang % 90.0
            if yaw > 45:
                yaw -= 90.0
            ring = cv2.dilate(m.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool) & ~m
            plate_frac = float(plate[ring].mean()) if ring.any() else 0.0
            box = cv2.boxPoints(rect)
            e1, e2 = box[1] - box[0], box[2] - box[1]
            lng = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
            x, y, w, h = int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
            border = x <= 2 or y <= 2 or x + w >= img.shape[1] - 2 or y + h >= img.shape[0] - 2
            out.append({"centre": [round(float(cx), 1), round(float(cy), 1)], "size": [round(float(sw), 1), round(float(sh), 1)],
                        "short": round(float(min(sw, sh)), 1), "area": a, "yaw": round(float(yaw), 1),
                        "aspect": round(float(max(sw, sh) / max(1.0, min(sw, sh))), 2),
                        "long_vertical": bool(abs(lng[1]) >= abs(lng[0])), "plate_frac": round(plate_frac, 2),
                        "border": bool(border), "hue": int(np.median(H[m])), "sat": int(np.median(S[m])),
                        "val": int(np.median(V[m])), "colour": cname})
    return out


def nearest(bl, px, max_d=1e9, ok=lambda b: True):
    c = [b for b in bl if ok(b)]
    if not c:
        return None
    b = min(c, key=lambda b: math.hypot(b["centre"][0] - px[0], b["centre"][1] - px[1]))
    return b if math.hypot(b["centre"][0] - px[0], b["centre"][1] - px[1]) <= max_d else None


def frame():
    return ga.fresh("wrist", time.time() + 0.25)


def m_base(q):
    return kin.m_base(kin.psi_of(q))


def move_rel(q, g, d, speed=6.0, comp=True, clamp=None):
    qn = kin.ik_rel(q, d)
    if qn is None:
        return None
    c = {"op": "move", "q": qn + [g], "speed": speed, "comp": comp}
    if clamp is not None:
        c["comp_clamp"] = clamp
    st = ga.cmd(c)
    if st["status"] != "ok":
        raise ga.Stop(f"move_rel {d} -> {st['status']} {st['msg']}")
    return qn


COMPAT = {"red": ("red", "orange"), "orange": ("orange", "red", "yellow"), "yellow": ("yellow", "white", "orange"),
          "white": ("white", "yellow"), "green": ("green",), "blue": ("blue",)}


def _ok(want, short_ref, strict):
    def ok(x):
        if x["plate_frac"] >= 0.3:
            return False
        if want and x.get("colour") not in ((want,) if strict else COMPAT.get(want, (want,))):
            return False
        if short_ref and not (0.6 * short_ref <= x["short"] <= 1.6 * short_ref):
            return False
        return True
    return ok


def find(bl, px, max_d, want=None, short_ref=None):
    """Blob nearest px of the tracked brick's colour class (exact class first, then a compatible one). 17:56Z: the
    servo jumped to the neighbouring pink brick after the yaw because it took any blob nearest the target."""
    b = nearest(bl, px, max_d=max_d, ok=_ok(want, short_ref, True))
    if b is None and want:
        b = nearest(bl, px, max_d=max_d, ok=_ok(want, short_ref, False))
    return b


SERVO_MIN_AREA = 1500  # 17:59Z: a brick half under the static jaw mask fell below 2500 px and was "not visible"


def servo(q, g, start_px, tol=7.0, max_it=6, log=print, want=None):
    """Bring the tracked brick (colour class `want`) to TARGET at the current height. Returns (q, blob, converged)."""
    px = np.array(start_px, float)
    b = None
    for it in range(max_it):
        bl = blobs(frame(), min_area=SERVO_MIN_AREA)
        b = find(bl, px, 140, want)
        if b is None:
            return q, None, False
        e = target() - np.array(b["centre"])
        log(f"servo it{it}: {b.get('colour')} brick {b['centre']} err {e.round(1).tolist()} short {b['short']}")
        if np.linalg.norm(e) <= tol:
            return q, b, True
        d = 0.65 * np.linalg.solve(m_base(q), e)  # damped: the 17:32Z servo overshot at gain 1
        n = np.linalg.norm(d)
        if n > 30:
            d *= 30 / n
        qn = move_rel(q, g, [d[0], d[1], 0.0], speed=SERVO_SPEED)
        if qn is None:
            qn = move_rel(q, g, [d[0] / 2, d[1] / 2, 0.0], speed=SERVO_SPEED)
            if qn is None:
                return q, b, False
        q = qn
        px = np.array(b["centre"], float) + 0.65 * e  # expected position after the damped step (tracking continuity)
    bl = blobs(frame(), min_area=SERVO_MIN_AREA)
    b = find(bl, target(), 140, want)
    return q, b, b is not None and np.linalg.norm(target() - np.array(b["centre"])) <= 2 * tol


def align_yaw(q, g, b, log=print):
    yaw = b["yaw"]
    if b["aspect"] > 1.4 and not b["long_vertical"]:
        yaw = yaw + 90.0 if yaw < 0 else yaw - 90.0
    if abs(yaw) <= 2.5:
        return q, False
    nr = q[4] - yaw / ga.YAW_PER_ROLL
    if abs(nr) > 97 and b["aspect"] <= 1.4:
        alt = yaw + 90.0 if yaw < 0 else yaw - 90.0
        nr = q[4] - alt / ga.YAW_PER_ROLL
    nr = float(np.clip(nr, -97, 97))
    qn = list(q)
    qn[4] = round(nr, 2)
    st = ga.cmd({"op": "move", "q": qn + [g], "speed": 25, "comp": True, "accel": 120})
    if st["status"] != "ok":
        raise ga.Stop(f"yaw -> {st['status']}")
    log(f"yaw {yaw:.1f} deg -> roll {qn[4]}")
    return qn, True


def tip_from_size(short_px):
    return SIZE_K / max(short_px, 1.0) - SIZE_C


def load_scan():
    return json.load(open(SCALF)) if os.path.exists(SCALF) else None
