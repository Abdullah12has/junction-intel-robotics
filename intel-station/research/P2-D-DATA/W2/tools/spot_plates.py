import json, sys, os
import numpy as np
from PIL import Image

FR = "/home/intel/intel-junction-2026/work/P2-D-DATA/W2/results3/frames/6d6bdce3-b382-478d-9697-fb9e573b97af"
TOP = "243622061421"


def load(p):
    return np.asarray(Image.open(p).convert("RGB")).astype(np.int32)


def blob(mask):
    ys, xs = np.nonzero(mask)
    if xs.size < 50:
        return None
    return {"cx": round(float(xs.mean()), 1), "cy": round(float(ys.mean()), 1),
            "sx": round(float(xs.std()), 1), "sy": round(float(ys.std()), 1), "px": int(xs.size)}


def plates(img):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    blue = (b > r + 60) & (b > 120) & (g > r + 20)
    green = (g > r + 40) & (g > b + 10)
    h, w = r.shape
    left = np.zeros_like(blue); left[: h // 2 + 40, : w // 2] = True
    right = np.zeros_like(green); right[: h // 2 + 40, w // 2 :] = True
    return {"blue_plate": blob(blue & left), "green_plate": blob(green & right)}


def changed(a, b, thr=60):
    d = np.abs(a - b).sum(axis=2) > thr
    m = np.zeros_like(d); m[60:380, 120:520] = True
    return d & m


out = {"method": "top view 243622061421, 640x480 PNG; plate centroid by colour threshold per half; "
                 "placed-brick = |start - end| sum-RGB > 60 inside mat box rows 60-380, cols 120-520; centroid and spread of changed pixels",
       "plates_at_ep3_start": plates(load(f"{FR}/ep3_{TOP}_start.png")), "per_episode": {}}
for ep in (3, 4, 5, 6):
    a = load(f"{FR}/ep{ep}_{TOP}_start.png")
    b = load(f"{FR}/ep{ep}_{TOP}_end.png")
    out["per_episode"][str(ep)] = {"placed_brick_top": blob(changed(a, b)), "plates_at_end": plates(b)}
print(json.dumps(out, indent=1))
