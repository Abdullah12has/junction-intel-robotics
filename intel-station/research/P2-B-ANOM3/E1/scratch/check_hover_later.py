"""Compare the current blobs() copy with the hover_blob recorded in each usable episode (where one is recorded).

Each line gives the recording time (UTC, from t_start), the recorded blob, the copy's nearest blob, and whether all
fields match. Nothing is written.
"""
import datetime
import json
import math
import os
import sys

R = "/home/intel/intel-junction-2026"
sys.path.insert(0, R + "/app/anomaly/anom3")

import cv2  # noqa: E402

import extract_crops as X  # noqa: E402

KEYS = ("centre", "size", "short", "area", "yaw", "aspect", "long_vertical", "plate_frac", "border", "hue", "sat",
        "val")


def same(a, b, tol=0.06):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(float(a) - float(b)) <= tol
    return a == b


def main():
    usable, why, counts = X.load_usable()
    jaw = X.jaw_mask()
    for key in sorted(usable):
        run, ep = key.split("/")
        d = os.path.join(R, "logs", "sort", run, ep)
        with open(os.path.join(d, "episode.json"), encoding="utf-8") as fh:
            rec_ep = json.load(fh)
        ts = rec_ep.get("t_start")
        when = datetime.datetime.fromtimestamp(float(ts), datetime.timezone.utc).strftime("%H:%M:%SZ") if ts else "?"
        rec = rec_ep.get("hover_blob")
        if rec is None:
            print(key, when, "no hover_blob recorded", flush=True)
            continue
        img = cv2.imread(os.path.join(d, "crops", "wrist_hover.png"), cv2.IMREAD_COLOR)
        if img is None:
            print(key, when, "no crops/wrist_hover.png", flush=True)
            continue
        bl = X.blobs(img, jaw, 2500, 60000)
        if not bl:
            print(key, when, "copy finds no blob", flush=True)
            continue
        near = min(bl, key=lambda b: math.hypot(b["centre"][0] - rec["centre"][0], b["centre"][1] - rec["centre"][1]))
        ok = all(same(near[k], rec[k]) for k in KEYS)
        print(key, when, "recorded", {k: rec[k] for k in ("centre", "size", "area", "sat", "hue")},
              "copy", {k: near[k] for k in ("centre", "size", "area", "sat", "hue")},
              "identical" if ok else "differs", flush=True)


main()
