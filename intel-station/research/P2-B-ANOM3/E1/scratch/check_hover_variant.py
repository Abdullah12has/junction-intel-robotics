"""Does the recorded hover_blob of any_1740/ep_0001 come from the bridged-component measure?

extract_crops.blobs (the current sorter logic) measures m = (lab == i) & (fg8 > 0), the real pixels only. The sorter's
comment dated 18:16Z says that change replaced an older measure. This script runs the same function with that one line
changed to m = (lab == i), the bridged component, and compares the result with episode.json hover_blob.
Nothing is written.
"""
import inspect
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
EP = os.path.join(R, "logs", "sort", "any_1740", "ep_0001")
OLD = "            m = (lab == i) & (fg8 > 0)\n"
NEW = "            m = (lab == i)\n"


def same(a, b, tol=0.06):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(float(a) - float(b)) <= tol
    return a == b


def nearest(bl, rec):
    return min(bl, key=lambda b: math.hypot(b["centre"][0] - rec["centre"][0], b["centre"][1] - rec["centre"][1]))


def main():
    src = inspect.getsource(X.blobs)
    n_old = src.count(OLD)
    print("measure line found %d time(s)" % n_old, flush=True)
    assert n_old == 1
    ns = dict(vars(X))
    exec(compile(src.replace(OLD, NEW), "blobs_bridged", "exec"), ns)
    blobs_bridged = ns["blobs"]
    with open(os.path.join(EP, "episode.json"), encoding="utf-8") as fh:
        rec = json.load(fh)["hover_blob"]
    img = cv2.imread(os.path.join(EP, "crops", "wrist_hover.png"), cv2.IMREAD_COLOR)
    jaw = X.jaw_mask()
    cur = nearest(X.blobs(img, jaw, 2500, 60000), rec)
    brg = nearest(blobs_bridged(img, jaw, 2500, 60000), rec)
    print("current measure  ", {k: cur[k] for k in KEYS}, flush=True)
    print("bridged measure  ", {k: brg[k] for k in KEYS}, flush=True)
    print("recorded         ", {k: rec[k] for k in KEYS}, flush=True)
    ok = all(same(brg[k], rec[k]) for k in KEYS)
    print("bridged measure reproduces the recorded hover_blob:", ok, flush=True)


main()
