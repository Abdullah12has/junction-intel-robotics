"""Validation of the copies in R/app/anomaly/anom3/extract_crops.py (round 3). Read-only: nothing is written.

1. blobs() on crops/wrist_hover.png must give the recorded hover_blob of any_1740/ep_0001.
2. close_pre_verdict() on crops/pre_close_full.jpg must give the recorded pre_close of the same episode.
3. Close-camera seek: the episode window is streamed at 10 fps with -ss (t - start_time). close_frame(t) must give
   the same frame, and the streamed frame closest to pre_close_full.jpg must give the recorded bbox.
4. assoc.json: the ep key must match the usable run/ep keys, and the frame paths must exist.
"""
import collections
import json
import math
import os
import subprocess
import sys

R = "/home/intel/intel-junction-2026"
sys.path.insert(0, R + "/app/anomaly/anom3")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import extract_crops as X  # noqa: E402

EP = os.path.join(R, "logs", "sort", "any_1740", "ep_0001")
KEYS = ("centre", "size", "short", "area", "yaw", "aspect", "long_vertical", "plate_frac", "border", "hue", "sat",
        "val")


def same(a, b, tol=0.06):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(float(a) - float(b)) <= tol
    return a == b


def main():
    with open(os.path.join(EP, "episode.json"), encoding="utf-8") as fh:
        ep = json.load(fh)
    jaw = X.jaw_mask()
    print("jaw mask", jaw.shape, "fraction True %.4f" % float(jaw.mean()), flush=True)

    wrist = cv2.imread(os.path.join(EP, "crops", "wrist_hover.png"), cv2.IMREAD_COLOR)
    print("wrist_hover.png", None if wrist is None else wrist.shape, flush=True)
    rec = ep["hover_blob"]
    blobs = X.blobs(wrist, jaw, 2500, 60000)
    near = min(blobs, key=lambda b: math.hypot(b["centre"][0] - rec["centre"][0], b["centre"][1] - rec["centre"][1]))
    ok1 = all(same(near[k], rec[k]) for k in KEYS)
    print("1 copy     ", {k: near[k] for k in KEYS})
    print("1 recorded ", {k: rec[k] for k in KEYS})
    print("1 PASS" if ok1 else "1 FAIL", flush=True)

    pimg = cv2.imread(os.path.join(EP, "crops", "pre_close_full.jpg"), cv2.IMREAD_COLOR)
    print("pre_close_full.jpg", None if pimg is None else pimg.shape, flush=True)
    prec = ep["pre_close"]
    v = X.close_pre_verdict(pimg, want_hue=None)
    print("2 copy     ", v)
    print("2 recorded ", prec)
    ok2 = all(same(v.get(k), prec.get(k)) for k in ("verdict", "reasons", "colour", "tape_score", "bbox", "dist",
                                                    "jaw_mid"))
    print("2 PASS" if ok2 else "2 FAIL", flush=True)

    start = X.ffprobe_start()
    t0, t1 = float(ep["t_start"]), float(ep["t_end"])
    print("close start_time %.3f; episode t_start %.3f t_end %.3f; -ss offset %.3f"
          % (start, t0, t1, t0 - start), flush=True)
    H, W = pimg.shape[:2]
    nbytes = W * H * 3
    cmd = ["ffmpeg", "-v", "error", "-ss", "%.3f" % (t0 - start), "-i", X.CLOSE_MKV, "-t", "%.3f" % (t1 - t0 + 2.0),
           "-vf", "fps=10", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frames = []
    while True:
        buf = proc.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        frames.append(np.frombuffer(buf, np.uint8).reshape(H, W, 3))
    err = proc.stderr.read()
    proc.wait()
    print("streamed %d frames at 10 fps; ffmpeg exit %d %s" % (len(frames), proc.returncode, err[:300].decode(
        "utf-8", "replace")), flush=True)
    ref = pimg.astype(np.float32)
    diffs = [float(np.abs(f.astype(np.float32) - ref).mean()) for f in frames]
    order = sorted(range(len(frames)), key=lambda i: diffs[i])[:5]
    for i in order:
        print("  streamed frame %d t=%.2f mean abs diff to pre_close_full %.2f" % (i, t0 + i / 10.0, diffs[i]),
              flush=True)
    bi = order[0]
    tb = t0 + bi / 10.0
    cf = X.close_frame(tb, start)
    if cf is None:
        print("3 FAIL close_frame returned no frame", flush=True)
    else:
        d_cf = float(np.abs(cf.astype(np.float32) - frames[bi].astype(np.float32)).mean())
        vb = X.close_pre_verdict(cf, want_hue=None)
        print("3 close_frame(%.2f) vs streamed frame: mean abs diff %.2f" % (tb, d_cf), flush=True)
        print("3 verdict on close_frame(t) of the best match:", vb, flush=True)
        ok3 = same(vb.get("bbox") if vb else None, prec["bbox"])
        print("3 PASS" if ok3 else "3 CHECK (bbox differs from recorded; see values above)", flush=True)

    assoc_path = os.path.join(R, "work", "eyeinhand", "assoc.json")
    with open(assoc_path, encoding="utf-8") as fh:
        assoc = json.load(fh)
    print("assoc entries", len(assoc), "keys", sorted(assoc[0].keys()), flush=True)
    print("assoc[0]", assoc[0], flush=True)
    usable, why, counts = X.load_usable()
    print("usable", sorted(usable), flush=True)
    print("skipped", counts, flush=True)
    eps = collections.Counter(str(e.get("ep")) for e in assoc)
    print("distinct ep values", sorted(eps.items())[:40], flush=True)
    print("assoc entries whose ep is a usable key:", sum(1 for e in assoc if str(e.get("ep")) in usable), flush=True)
    missing = [e.get("frame") for e in assoc
               if not (e.get("frame") and os.path.isfile(e["frame"] if os.path.isabs(e["frame"])
                                                         else os.path.join(R, e["frame"])))]
    print("assoc frames missing", len(missing), missing[:5], flush=True)
    ts = [float(e["t"]) for e in assoc if e.get("t") is not None]
    print("assoc t range %.3f .. %.3f (close start %.3f)" % (min(ts), max(ts), start), flush=True)


main()
