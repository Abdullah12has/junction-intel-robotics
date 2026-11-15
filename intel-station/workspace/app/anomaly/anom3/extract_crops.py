"""anom3 round 3: brick-centred crops from every associated frame (wrist_multi and close_multi).

wrist_multi: one crop per entry of R/work/eyeinhand/assoc.json (Opus eye-in-hand fit; u = target pixel in the wrist
frame). The blob nearest u (within 25 px) is cropped as a square of side 1.6 x its longer side, clipped to the frame,
and resized to 256 px. A crop with more than 20 percent of its pixels inside wrist_jaw_mask.png is skipped.
close_multi: one crop per non-scan entry that has a wrist blob. The close-camera frame at the same epoch t is read from
the close recording (ffmpeg, -ss = t - start_time, start_time from ffprobe). The brick under the jaws is found with
close_pre_verdict(img, want_hue), where want_hue is the hue of that entry's wrist blob. The brick box is padded by
12 px (as sort_loop.save_crop) and resized to 256 px. A close crop with close_brick_hue_mismatch is kept in the
manifest as ok_hue_mismatch and is left out of the LOEO training and of the main scores.

Usable episodes (episode.json): label GOOD or DEFECT, success not false, needs_review not true, plate matching the
label. Labels are never changed. The multi views have no held_close crops, so the polluted held_close crops of
any_1740/ep_0005 and any_1731/ep_0003 cannot enter.

blobs(), mask_jaws() and close_pre_verdict() keep the logic of pick_any.blobs, sort_loop.mask_jaws and
sort_any.close_pre_verdict (text check in scratch/verify_copies.log; blobs() takes the jaw mask as an argument).
Those modules are not imported, so this process loads no robot-control code. The only import is inspector.py
(det2_f2: numpy and OpenCV at import).

Output (outside the repo): R/work/P2-B-ANOM3/E1/multi/manifest.tsv and crops/<view>/<run>_<ep>/<idx>.png.
Run: nice -n 19 env OMP_NUM_THREADS=4 $PY -I -B extract_crops.py
"""
import argparse
import csv
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

R = "/home/intel/intel-junction-2026"
sys.path.insert(0, os.path.join(R, "app", "anomaly", "det2_f2"))
import inspector as DET2  # noqa: E402

ASSOC = os.path.join(R, "work", "eyeinhand", "assoc.json")
LOGS_SORT = os.path.join(R, "logs", "sort")
CLOSE_MKV = os.path.join(R, "logs", "monitor", "close_20261114T170351Z.mkv")
JAW_PNG = os.path.join(R, "app", "tools", "wrist_jaw_mask.png")
OUT = os.path.join(R, "work", "P2-B-ANOM3", "E1", "multi")
SIZE = 256
SIDE_K = 1.6
NEAR_PX = 25.0
MAX_JAW_FRAC = 0.20
MIN_AREA, MAX_AREA = 1200, 120000
PAD = 12
PLATE_FOR_LABEL = {"GOOD": "blue", "DEFECT": "green"}
PLATE_HUES = ((85, 112), (38, 85))  # blue, green plates (OpenCV hue), as pick_any
COLOUR_CLASSES = {
    "red": lambda H, S, V: ((H <= 8) | (H >= 165)) & (S >= 90) & (V >= 60),
    "orange": lambda H, S, V: (H >= 9) & (H <= 16) & (S >= 90) & (V >= 70),
    "yellow": lambda H, S, V: (H >= 17) & (H <= 38) & (S >= 45) & (V >= 90),
    "green": lambda H, S, V: (H >= 39) & (H <= 85) & (S >= 70) & (V >= 60),
    "blue": lambda H, S, V: (H >= 86) & (H <= 130) & (S >= 90) & (V >= 60),
    "white": lambda H, S, V: (S < 45) & (V >= 150),
}
COLS = ["view", "key", "run", "ep", "idx", "label", "plate", "success", "needs_review", "scan", "t", "frame",
        "u_x", "u_y", "status", "reason", "blob_colour", "blob_cx", "blob_cy", "blob_w", "blob_h", "blob_hue",
        "blob_dist", "side_px", "clipped", "jaw_frac", "close_offset", "close_verdict", "close_bbox",
        "close_hue_diff", "close_hue_mismatch", "crop_path"]


def jaw_mask():
    m = cv2.imread(JAW_PNG, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise SystemExit("cannot read " + JAW_PNG)
    return m > 0


def blobs(img, jaw, min_area=2500, max_area=60000):
    """Brick-like blobs in a wrist frame (pick_any.blobs logic; jaw = jaw mask as a bool array, or None)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if jaw is None:
        jaw = np.zeros(H.shape, bool)
    plate = np.zeros(H.shape, bool)
    for lo, hi in PLATE_HUES:
        plate |= (H >= lo) & (H < hi) & (S >= 70) & (V >= 70)
    out = []
    for cname, fn in COLOUR_CLASSES.items():
        fg = fn(H, S, V) & ~jaw
        fg8 = cv2.morphologyEx(fg.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        fg8 = cv2.morphologyEx(fg8, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
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
            out.append({"centre": [round(float(cx), 1), round(float(cy), 1)],
                        "size": [round(float(sw), 1), round(float(sh), 1)],
                        "short": round(float(min(sw, sh)), 1), "area": a, "yaw": round(float(yaw), 1),
                        "aspect": round(float(max(sw, sh) / max(1.0, min(sw, sh))), 2),
                        "long_vertical": bool(abs(lng[1]) >= abs(lng[0])), "plate_frac": round(plate_frac, 2),
                        "border": bool(border), "hue": int(np.median(H[m])), "sat": int(np.median(S[m])),
                        "val": int(np.median(V[m])), "colour": cname})
    return out


def mask_jaws(img):
    """sort_loop.mask_jaws logic: black out red components that touch the top border."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = (((H <= 8) | (H >= 170)) & (S >= 100) & (V >= 50)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(red)
    out = img.copy()
    for i in range(1, n):
        if st[i, 1] <= 3 and st[i, 4] > 800:
            out[cv2.dilate((lab == i).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0] = 0
    return out


def close_pre_verdict(img, want_hue=None):
    """sort_any.close_pre_verdict logic: DET2 close tape rule for the brick under the jaws (None if no jaw tips)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = (((H <= 8) | (H >= 170)) & (S >= 100) & (V >= 50)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(red)
    tips = []
    for i in range(1, n):
        if st[i, 1] <= 3 and st[i, 4] > 800:
            ys, xs = np.nonzero(lab == i)
            yb = ys.max()
            tips.append((float(xs[ys >= yb - 6].mean()), float(yb)))
    if not tips:
        return None
    tips.sort(key=lambda t: -t[1])
    mx = float(np.mean([t[0] for t in tips[:2]]))
    my = float(max(t[1] for t in tips[:2])) + 40.0
    try:
        seg = DET2.segment_close(mask_jaws(img))
    except Exception as e:  # noqa: BLE001
        return {"verdict": "UNCERTAIN", "reasons": ["det2_error %r" % (e,)]}
    best = None
    for rec in seg["bricks"]:
        cx, cy = rec["centre_px"]
        d = math.hypot(cx - mx, cy - my)
        if best is None or d < best[0]:
            best = (d, rec)
    if best is None or best[0] > 130:
        return {"verdict": "UNCERTAIN", "reasons": ["no_brick_under_jaws_in_close_view"], "jaw_mid": [mx, my]}
    rec = best[1]
    verdict, reasons = DET2.triage(rec, DET2.load_thresholds())
    x, y, w, h = rec["bbox"]
    out = {"verdict": verdict, "reasons": reasons, "colour": rec.get("colour"), "tape_score": rec.get("tape_score"),
           "bbox": [int(x), int(y), int(w), int(h)], "dist": round(best[0], 1), "jaw_mid": [round(mx), round(my)],
           "hue": int(rec.get("hsv_med", [0])[0])}
    if want_hue is not None:
        dh = abs(out["hue"] - want_hue)
        dh = min(dh, 180 - dh)
        out["hue_diff"] = int(dh)
        if dh > 10:
            out["verdict_raw"] = verdict
            out["verdict"] = "UNCERTAIN"
            out["reasons"] = list(reasons) + ["close_brick_hue_mismatch_%d" % int(dh)]
    return out


def load_usable():
    usable, why, counts = {}, {}, {}
    for ep_json in sorted(glob.glob(os.path.join(LOGS_SORT, "*", "ep_*", "episode.json"))):
        ep_dir = os.path.dirname(ep_json)
        run = os.path.basename(os.path.dirname(ep_dir))
        key = run + "/" + os.path.basename(ep_dir)
        try:
            with open(ep_json, encoding="utf-8") as fh:
                ep = json.load(fh)
        except (OSError, ValueError):
            reason = "unreadable"
        else:
            label = ep.get("label")
            plate = ep.get("plate")
            if label not in PLATE_FOR_LABEL:
                reason = "no GOOD/DEFECT label"
            elif ep.get("success") is False:
                reason = "success false"
            elif ep.get("needs_review") is True:
                reason = "needs_review true"
            elif plate is not None and plate != PLATE_FOR_LABEL[label]:
                reason = "plate %s on %s" % (plate, label)
            else:
                usable[key] = {"run": run, "ep": os.path.basename(ep_dir), "label": label,
                               "plate": plate or "",
                               "success": "true" if ep.get("success") is True else "missing",
                               "needs_review": "absent" if "needs_review" not in ep else str(ep["needs_review"])}
                continue
        why[key] = reason
        counts[reason] = counts.get(reason, 0) + 1
    return usable, why, counts


def ffprobe_start():
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=start_time", "-of",
                        "default=nw=1:nk=1", CLOSE_MKV], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    return float(p.stdout.decode().strip())


def close_frame(t, start):
    off = float(t) - start
    cmd = ["ffmpeg", "-v", "error", "-ss", "%.3f" % off, "-i", CLOSE_MKV, "-frames:v", "1", "-f", "image2pipe",
           "-vcodec", "png", "pipe:1"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    if p.returncode != 0 or not p.stdout:
        return None
    return cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)


def wrist_match(img, u, jaw):
    bl = blobs(img, jaw, MIN_AREA, MAX_AREA)
    if not bl:
        return None, "skip_no_blob"
    best = min(bl, key=lambda b: math.hypot(b["centre"][0] - u[0], b["centre"][1] - u[1]))
    d = math.hypot(best["centre"][0] - u[0], best["centre"][1] - u[1])
    if not d < NEAR_PX:
        return None, "skip_far_blob"
    best = dict(best)
    best["dist"] = round(d, 1)
    return best, "ok"


def wrist_crop(img, blob, jaw):
    H, W = img.shape[:2]
    cx, cy = blob["centre"]
    side = SIDE_K * max(blob["size"])
    x0, y0 = cx - side / 2.0, cy - side / 2.0
    x1, y1 = cx + side / 2.0, cy + side / 2.0
    clipped = x0 < 0 or y0 < 0 or x1 > W or y1 > H
    xa, ya = max(0, int(round(x0))), max(0, int(round(y0)))
    xb, yb = min(W, int(round(x1))), min(H, int(round(y1)))
    if xb <= xa or yb <= ya:
        return None, clipped, 0.0, "skip_empty"
    jaw_frac = float(jaw[ya:yb, xa:xb].mean())
    if jaw_frac > MAX_JAW_FRAC:
        return None, clipped, jaw_frac, "skip_jaw_frac"
    crop = cv2.resize(img[ya:yb, xa:xb], (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    return crop, clipped, jaw_frac, "ok"


def close_crop(img, bbox):
    H, W = img.shape[:2]
    x, y, w, h = bbox
    crop = img[max(0, y - PAD):min(H, y + h + PAD), max(0, x - PAD):min(W, x + w + PAD)]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)


def save_crop_png(out, view, base, crop):
    d = os.path.join(out, "crops", view, "%s_%s" % (base["run"], base["ep"]))
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "%03d.png" % base["idx"])
    cv2.imwrite(path, crop)
    return path


def main():
    ap = argparse.ArgumentParser(description="Brick-centred crops from every associated frame (anom3 round 3).")
    ap.add_argument("--limit", type=int, default=0, help="only the first N association entries (testing)")
    ap.add_argument("--views", default="wrist_multi,close_multi")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    cv2.setNumThreads(4)
    views = [v for v in args.views.split(",") if v]
    usable, why, skipped = load_usable()
    print("usable episodes", len(usable), sorted(usable), flush=True)
    print("skipped episodes", skipped, flush=True)
    with open(ASSOC, encoding="utf-8") as fh:
        entries = json.load(fh)
    if args.limit:
        entries = entries[: args.limit]
    shutil.rmtree(os.path.join(args.out, "crops"), ignore_errors=True)
    jaw = jaw_mask()
    start = ffprobe_start()
    print("close recording start_time %.3f" % start, flush=True)
    rows = []
    t0 = time.time()
    for i, e in enumerate(entries):
        key = str(e.get("ep", ""))
        u = e.get("u") or [float("nan"), float("nan")]
        base = {"key": key, "idx": i, "frame": e.get("frame", ""), "t": e.get("t", ""),
                "scan": int(bool(e.get("scan"))), "u_x": round(float(u[0]), 1), "u_y": round(float(u[1]), 1)}
        ep = usable.get(key)
        if ep is None:
            for v in views:
                rows.append(dict(base, view=v, status="skip_episode",
                                 reason=why.get(key, "not in the run folders")))
            continue
        base.update(run=ep["run"], ep=ep["ep"], label=ep["label"], plate=ep["plate"], success=ep["success"],
                    needs_review=ep["needs_review"])
        fp = e.get("frame") or ""
        if fp and not os.path.isabs(fp):
            fp = os.path.join(R, fp)
        img = cv2.imread(fp, cv2.IMREAD_COLOR) if fp else None
        if img is None or img.shape[:2] != jaw.shape:
            for v in views:
                rows.append(dict(base, view=v, status="skip_frame", reason="frame unreadable or not 480x640"))
            continue
        blob, bstat = wrist_match(img, u, jaw)
        if "wrist_multi" in views:
            row = dict(base, view="wrist_multi")
            if blob is None:
                row.update(status=bstat, reason=("no blob within %d px of u" % NEAR_PX)
                           if bstat == "skip_far_blob" else "no blob in frame")
            else:
                crop, clipped, jf, cstat = wrist_crop(img, blob, jaw)
                row.update(blob_colour=blob["colour"], blob_cx=blob["centre"][0], blob_cy=blob["centre"][1],
                           blob_w=blob["size"][0], blob_h=blob["size"][1], blob_hue=blob["hue"],
                           blob_dist=blob["dist"], side_px=round(SIDE_K * max(blob["size"]), 1),
                           clipped=int(clipped), jaw_frac=round(jf, 3))
                if crop is None:
                    row.update(status=cstat, reason=("more than 20 percent inside the jaw mask"
                                                     if cstat == "skip_jaw_frac" else cstat))
                else:
                    row.update(status="ok", crop_path=save_crop_png(args.out, "wrist_multi", base, crop))
            rows.append(row)
        if "close_multi" in views:
            row = dict(base, view="close_multi")
            if base["scan"]:
                row.update(status="skip_scan", reason="scan entry: no close frame at this epoch")
            elif not e.get("t"):
                row.update(status="skip_no_t", reason="no epoch")
            elif blob is None:
                row.update(status="skip_no_hue_ref", reason="no wrist blob, so no brick hue to match in close view")
            else:
                row.update(close_offset=round(float(e["t"]) - start, 3))
                cimg = close_frame(e["t"], start)
                if cimg is None:
                    row.update(status="skip_no_close_frame", reason="ffmpeg returned no frame")
                else:
                    v = close_pre_verdict(cimg, want_hue=blob["hue"])
                    if v is None or v.get("bbox") is None:
                        why_no = (v or {}).get("reasons", ["jaws not in close view"])
                        row.update(status="skip_no_bbox", reason=str(why_no[0]),
                                   close_verdict=(v or {}).get("verdict", ""))
                    else:
                        crop = close_crop(cimg, v["bbox"])
                        mism = any(str(r).startswith("close_brick_hue_mismatch") for r in v.get("reasons", []))
                        if crop is None:
                            row.update(status="skip_empty", reason="empty crop")
                        else:
                            row.update(status="ok_hue_mismatch" if mism else "ok",
                                       crop_path=save_crop_png(args.out, "close_multi", base, crop),
                                       close_verdict=v["verdict"],
                                       close_bbox=" ".join(str(x) for x in v["bbox"]),
                                       close_hue_diff=v.get("hue_diff", ""), close_hue_mismatch=int(mism))
            rows.append(row)
        print("entry %d %s done" % (i, key), flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "manifest.tsv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=COLS, delimiter="\t", extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    counts = {}
    for r in rows:
        k = (r["view"], r.get("key", ""), r.get("status", ""))
        counts[k] = counts.get(k, 0) + 1
    for k in sorted(counts):
        print(k, counts[k], flush=True)
    print("rows", len(rows), "seconds %.1f" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
