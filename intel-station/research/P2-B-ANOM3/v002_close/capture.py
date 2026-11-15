"""Capture one top/close frame pair and harvest labelled close-camera brick crops for anom3 v002 (pre_close view).

Labels: Hassan, 15 Nov 2026 ~03:30Z: the yellow and the blue brick are taped (DEFECT); every other colour is clean.
A crop is kept only when det2_f2 sees the brick whole: not merged, not a split piece, not touching the frame border.
Run: nice -n 19 env OMP_NUM_THREADS=4 envs/anomalib-env/bin/python -I -B work/P2-B-ANOM3/v002_close/capture.py
"""
import json
import os
import subprocess
import sys
import time

import cv2

R = "/home/intel/intel-junction-2026"
ROOT = R + "/work/P2-B-ANOM3/v002_close"
sys.path.insert(0, R + "/app/anomaly/det2_f2")
import inspector as I  # noqa: E402

DEFECT_COLOURS = {"yellow", "blue"}
PAD = 12  # as sort_loop.save_crop and anom3 extract_crops.close_crop


def grab(dev, path):
    subprocess.run(["timeout", "20", "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "v4l2", "-video_size",
                    "640x480", "-framerate", "30", "-i", dev, "-vf", "select=gte(n\\,20)", "-frames:v", "1", "-y",
                    path], check=True)


def next_dir():
    caps = sorted(d for d in os.listdir(ROOT + "/captures") if d.startswith("cap"))
    out = "%s/captures/cap%02d" % (ROOT, len(caps) + 1)
    os.makedirs(out + "/crops")
    return out


def harvest(out, top, close, res=None):
    """Save the frame pair, the kept crops, labels.json and annotated_close.jpg; print one summary line."""
    n = int(os.path.basename(out)[3:])
    cv2.imwrite(out + "/top.jpg", top)
    cv2.imwrite(out + "/close.jpg", close)
    res = res if res is not None else I.scan_full(top, close)
    H, W = close.shape[:2]
    ann, recs = close.copy(), []
    for i, b in enumerate(res["bricks"]):
        if b.get("view") != "close":
            continue
        x, y, w, h = b["bbox"]
        reject = [k for k in ("merged", "split") if b.get(k)]
        if x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2:
            reject.append("border")
        label = "DEFECT" if b.get("colour") in DEFECT_COLOURS else "GOOD"
        rec = {"id": i, "colour": b.get("colour"), "label": label, "bbox": b["bbox"], "det2": b.get("verdict"),
               "reasons": b.get("reasons"), "reject": reject}
        if not reject:
            rec["crop"] = "crops/%02d_%s_%s.png" % (i, b.get("colour"), label)
            cv2.imwrite(out + "/" + rec["crop"], close[max(0, y - PAD):min(H, y + h + PAD), max(0, x - PAD):min(W, x + w + PAD)])
        c = (128, 128, 128) if reject else ((0, 0, 255) if label == "DEFECT" else (0, 200, 0))
        cv2.rectangle(ann, (x, y), (x + w, y + h), c, 2)
        cv2.putText(ann, "%d %s %s%s" % (i, b.get("colour"), label, (" X:" + ",".join(reject)) if reject else ""),
                    (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        recs.append(rec)
    cv2.imwrite(out + "/annotated_close.jpg", ann)
    json.dump({"capture": n, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "label_source": "hassan",
               "defect_colours": sorted(DEFECT_COLOURS), "bricks": recs}, open(out + "/labels.json", "w"), indent=1)
    kept = [r for r in recs if not r["reject"]]
    print("cap%02d: %d close detections, kept %d (GOOD %d, DEFECT %d)" % (
        n, len(recs), len(kept), sum(r["label"] == "GOOD" for r in kept), sum(r["label"] == "DEFECT" for r in kept)))
    for r in recs:
        print(" ", r["id"], r["colour"], r["label"], ("REJECT " + ",".join(r["reject"])) if r["reject"] else "kept")
    sys.stdout.flush()


def main():
    busy = subprocess.run(["fuser", "/dev/video8", "/dev/video14"], capture_output=True, text=True).stdout.strip()
    if busy:
        sys.exit("cameras busy: " + busy)
    out = next_dir()
    grab("/dev/video8", out + "/top.jpg")
    grab("/dev/video14", out + "/close.jpg")
    harvest(out, cv2.imread(out + "/top.jpg"), cv2.imread(out + "/close.jpg"))


if __name__ == "__main__":
    main()
