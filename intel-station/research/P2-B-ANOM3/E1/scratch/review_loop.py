"""Review sheets for every usable loop crop (anom3 E1 scratch, not part of the app).

Writes scratch/review/sheet_VIEW.png (one tile per crop: index, split, label, group, success state, v000 score and
prediction), scratch/review/loop_crops_v000.tsv, and enlarged PNGs for every v000 disagreement.
"""
import csv
import os
import sys

HERE = "/home/intel/intel-junction-2026/app/anomaly/anom3"
sys.path.insert(0, HERE)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import common  # noqa: E402
import infer  # noqa: E402

OUT = "/home/intel/intel-junction-2026/work/P2-B-ANOM3/E1/scratch/review"
BASE = os.path.join(HERE, "models", "v000")


def tile(img, box=360, head=66):
    h, w = img.shape[:2]
    s = min(box / float(w), box / float(h))
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    canvas = np.full((box + head, box, 3), 32, np.uint8)
    y0, x0 = head + (box - nh) // 2, (box - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = im
    return canvas


def put(canvas, lines):
    for i, text in enumerate(lines):
        cv2.putText(canvas, text, (6, 18 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    os.makedirs(OUT, exist_ok=True)
    items, excluded = common.load("episodes")
    print("excluded", excluded)
    table = []
    for view in common.VIEWS:
        vi = sorted([it for it in items if it["view"] == view],
                    key=lambda it: (it["split"], it["label"], it["group"]))
        if not vi:
            print(view, "no crops")
            continue
        thr, scores = None, [None] * len(vi)
        if os.path.isfile(os.path.join(BASE, view, "model.xml")):
            scores, dev = infer.score_files(os.path.join(BASE, view), [it["path"] for it in vi], infer.DEVICES)
            thr = infer.read_threshold(os.path.join(BASE, view))
            print(view, "v000 device", dev, "threshold", thr)
        tiles = []
        for n, (it, sc) in enumerate(zip(vi, scores), 1):
            img = cv2.imread(it["path"], cv2.IMREAD_COLOR)
            if img is None:
                raise SystemExit("cannot read " + it["path"])
            pred = None if sc is None else ("DEFECT" if sc > thr else "GOOD")
            agree = "" if pred is None else ("agree" if pred == it["label"] else "DISAGREE")
            table.append([view, it["split"], it["label"], it["group"], it["plate"], it["success"], it["source"],
                          it["path"], img.shape[1], img.shape[0], "" if sc is None else "%.2f" % sc,
                          pred or "", agree])
            print("%-13s %d %-8s %-8s %-22s succ=%-8s %-7s %s  %s %s" % (
                view, n, it["split"], it["label"], it["group"], it["success"],
                "" if sc is None else "%.1f" % sc, pred or "", agree, os.path.basename(it["path"])))
            t = tile(img)
            put(t, ["%d %s %s" % (n, it["split"], it["label"]),
                    "%s %s" % (it["group"], it["source"].split(":")[-1]),
                    "succ=%s plate=%s %dx%d" % (it["success"], it["plate"], img.shape[1], img.shape[0]),
                    ("v000 %.1f %s %s" % (sc, pred, agree)) if sc is not None else "no v000 model"])
            tiles.append(t)
            if agree == "DISAGREE":
                big = cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(OUT, "disagree_%s_%d_%s.png" % (view, n, it["group"].replace("/", "_"))), big)
        cols = min(4, len(tiles))
        nrows = (len(tiles) + cols - 1) // cols
        H, W = tiles[0].shape[:2]
        sheet = np.zeros((nrows * H, cols * W, 3), np.uint8)
        for k, t in enumerate(tiles):
            r, c = divmod(k, cols)
            sheet[r * H:(r + 1) * H, c * W:(c + 1) * W] = t
        cv2.imwrite(os.path.join(OUT, "sheet_%s.png" % view), sheet)
    with open(os.path.join(OUT, "loop_crops_v000.tsv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh, delimiter="\t")
        wr.writerow(["view", "split", "label", "group", "plate", "success", "source", "path", "w", "h",
                     "v000_score", "v000_pred", "agree"])
        wr.writerows(table)
    print("rows", len(table))


if __name__ == "__main__":
    main()
