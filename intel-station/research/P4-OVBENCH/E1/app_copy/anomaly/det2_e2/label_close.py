"""E2 visual labels for close-view bricks (verified by zoom sheet work/P2-B-DET2/E2/sheets/zoom_close_ambiguous.jpg
and close_sheet.jpg) and episode-disjoint crop folders for PatchCore training/testing.

G   clean 2x2 (no tape; print such as '2' or 'O' is not tape)  -> good bank / test-good
T   tape visible (dark or brown/grey band or patch), 2x2        -> defect
NT  tape visible and not 2x2 (2x4)                              -> defect
X   merged cluster, cut by the border, or shape and appearance unclear -> excluded

usage: python3 -I label_close.py
"""
import json
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLOSE, WORK  # noqa: E402

LABELS = {
    ("6d6bdce3_e03_end", 1): "G", ("6d6bdce3_e03_end", 2): "G", ("6d6bdce3_e03_start", 1): "G",
    ("6d6bdce3_e04_start", 0): "G", ("6d6bdce3_e05_end", 0): "G", ("6d6bdce3_e06_start", 0): "G",
    ("6d6bdce3_e07_start", 1): "G", ("6d6bdce3_e07_start", 4): "G", ("6d6bdce3_e11_end", 0): "G",
    ("6d6bdce3_e11_end", 2): "G", ("6d6bdce3_e11_start", 0): "G", ("6d6bdce3_e12_start", 0): "G",
    ("6d6bdce3_e12_start", 3): "G", ("6d6bdce3_e13_start", 0): "G", ("6d6bdce3_e13_start", 1): "G",
    ("6d6bdce3_e13_start", 2): "G", ("6d6bdce3_e14_start", 0): "G", ("6d6bdce3_e14_start", 1): "G",
    ("6d6bdce3_e07_start", 2): "T", ("6d6bdce3_e07_start", 3): "T", ("6d6bdce3_e11_end", 3): "T",
    ("6d6bdce3_e11_end", 4): "T", ("6d6bdce3_e12_start", 2): "T", ("6d6bdce3_e09_end", 0): "T",
    ("6d6bdce3_e07_start", 0): "NT",
    ("6d6bdce3_e03_end", 0): "X", ("6d6bdce3_e03_start", 0): "X", ("6d6bdce3_e05_start", 0): "X",
    ("6d6bdce3_e06_end", 0): "X", ("6d6bdce3_e09_end", 1): "X", ("6d6bdce3_e10_end", 0): "X",
    ("6d6bdce3_e10_end", 1): "X", ("6d6bdce3_e11_end", 1): "X", ("6d6bdce3_e11_start", 1): "X",
    ("6d6bdce3_e12_start", 1): "X",
}
TRAIN_EPS = {3, 4, 5, 6}  # episode-disjoint: PatchCore fit on these; tested on 7, 9, 11-14


def main():
    d = json.load(open(os.path.join(WORK, "seg", "bricks.json")))
    samples = json.load(open(os.path.join(WORK, "samples", "samples.json")))
    root = os.path.join(WORK, "anomalib", "bricks_close")
    for sub in ("train/good", "test/good", "test/defect"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    rows, counts = [], {}
    for b in d["bricks"]:
        if b["view"] != "close":
            continue
        key = (b["sample"], b["k"])
        lab = LABELS.get(key)
        if lab is None:
            raise SystemExit(f"unlabelled close brick {key}")
        bid = b["id"]
        rows.append([bid, b["sample"], b["episode"], b["kind"], b["k"], lab, b["colour"], b["shape"],
                     str(b["aspect"]), str(b["tape_features"]["off_frac"]), str(b["tape_features"]["glare_frac"])])
        counts[lab] = counts.get(lab, 0) + 1
        if lab in ("G", "T", "NT"):
            img = cv2.imread(samples[b["sample"]]["views"][CLOSE]["png"])
            x, y, w, h = b["bbox"]
            pad = 6
            crop = img[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
            s = 256 / max(crop.shape[:2])
            crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
            canvas = np.zeros((256, 256, 3), np.uint8)
            canvas[:crop.shape[0], :crop.shape[1]] = crop
            if lab == "G":
                sub = "train/good" if b["episode"] in TRAIN_EPS else "test/good"
            else:
                sub = "test/defect"
            cv2.imwrite(os.path.join(root, sub, bid + ".png"), canvas)
    hdr = ["brick_id", "sample", "episode", "kind", "k", "label", "colour_auto", "shape_auto", "aspect",
           "off_frac", "glare_frac"]
    os.makedirs(os.path.join(WORK, "labels"), exist_ok=True)
    with open(os.path.join(WORK, "labels", "close_labels_E2.tsv"), "w") as f:
        f.write("\t".join(hdr) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print("labels", counts)
    for sub in ("train/good", "test/good", "test/defect"):
        print(sub, len(os.listdir(os.path.join(root, sub))))


if __name__ == "__main__":
    main()
