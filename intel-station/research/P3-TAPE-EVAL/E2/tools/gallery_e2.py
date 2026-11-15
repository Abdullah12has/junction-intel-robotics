#!/usr/bin/env python3
"""Crop every decided error (base run) from the Studio table into gallery/, 2x scale, with a label file.
usage: python3 -I gallery_e2.py <E2 folder>"""
import sys, os, csv
import cv2
E2 = sys.argv[1]
FR = "/home/intel/intel-junction-2026/work/P3-LABEL-STUDIO/E1/frames"
G = os.path.join(E2, "gallery")
os.makedirs(G, exist_ok=True)
rows = list(csv.DictReader(open(os.path.join(E2, "base", "studio.tsv")), delimiter="\t"))
out = []
n = 0
for r in rows:
    if r["pred"] in ("yes", "no") and r["pred"] != r["truth"]:
        n += 1
        img = cv2.imread(os.path.join(FR, r["file"]))
        x, y, w, h = [int(float(v)) for v in r["bbox"].strip("[]").split(",")]
        crop = img[max(0, y - 12):y + h + 12, max(0, x - 12):x + w + 12]
        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        name = f"studio_err_{n:02d}_{r['unit']}_{r['stage']}_{r['colour']}_truth{r['truth']}_pred{r['pred']}.png"
        cv2.imwrite(os.path.join(G, name), crop)
        out.append(f"{name}\t{r['file']}\t{r['entry']}\t{r['reasons']}\ttape_score={r['tape_score']}\tbbox={r['bbox']}")
open(os.path.join(G, "index.tsv"), "w").write("gallery_file\tsource_frame\tentry\treasons\ttape_score\tbbox\n" + "\n".join(out) + "\n")
print(n, "errors cropped")
print("\n".join(out))
