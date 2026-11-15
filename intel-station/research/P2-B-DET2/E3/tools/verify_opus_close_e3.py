"""Visual check of the Opus close labels (E3). One tile per Opus label used in the eval (29): the crop of the brick
my k points at, or the whole frame when no brick was segmented. Tiles carry an index (#n) that matches
out/opus_close_check_e3_index.tsv. Verdicts are written by hand afterwards.
Run: envs/data-env/bin/python -I -B work/P2-B-DET2/E3/tools/verify_opus_close_e3.py"""
import json
import os
import sys

import cv2
import numpy as np

WORK = "/home/intel/intel-junction-2026/work/P2-B-DET2/E3"
APP = "/home/intel/intel-junction-2026/app/anomaly/det2_e3"
sys.path.insert(0, APP)
import inspector  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX
TILE = 280
COLS = 5
PER_SHEET = 15


def fit(img):
    h, w = img.shape[:2]
    s = TILE / max(h, w)
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    out = np.zeros((TILE, TILE, 3), np.uint8)
    hh, ww = small.shape[:2]
    oy, ox = (TILE - hh) // 2, (TILE - ww) // 2
    out[oy:oy + hh, ox:ox + ww] = small
    return out


def tile(idx, row, img):
    im = fit(img)
    cv2.putText(im, f"#{idx}", (6, 34), FONT, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(im, f"#{idx}", (6, 34), FONT, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    bar = np.zeros((44, TILE, 3), np.uint8)
    tail = f"dg={row['dark_grey']}" if row.get("segmented") else "NOT SEGMENTED, frame shown"
    cv2.putText(bar, row["key"].replace("close/", ""), (4, 16), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, f"{row['label']} {tail}", (4, 36), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([im, bar])


def main():
    data = json.load(open(os.path.join(WORK, "out", "eval_close_opus_e3.json")))
    rows = [r for r in data["rows"] if r.get("used")]
    cache, tiles, index = {}, [], []
    for i, r in enumerate(rows, 1):
        fid = r["frame"]
        if fid not in cache:
            img = cv2.imread(f"{WORK}/frames/6d6bdce3_{fid}_243722064316.png")
            cache[fid] = (img, {x["id"]: x for x in inspector.scan(close_rgb=img)})
        img, recs = cache[fid]
        rec = recs.get(f"scan_close_{r['k']}") if r.get("k") is not None else None
        if rec is not None and rec.get("bbox") is not None:
            x, y, w, h = rec["bbox"]
            p = 6
            crop = img[max(0, y - p):y + h + p, max(0, x - p):x + w + p]
        else:
            crop = img
        tiles.append(tile(i, r, crop))
        index.append(f"{i}\t{r['key']}\t{r['label']}\t{r.get('segmented')}\t{r['k']}\t{r.get('note', '')}")
    out_dir = os.path.join(WORK, "sheets")
    th = TILE + 44
    for n, start in enumerate(range(0, len(tiles), PER_SHEET)):
        chunk = tiles[start:start + PER_SHEET]
        rows_n = (len(chunk) + COLS - 1) // COLS
        canvas = np.zeros((rows_n * th, COLS * TILE, 3), np.uint8)
        for j, t in enumerate(chunk):
            r0, c0 = (j // COLS) * th, (j % COLS) * TILE
            canvas[r0:r0 + th, c0:c0 + TILE] = t
        path = os.path.join(out_dir, f"opus_close_check_{'ab'[n]}.jpg")
        cv2.imwrite(path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(path, canvas.shape)
    with open(os.path.join(WORK, "out", "opus_close_check_e3_index.tsv"), "w") as fh:
        fh.write("index\tkey\tlabel\tsegmented\tk\topus_note\n")
        fh.write("\n".join(index) + "\n")
    print("tiles", len(tiles))


if __name__ == "__main__":
    main()
