"""Contact sheets for Opus (E3): brick crops with ids, suggested colour, shape and tape flag; annotated frame grids.
Run: envs/data-env/bin/python -I -B app/anomaly/det2_e3/sheets.py"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import det2  # noqa: E402

WORK = "/home/intel/intel-junction-2026/work/P2-B-DET2/E3"
SHEETS = f"{WORK}/sheets"
COL = {"red": (0, 0, 255), "orange": (0, 140, 255), "yellow": (0, 255, 255), "white": (255, 255, 255),
       "blue": (255, 0, 0), "green": (0, 255, 0), "grey": (160, 160, 160), "other": (255, 0, 255)}


def tape_score(rec, cal):
    p = cal["views"][rec["view"]]["tape_stats"]
    return max(rec["tape_feats"][f] / max(p[f]["p95"], 1e-6) for f in det2.TAPE_FEATS)


def crop_label(rec, score):
    sh = rec.get("shape", "")
    return f"{rec['colour'][:4]} {sh} t{score:.2f}"


def crops_sheet(recs, cal, out, size=96, cols=12):
    tiles = []
    for r in recs:
        im = cv2.imread(r["png"])
        x, y, w, h = r["bbox"]
        p = 6
        cr = im[max(0, y - p):min(im.shape[0], y + h + p), max(0, x - p):min(im.shape[1], x + w + p)]
        cr = cv2.resize(cr, (size, size), interpolation=cv2.INTER_AREA)
        bar = np.zeros((30, size, 3), np.uint8)
        cv2.putText(bar, f"e{r['ep']:02d} {r['which'][0]} {r['serial'][-3:]} k{r['k']}", (2, 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1, cv2.LINE_AA)
        sc = tape_score(r, cal)
        cv2.putText(bar, crop_label(r, sc), (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                    (0, 0, 255) if sc > 1.0 else (200, 200, 200), 1, cv2.LINE_AA)
        tile = np.vstack([cr, bar])
        cv2.rectangle(tile, (0, 0), (size - 1, size + 29), COL.get(r["colour"], (255, 0, 255)), 1)
        tiles.append(tile)
    if not tiles:
        return None
    th, tw = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for i, t in enumerate(tiles):
        canvas[(i // cols) * th:(i // cols + 1) * th, (i % cols) * tw:(i % cols + 1) * tw] = t
    cv2.imwrite(out, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return canvas.shape


def frames_sheet(recs, allrecs, out, cols=4, tw=320, th=240):
    tiles = []
    by = {}
    for r in allrecs:
        by.setdefault(r["png"], []).append(r)
    for fr in recs:
        im = cv2.imread(fr["png"])
        for r in by.get(fr["png"], []):
            x, y, w, h = r["bbox"]
            cv2.rectangle(im, (x, y), (x + w, y + h), COL.get(r["colour"], (255, 0, 255)), 2)
            cv2.putText(im, str(r["k"]), (x, max(12, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(im, fr["id"][-26:] if "id" in fr else os.path.basename(fr["png"])[:-4], (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(cv2.resize(im, (tw, th)))
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for i, t in enumerate(tiles):
        canvas[(i // cols) * th:(i // cols + 1) * th, (i % cols) * tw:(i % cols + 1) * tw] = t
    cv2.imwrite(out, canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return canvas.shape


def main():
    cal = det2.load_cal()
    allrecs = json.load(open(f"{WORK}/out/bricks_build.json"))
    idx = json.load(open(f"{WORK}/out/samples_index.json"))
    os.makedirs(SHEETS, exist_ok=True)
    paths = []
    for view in ("top", "close"):
        rv = [r for r in allrecs if r["view"] == view]
        out_c = f"{SHEETS}/{view}_bricks_crops.jpg"
        print("crops", view, len(rv), crops_sheet(rv, cal, out_c, size=96 if view == "top" else 128,
                                                   cols=12 if view == "top" else 10))
        paths.append(out_c)
        frames = [r for r in idx if r["ds"] == "6d6bdce3" and r["cam"] == view]
        out_f = f"{SHEETS}/{view}_frames_annotated.jpg"
        print("frames", view, frames_sheet(frames, rv, out_f))
        paths.append(out_f)
    with open(f"{WORK}/out/sheet_paths.txt", "w") as fh:
        fh.write("\n".join(paths) + "\n")
    print("\n".join(paths))


if __name__ == "__main__":
    main()
