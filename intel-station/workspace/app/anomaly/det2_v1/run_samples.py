"""Segment every harvested sample (dataset 6d6bdce3 only; a4ba6ede is a duplicate copy) and write
bricks.json, crops, annotated frames and contact sheets under work/P2-B-DET2/E2/.

usage: python3 -I run_samples.py
"""
import json
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inspector as I  # noqa: E402
from common import CLOSE, TOP, WORK  # noqa: E402

SAMPLES = os.path.join(WORK, "samples", "samples.json")
OUT = os.path.join(WORK, "seg")
SHEETS = os.path.join(WORK, "sheets")
DS_KEEP = "6d6bdce3"


def tile(img, bbox, size, pad, lines):
    x, y, w, h = bbox
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
    crop = img[y0:y1, x0:x1].copy()
    if crop.size == 0:
        crop = np.zeros((size, size, 3), np.uint8)
    scale = size / max(crop.shape[:2])
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
                      interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((size + 34, size, 3), np.uint8)
    canvas[:crop.shape[0], :crop.shape[1]] = crop
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (2, size + 12 + 11 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return canvas


def grid(tiles, cols):
    if not tiles:
        return np.zeros((40, 400, 3), np.uint8)
    th, tw = tiles[0].shape[:2]
    rows = math.ceil(len(tiles) / cols)
    out = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        out[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    return out


COLOURS = {"2x2": (0, 255, 0), "not_2x2": (0, 0, 255), "unknown": (0, 255, 255)}


def annotate(img, bricks, title):
    vis = img.copy()
    for b in bricks:
        box = cv2.boxPoints(((b["centre_px"][0], b["centre_px"][1]), tuple(b["footprint_px"]), b["angle_deg"]))
        cv2.polylines(vis, [box.astype(np.int32)], True, COLOURS.get(b["shape"], (255, 0, 255)), 1)
        cv2.putText(vis, str(b["k"]), (int(b["centre_px"][0]) + 6,
                    int(b["centre_px"][1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def main():
    samples = json.load(open(SAMPLES))
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(SHEETS, exist_ok=True)
    top_tiles, close_tiles, annot_top, annot_close, all_bricks, frames = [], [], [], [], [], []
    for sid in sorted(samples):
        rec = samples[sid]
        if rec["ds8"] != DS_KEEP:
            continue
        top_img = cv2.imread(rec["views"][TOP]["png"])
        close_img = cv2.imread(rec["views"][CLOSE]["png"])
        frame = {"sample": sid, "episode": rec["episode"], "kind": rec["kind"], "frame": rec["frame"],
                 "home_dist": rec["home_dist"], "hands_top": rec["views"][TOP]["hands_flag"],
                 "hands_close": rec["views"][CLOSE]["hands_flag"]}
        seg_t = I.segment_top(top_img)
        seg_c = I.segment_close(close_img)
        frame.update({"hands_blobs_top": seg_t["hands"], "plates_rejected_top": seg_t["plates_rejected"],
                      "n_top": len(seg_t["bricks"]), "n_close": len(seg_c["bricks"]),
                      "partial_close": seg_c["partial_at_border"], "rejected_close": seg_c["rejected"]})
        for view, seg, img in (("top", seg_t, top_img), ("close", seg_c, close_img)):
            ser = TOP if view == "top" else CLOSE
            for k, b in enumerate(seg["bricks"]):
                b["k"] = k
                b["id"] = f"{sid}_{ser.split()[-1]}_{k}"
                b["view"] = view
                b["sample"] = sid
                b["episode"] = rec["episode"]
                b["kind"] = rec["kind"]
                m = b.pop("_mask")
                if view == "close":
                    desc = I.patch_desc(img, m, b["bbox"])
                    np.save(os.path.join(OUT, f"{b['id']}_patches.npy"), desc)
                    b["n_patches"] = int(desc.shape[0])
                    close_tiles.append(tile(img, b["bbox"], 160, 5, [
                        f"{b['sample'][9:]} k{b['k']}", f"{b['colour']} {b['shape']} a{b['aspect']:.2f}",
                        f"off{b['tape_features']['off_frac']:.2f} gl{b['tape_features']['glare_frac']:.2f}"]))
                else:
                    top_tiles.append(tile(img, b["bbox"], 96, 4, [
                        f"{b['sample'][9:]} k{b['k']}", f"{b['colour']} {b['shape']} a{b['aspect']:.2f}",
                        f"A{b['area']} ang{b['angle_deg']:.0f}"]))
                all_bricks.append(b)
            (annot_top if view == "top" else annot_close).append(
                annotate(img, seg["bricks"], f"{sid} {view} n={len(seg['bricks'])}"))
        frames.append(frame)
    cv2.imwrite(os.path.join(SHEETS, "top_sheet.jpg"), grid(top_tiles, 10), [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(os.path.join(SHEETS, "close_sheet.jpg"), grid(close_tiles, 8), [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(os.path.join(SHEETS, "annot_top.jpg"), grid([cv2.resize(a, (320, 240)) for a in annot_top], 4),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(os.path.join(SHEETS, "annot_close.jpg"), grid([cv2.resize(a, (320, 240)) for a in annot_close], 4),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    json.dump({"frames": frames, "bricks": [{k: v for k, v in b.items() if k != "_mask"} for b in all_bricks]},
              open(os.path.join(OUT, "bricks.json"), "w"), indent=1, default=str)
    from collections import Counter
    print("frames", len(frames), "bricks top", sum(b["view"] == "top" for b in all_bricks),
          "bricks close", sum(b["view"] == "close" for b in all_bricks))
    for view in ("top", "close"):
        bs = [b for b in all_bricks if b["view"] == view]
        print(view, "colour", dict(Counter(b["colour"] for b in bs)), "shape", dict(Counter(b["shape"] for b in bs)))
    print("sheets", sorted(os.listdir(SHEETS)))


if __name__ == "__main__":
    main()
