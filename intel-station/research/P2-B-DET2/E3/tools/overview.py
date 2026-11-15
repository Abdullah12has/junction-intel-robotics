"""P2-B-DET2 E3 overview grids of harvested frames (6d6bdce3 only). Run: envs/data-env/bin/python -I -B tools/overview.py"""
import json

import cv2
import numpy as np

W = "/home/intel/intel-junction-2026/work/P2-B-DET2/E3"


def grid(recs, out, cols=6, tw=320, th=240):
    ims = []
    for r in recs:
        im = cv2.resize(cv2.imread(r["png"]), (tw, th))
        cv2.putText(im, r["id"], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        ims.append(im)
    rows = (len(ims) + cols - 1) // cols
    canvas = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for i, im in enumerate(ims):
        y, x = (i // cols) * th, (i % cols) * tw
        canvas[y:y + th, x:x + tw] = im
    cv2.imwrite(out, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return canvas.shape


def main():
    idx = json.load(open(f"{W}/out/samples_index.json"))
    top = [r for r in idx if r["ds"] == "6d6bdce3" and r["cam"] == "top"]
    close = [r for r in idx if r["ds"] == "6d6bdce3" and r["cam"] == "close"]
    print("top", grid(top, f"{W}/out/overview_top.jpg"), len(top))
    print("close", grid(close, f"{W}/out/overview_close.jpg"), len(close))


if __name__ == "__main__":
    main()
