import glob
import os

import cv2
import numpy as np

OUT = "/home/intel/intel-junction-2026/work/P2-R8-PICK/V3/out"
EPS = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 14, 15]
X0, X1, Y0, Y1 = 300, 640, 20, 330
SCALE = 1.5


def grasp_path(ei):
    hits = glob.glob(os.path.join(OUT, "frames", f"ep{ei:02d}_grasp_t*.png"))
    return hits[0]


def zoom(path, label):
    im = cv2.imread(path)
    crop = im[Y0:Y1, X0:X1]
    big = cv2.resize(crop, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    for x in range(20 * ((X0 + 19) // 20), X1, 20):
        px = int((x - X0) * SCALE)
        col = (0, 255, 255) if x % 100 == 0 else (0, 160, 160)
        cv2.line(big, (px, 0), (px, big.shape[0] - 1), col, 1)
        if x % 40 == 0:
            cv2.putText(big, str(x), (px + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    for y in range(20 * ((Y0 + 19) // 20), Y1, 20):
        py = int((y - Y0) * SCALE)
        col = (0, 255, 255) if y % 100 == 0 else (0, 160, 160)
        cv2.line(big, (0, py), (big.shape[1] - 1, py), col, 1)
        if y % 40 == 0:
            cv2.putText(big, str(y), (2, py + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(big, label, (6, big.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return big


def main():
    pairs = []
    for ei in EPS:
        s = zoom(os.path.join(OUT, "frames", f"ep{ei:02d}_start_t0.png"), f"ep{ei} START t=0")
        g = zoom(grasp_path(ei), f"ep{ei} GRASP")
        pairs.append(np.hstack([s, np.full((s.shape[0], 6, 3), 255, np.uint8), g]))
    for k in range(0, len(pairs), 2):
        block = pairs[k:k + 2]
        sheet = block[0] if len(block) == 1 else np.vstack([block[0], np.full((8, block[0].shape[1], 3), 255, np.uint8), block[1]])
        cv2.imwrite(os.path.join(OUT, "sheets", f"zoom_{k // 2 + 1}.png"), sheet)
        print("ZOOM", k // 2 + 1, [e for e in EPS[k:k + 2]], sheet.shape, flush=True)


if __name__ == "__main__":
    main()
