import glob
import os
import subprocess

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = "/home/intel/intel-junction-2026/data/studio-datasets"
DS = "6d6bdce3-b382-478d-9697-fb9e573b97af"
BASE = os.path.join(ROOT, DS)
OUT = "/home/intel/intel-junction-2026/work/P2-R8-PICK/V3/out"
KEY = "observation.images.realsense 243622061421"
FPS = 30.0
EPS = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 14, 15]
X0, X1, Y0, Y1 = 300, 640, 20, 330
SCALE = 1.2


def load(pattern):
    files = sorted(glob.glob(os.path.join(BASE, pattern)))
    return pa.concat_tables([pq.read_table(f) for f in files], promote_options="default")


def zoom(im, label):
    crop = im[Y0:Y1, X0:X1]
    big = cv2.resize(crop, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    for x in range(20 * ((X0 + 19) // 20), X1, 20):
        px = int((x - X0) * SCALE)
        cv2.line(big, (px, 0), (px, big.shape[0] - 1), (0, 160, 160), 1)
        if x % 40 == 0:
            cv2.putText(big, str(x), (px + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    for y in range(20 * ((Y0 + 19) // 20), Y1, 20):
        py = int((y - Y0) * SCALE)
        cv2.line(big, (0, py), (big.shape[1] - 1, py), (0, 160, 160), 1)
        if y % 40 == 0:
            cv2.putText(big, str(y), (2, py + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(big, label, (6, big.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    return big


def main():
    eps = {e["episode_index"]: e for e in load("meta/episodes/chunk-000/*.parquet").to_pylist()}
    rows = []
    for ei in EPS:
        e = eps[ei]
        n = int(e["length"])
        fidx = e[f"videos/{KEY}/file_index"]
        from_ts = e[f"videos/{KEY}/from_timestamp"]
        video = os.path.join(BASE, "videos", KEY, "chunk-000", f"file-{fidx:03d}.mp4")
        endp = os.path.join(OUT, "frames", f"ep{ei:02d}_end.png")
        if not os.path.exists(endp):
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", video,
                            "-ss", f"{from_ts + (n - 4 + 0.5) / FPS:.4f}", "-frames:v", "1", endp], check=True)
        gp = glob.glob(os.path.join(OUT, "frames", f"ep{ei:02d}_grasp_t*.png"))[0]
        s = zoom(cv2.imread(os.path.join(OUT, "frames", f"ep{ei:02d}_start_t0.png")), f"ep{ei} START")
        g = zoom(cv2.imread(gp), f"ep{ei} GRASP")
        en = zoom(cv2.imread(endp), f"ep{ei} END t={n - 4}")
        sep = np.full((s.shape[0], 6, 3), 255, np.uint8)
        rows.append(np.hstack([s, sep, g, sep, en]))
    for k in range(0, len(rows), 3):
        block = rows[k:k + 3]
        parts = []
        for r in block:
            parts.append(r)
            parts.append(np.full((8, r.shape[1], 3), 255, np.uint8))
        sheet = np.vstack(parts[:-1])
        cv2.imwrite(os.path.join(OUT, "sheets", f"trio_{k // 3 + 1}.png"), sheet)
        print("TRIO", k // 3 + 1, EPS[k:k + 3], sheet.shape, flush=True)


if __name__ == "__main__":
    main()
