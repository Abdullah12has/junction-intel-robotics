import glob
import os
import subprocess
import sys

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


def load(pattern):
    files = sorted(glob.glob(os.path.join(BASE, pattern)))
    return pa.concat_tables([pq.read_table(f) for f in files], promote_options="default")


def grasp_index(g):
    for t in range(len(g) - 15):
        if g[t] < 4.0 and np.all(g[t:t + 15] < 4.0):
            return t
    return None


def frame_png(video, t_sec, out_path):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", video,
                    "-ss", f"{t_sec:.4f}", "-frames:v", "1", out_path], check=True)


def annotate(img, label):
    h, w = img.shape[:2]
    for x in range(0, w, 80):
        cv2.line(img, (x, 0), (x, h - 1), (0, 255, 255), 1)
        cv2.putText(img, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    for y in range(0, h, 80):
        cv2.line(img, (0, y), (w - 1, y), (0, 255, 255), 1)
        cv2.putText(img, str(y), (2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(img, label, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return img


def main():
    os.makedirs(os.path.join(OUT, "frames"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "sheets"), exist_ok=True)
    eps = load("meta/episodes/chunk-000/*.parquet").to_pylist()
    data = load("data/chunk-000/*.parquet").to_pylist()
    by_ep = {}
    for r in data:
        by_ep.setdefault(r["episode_index"], []).append(r)
    rows_out = []
    for e in sorted(eps, key=lambda x: x["episode_index"]):
        ei = e["episode_index"]
        rows = sorted(by_ep[ei], key=lambda r: r["frame_index"])
        g = np.array([r["observation.state"][5] for r in rows], dtype=float)
        gi = grasp_index(g)
        if gi is None or gi < 30:
            print("SKIP", ei, "grasp", gi)
            continue
        rel = None
        for t in range(gi, len(g)):
            if g[t] > 8.0:
                rel = t
                break
        fidx = e[f"videos/{KEY}/file_index"]
        from_ts = e[f"videos/{KEY}/from_timestamp"]
        video = os.path.join(BASE, "videos", KEY, "chunk-000", f"file-{fidx:03d}.mp4")
        tiles = {}
        for tag, t in (("start", 0), ("grasp", gi)):
            p = os.path.join(OUT, "frames", f"ep{ei:02d}_{tag}_t{t}.png")
            frame_png(video, from_ts + (t + 0.5) / FPS, p)
            tiles[tag] = p
        rows_out.append((ei, gi, rel, tiles))
        print("FRAMES", ei, "grasp_t", gi, "release_t", rel, "file", fidx, "from", round(from_ts, 3))
    for k in range(0, len(rows_out), 2):
        pair = rows_out[k:k + 2]
        strips = []
        for ei, gi, rel, tiles in pair:
            a = annotate(cv2.imread(tiles["start"]), f"ep{ei} START t=0")
            b = annotate(cv2.imread(tiles["grasp"]), f"ep{ei} GRASP t={gi}")
            strips.append(np.hstack([a, b]))
        sheet = np.vstack(strips)
        cv2.imwrite(os.path.join(OUT, "sheets", f"sheet_{k // 2 + 1}.png"), sheet)
        print("SHEET", k // 2 + 1, [p[0] for p in pair])


if __name__ == "__main__":
    main()
