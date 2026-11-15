import glob
import json
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
VALID = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 14, 15]
# approximate plate discs in the top view (640x480), from the contact sheet; used only to exclude plates
PLATES = [(244, 150, 92), (589, 184, 96)]


def load(pattern):
    files = sorted(glob.glob(os.path.join(BASE, pattern)))
    return pa.concat_tables([pq.read_table(f) for f in files], promote_options="default")


def frame_png(video, t_sec, path):
    if not os.path.exists(path):
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", video,
                        "-ss", f"{t_sec:.4f}", "-frames:v", "1", path], check=True)
    return cv2.imread(path)


def plate_mask(h, w):
    m = np.zeros((h, w), np.uint8)
    for cx, cy, r in PLATES:
        cv2.circle(m, (cx, cy), r, 255, -1)
    return m


def roi_mask(h, w):
    m = np.zeros((h, w), np.uint8)
    m[0:335, 170:w] = 255
    return m


def main():
    os.makedirs(os.path.join(OUT, "overlay"), exist_ok=True)
    eps = {e["episode_index"]: e for e in load("meta/episodes/chunk-000/*.parquet").to_pylist()}
    result = {}
    for ei in VALID:
        e = eps[ei]
        fidx = e[f"videos/{KEY}/file_index"]
        from_ts = e[f"videos/{KEY}/from_timestamp"]
        n = int(e["length"])
        video = os.path.join(BASE, "videos", KEY, "chunk-000", f"file-{fidx:03d}.mp4")
        st = frame_png(video, from_ts + 0.5 / FPS, os.path.join(OUT, "frames", f"ep{ei:02d}_start_t0.png"))
        en = frame_png(video, from_ts + (n - 1 + 0.5) / FPS, os.path.join(OUT, "frames", f"ep{ei:02d}_end_t{n - 1}.png"))
        h, w = st.shape[:2]
        hsv_s = cv2.cvtColor(st, cv2.COLOR_BGR2HSV)
        hsv_e = cv2.cvtColor(en, cv2.COLOR_BGR2HSV)
        bright = (hsv_s[..., 2] > 110).astype(np.uint8) * 255
        mask = cv2.bitwise_and(bright, roi_mask(h, w))
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(plate_mask(h, w)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        num, lab, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        comps = []
        vend = hsv_e[..., 2]
        img = st.copy()
        cid = 0
        for k in range(1, num):
            area = int(stats[k, cv2.CC_STAT_AREA])
            if area < 250 or area > 4500:
                continue
            ys, xs = np.nonzero(lab == k)
            pts = np.stack([xs, ys], 1).astype(np.float32)
            (rcx, rcy), (rw, rh), ang = cv2.minAreaRect(pts)
            dark_end = float(np.mean(vend[ys, xs] < 80))
            col_s = st[ys, xs].mean(axis=0)
            comp = {"id": cid, "cx": round(float(cents[k][0]), 1), "cy": round(float(cents[k][1]), 1),
                    "area": area, "rect_w": round(float(rw), 1), "rect_h": round(float(rh), 1),
                    "rect_angle": round(float(ang), 1), "frac_dark_at_end": round(dark_end, 3),
                    "bgr_mean": [round(float(c), 1) for c in col_s]}
            comp["missing_at_end"] = bool(dark_end > 0.6)
            comps.append(comp)
            colr = (0, 255, 0) if comp["missing_at_end"] else (0, 255, 255)
            cv2.drawContours(img, [cv2.boxPoints(((rcx, rcy), (rw, rh), ang)).astype(np.int32)], 0, colr, 2)
            cv2.putText(img, str(cid), (int(rcx) + 4, int(rcy) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colr, 2)
            cid += 1
        cv2.putText(img, f"ep{ei} START comps green=missing at END", (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(OUT, "overlay", f"ep{ei:02d}.png"), img)
        result[str(ei)] = {"n": n, "from_ts": from_ts, "file_index": fidx, "components": comps}
        missing = [c["id"] for c in comps if c["missing_at_end"]]
        print("EP", ei, "n_comps", len(comps), "missing_ids", missing, flush=True)
    with open(os.path.join(OUT, "diff_picks.json"), "w") as f:
        json.dump(result, f, indent=1)
    print("WROTE diff_picks.json", flush=True)


if __name__ == "__main__":
    main()
