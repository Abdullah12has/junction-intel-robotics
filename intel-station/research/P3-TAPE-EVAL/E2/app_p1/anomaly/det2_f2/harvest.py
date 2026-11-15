"""Harvest arm-at-home, still frames (start and end of every episode) from synced LeRobot datasets.

usage: python3 -I harvest.py <dataset_root> <out_dir> [--diag]
Writes <out_dir>/<ds8>_e<ep>_<start|end>_<serial>.png (TOP and CLOSE) and merges <out_dir>/samples.json.
A start run must begin in the first 60 frames; an end run must end in the last 120 frames.
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLOSE, FPS, HOME, TOP, skin_fraction  # noqa: E402

HOME_TOL = 8.0  # units, L-inf over the 5 arm joints (pan, lift, elbow, wrist_flex, wrist_roll)
STILL_TOL = 2.0  # units per frame, L-inf
MIN_RUN = 15  # frames (0.5 s at 30 fps)
HANDS_FRAC = 0.01  # skin-hue pixel fraction above which a view is flagged 'hands'


def meta_rows(root):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "meta/episodes/**/*.parquet"), recursive=True)):
        rows += pq.read_table(f).to_pylist()
    return {int(r["episode_index"]): r for r in rows}


def states(root):
    acc = {}
    for f in sorted(glob.glob(os.path.join(root, "data/**/*.parquet"), recursive=True)):
        t = pq.read_table(f, columns=["observation.state", "timestamp", "frame_index", "episode_index"]).to_pydict()
        for s, ts, fi, ep in zip(t["observation.state"], t["timestamp"], t["frame_index"], t["episode_index"]):
            acc.setdefault(int(ep), []).append((int(fi), float(ts), s))
    out = {}
    for ep, lst in acc.items():
        lst.sort(key=lambda x: x[0])
        out[ep] = (np.array([x[2] for x in lst], float), np.array([x[1] for x in lst], float))
    return out


def still_home_flags(S):
    d_home = np.abs(S[:, :5] - HOME).max(axis=1)
    vel = np.zeros(len(S))
    vel[1:] = np.abs(np.diff(S[:, :5], axis=0)).max(axis=1)
    return d_home, vel


def still_home_runs(S):
    d_home, vel = still_home_flags(S)
    ok = (d_home <= HOME_TOL) & (vel <= STILL_TOL)
    runs, i = [], 0
    while i < len(ok):
        if ok[i]:
            j = i
            while j + 1 < len(ok) and ok[j + 1]:
                j += 1
            if j - i + 1 >= MIN_RUN:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def pick_runs(runs, n):
    start = next((r for r in runs if r[0] <= 60), None)
    end = next((r for r in reversed(runs) if r[1] >= n - 120 and r != start), None)
    picks = []
    if start:
        picks.append(("start", start))
    if end:
        picks.append(("end", end))
    return picks


def extract(video, t, out):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.4f}", "-i", video,
                    "-frames:v", "1", out], check=True)


def main():
    import cv2

    ds_root = os.path.abspath(sys.argv[1])
    out_dir = os.path.abspath(sys.argv[2])
    diag = "--diag" in sys.argv
    os.makedirs(out_dir, exist_ok=True)
    ds = os.path.basename(ds_root.rstrip("/"))
    ds8 = ds[:8]
    meta, st = meta_rows(ds_root), states(ds_root)
    idx_path = os.path.join(out_dir, "samples.json")
    index = json.load(open(idx_path)) if os.path.exists(idx_path) else {}
    n_new = 0
    for ep in sorted(meta):
        if ep not in st:
            continue
        S, TS = st[ep]
        runs = still_home_runs(S)
        d_home, _ = still_home_flags(S)
        if diag:
            print(f"ep{ep:02d} len={len(S)} min_home_dist={d_home.min():.2f} near_home={int((d_home <= HOME_TOL).sum())} runs={runs}")
        for kind, (a, b) in pick_runs(runs, len(S)):
            f = a + min(30, (b - a) // 2)
            sid = f"{ds8}_e{ep:02d}_{kind}"
            rec = {"id": sid, "dataset": ds, "ds8": ds8, "episode": ep, "kind": kind, "frame": int(f),
                   "run": [int(a), int(b)], "state": [round(float(x), 3) for x in S[f]],
                   "home_dist": round(float(d_home[f]), 2), "views": {}}
            cm = meta[ep]
            for cam in (TOP, CLOSE):
                vk = f"observation.images.{cam}"
                video = os.path.join(ds_root, "videos", vk, f"chunk-{cm[f'videos/{vk}/chunk_index']:03d}",
                                     f"file-{cm[f'videos/{vk}/file_index']:03d}.mp4")
                t = float(cm[f"videos/{vk}/from_timestamp"]) + float(TS[f])
                out = os.path.join(out_dir, f"{sid}_{cam.split()[-1]}.png")
                if not os.path.exists(out):
                    extract(video, t, out)
                    n_new += 1
                img = cv2.imread(out)
                frac, _ = skin_fraction(img)
                rec["views"][cam] = {"png": out, "video_t": round(t, 3), "skin_frac": round(frac, 4),
                                     "hands_flag": bool(frac > HANDS_FRAC)}
            index[sid] = rec
            if diag:
                print(f"  {kind} frame={f} run={a}-{b} home_dist={rec['home_dist']} "
                      f"skin_top={rec['views'][TOP]['skin_frac']} skin_close={rec['views'][CLOSE]['skin_frac']}")
    json.dump(index, open(idx_path, "w"), indent=1, sort_keys=True)
    print(f"{ds8}: episodes={len(meta)} samples_total={len(index)} new_pngs={n_new}")


if __name__ == "__main__":
    main()
