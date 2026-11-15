"""Collect eye-in-hand calibration observations from the recorded sort episodes (Opus I-61, offline).

For every episode with joints.csv: find the windows where the arm is still (all body joints move < 0.25 units in
0.5 s), take the wrist-camera frame at the middle of each window from the continuous MJPEG segments (absolute pts
since 17:04Z), and store (t, q, all colour-class blobs). Only windows before the gripper closes (the brick is still
on the mat) are kept. Output: work/eyeinhand/obs.json + the frames (for review).
"""
import glob
import json
import os
import subprocess
import sys

import cv2
import numpy as np

R = "/home/intel/intel-junction-2026"
OUT = f"{R}/work/eyeinhand"
sys.path[:0] = [f"{R}/app/tools", f"{R}/app/anomaly/det2_f2"]
import pick_any as pa  # noqa: E402

os.makedirs(f"{OUT}/frames", exist_ok=True)


def seg_table():
    cache = f"{OUT}/segments.json"
    old = json.load(open(cache)) if os.path.exists(cache) else {}
    out = {}
    for f in sorted(glob.glob(f"{R}/data/wrist/raw/w_*.mkv")):
        if f in old:
            out[f] = old[f]
            continue
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=start_time,duration", "-of", "json", f],
                           capture_output=True, text=True)
        try:
            fm = json.loads(r.stdout)["format"]
            st, du = float(fm["start_time"]), float(fm.get("duration", 60.0))
        except Exception:  # noqa: BLE001
            continue
        if st > 1e9:  # absolute pts (recorder v2); with -copyts ffprobe reports the END time as "duration"
            out[f] = [st, du if du > 1e9 else st + du]
    json.dump(out, open(cache, "w"))
    return out


SEGS = seg_table()


def frame_at(t):
    for f, (st, en) in SEGS.items():
        if st <= t < en - 0.05:
            p = f"{OUT}/frames/w_{t:.2f}.jpg"
            if not os.path.exists(p):
                subprocess.run(["nice", "-n", "19", "ffmpeg", "-v", "error", "-ss", f"{t - st:.3f}", "-i", f, "-frames:v", "1",
                                "-q:v", "2", "-y", p], check=False)
            img = cv2.imread(p)
            return img, p
    return None, None


def still_windows(T, Q, max_move=0.25, min_len=0.5):
    out = []
    n = len(T)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and np.abs(Q[j + 1] - Q[i]).max() < max_move:
            j += 1
        if T[j] - T[i] >= min_len:
            out.append((T[i], T[j]))
        i = j + 1
    return out


obs = []
for epdir in sorted(glob.glob(f"{R}/logs/sort/any_17*/ep_*")):
    jf = f"{epdir}/joints.csv"
    ef = f"{epdir}/episode.json"
    if not (os.path.exists(jf) and os.path.exists(ef)):
        continue
    ep = json.load(open(ef))
    a = np.genfromtxt(jf, delimiter=",", names=True)
    if a.size < 30:
        continue
    T = a["t"]
    Q = np.stack([a[f"pos{k}"] for k in range(5)], 1)
    G = a["goal5"]
    closed = T[np.argmax(G < 5)] if (G < 5).any() else T[-1]  # gripper close command = brick leaves the mat
    hb = ep.get("hover_blob") or {}
    want = ep.get("target_colour") or hb.get("colour")
    for t0, t1 in still_windows(T, Q):
        tm = 0.5 * (t0 + t1)
        if tm > closed - 0.3:
            continue
        img, p = frame_at(tm)
        if img is None:
            continue
        q = Q[np.argmin(np.abs(T - tm))].tolist()
        bl = [{k: b[k] for k in ("centre", "short", "aspect", "yaw", "colour", "hue", "sat", "border", "plate_frac", "area")}
              for b in pa.blobs(img, min_area=1200, max_area=120000)]
        obs.append({"ep": os.path.relpath(epdir, f"{R}/logs/sort"), "t": round(tm, 3), "q": q, "frame": p,
                    "want": want, "hover_q": ep.get("hover_q"), "hover_blob": hb, "blobs": bl,
                    "dur": round(t1 - t0, 2)})
json.dump(obs, open(f"{OUT}/obs.json", "w"))
print(len(obs), "still-window observations from", len({o["ep"] for o in obs}), "episodes;", len(SEGS), "abs segments")
