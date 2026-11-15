#!/usr/bin/env python3
"""P2-B-ANOM-REAL phase A2 (OD-11): held-brick crops from every episode of every synced Studio dataset.

Run isolated with the data env:  python3 -I held_harvest.py
Candidate frames per episode: gripper.pos (observation.state[5]) in [GLO, GHI] for STABLE frames
(closed on a brick, not empty) and the arm moved away from its start pose (joints 0-3, > MOVED_UNITS).
Detection per RealSense view: strong-red jaw pixels give the jaw centre; the held brick is the nearest
saturated blob (not jaw) within MAX_DIST of that centre and within the area range.
Per episode and camera: up to PER_EP frames, sharpest first (Laplacian variance of the crop), at least
SEP episode frames apart. Crops 256x256 raw, gray-world and mat-normalised.
Stable ids: <dataset8>_e<ep>_<camera serial>_<k>; episodes whose state is byte-identical to an earlier
dataset's episode are skipped (duplicate copies). Incremental: episodes already in state.json are skipped.
Video is AV1: decoded with the ffmpeg CLI (libdav1d).
"""
import csv
import glob
import hashlib
import json
import os
import subprocess

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATA = "/home/intel/intel-junction-2026/data/studio-datasets"
OUT = "/home/intel/intel-junction-2026/data/anomaly/real/held"
SHEETS = "/home/intel/intel-junction-2026/data/anomaly/real/label_sheets"
FPS = 30
W, H = 640, 480
CAMS = ["realsense 243722064316", "realsense 243622061421"]
SERIAL = {CAMS[0]: "243722064316", CAMS[1]: "243622061421"}
ROLE = {CAMS[0]: "close", CAMS[1]: "top"}
GLO, GHI = 1.6, 4.5
STABLE = 15
MOVED_UNITS = 20.0
PER_EP = 4
SEP = 15
MIN_AREA = {CAMS[0]: 1500, CAMS[1]: 300}
MAX_AREA = {CAMS[0]: 40000, CAMS[1]: 6000}
MAX_DIST = {CAMS[0]: 160, CAMS[1]: 90}
CROP = 256
ASPECT_NOT2x2 = 1.6


def stream(path):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = W * H * 3
    i = 0
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield i, np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
            i += 1
    finally:
        p.kill()
        p.stdout.close()
        p.wait()


def detect(img, cam):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    jaw = ((h <= 6) | (h >= 174)) & (s > 120) & (v > 60)
    ys, xs = np.nonzero(jaw)
    if xs.size < 400:
        return None
    jc = (float(np.median(xs)), float(np.median(ys)))
    sat = ((s > 90) & (v > 110) & (~jaw)).astype(np.uint8) * 255
    sat = cv2.morphologyEx(sat, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    sat = cv2.morphologyEx(sat, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(sat)
    best = None
    for k in range(1, n):
        x, y, w, hh, a = st[k]
        if a < MIN_AREA[cam] or a > MAX_AREA[cam]:
            continue
        dist = float(np.hypot(cen[k][0] - jc[0], cen[k][1] - jc[1]))
        if dist > MAX_DIST[cam]:
            continue
        cm = lab == k
        if 16 <= float(h[cm].mean()) <= 34 and float(s[cm].mean()) > 160:
            continue  # yellow-orange saturated plastic: wrist-camera mount, not a brick
        if best is None or dist < best[0]:
            best = (dist, k, int(x), int(y), int(w), int(hh), int(a))
    if best is None:
        return None
    dist, k, x, y, w, hh, a = best
    comp = (lab == k)
    cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (rw, rh) = cv2.minAreaRect(max(cnts, key=cv2.contourArea))[1]
    aspect = max(rw, rh) / max(min(rw, rh), 1e-6)
    tape = float(((v > 240) & (s < 50) & comp).sum()) / max(1, a)
    dark = img[v <= 40]
    mat = dark.reshape(-1, 3).astype(np.float32).mean(0) if dark.shape[0] > 1000 else None
    return {"bbox": (x, y, w, hh), "area": a, "aspect": float(aspect), "tape": tape, "dist": dist, "mat": mat}


def square_crop(img, x, y, w, h, margin=0.15):
    cx, cy = x + w / 2, y + h / 2
    s = max(w, h) * (1 + 2 * margin)
    x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
    x1, y1 = x0 + int(round(s)), y0 + int(round(s))
    pad = max(0, -x0, -y0, x1 - W, y1 - H)
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    crop = padded[y0 + pad:y1 + pad, x0 + pad:x1 + pad]
    return cv2.resize(crop, (CROP, CROP), interpolation=cv2.INTER_AREA)


def variants(crop, mat):
    c = crop.astype(np.float32)
    g = c.reshape(-1, 3).mean(0)
    gw = np.clip(c * (g.mean() / np.maximum(g, 1e-3)), 0, 255).astype(np.uint8)
    ref = mat if mat is not None else g
    mn = np.clip(c * (ref.mean() / np.maximum(ref, 1e-3)), 0, 255).astype(np.uint8)
    return crop, gw, mn


def lapvar(crop):
    return float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def candidates(S):
    g = S[:, 5]
    band = ((g >= GLO) & (g <= GHI)).astype(float)
    stable = pd.Series(band).rolling(STABLE, center=True, min_periods=STABLE).min().fillna(0).to_numpy() > 0.5
    moved = np.abs(S[:, :4] - S[0, :4]).max(axis=1) > MOVED_UNITS
    return np.nonzero(stable & moved)[0]


def load_json(p, default):
    return json.load(open(p)) if os.path.exists(p) else default


def main():
    os.makedirs(OUT, exist_ok=True)
    for sub in ("crops/raw", "crops/gw", "crops/mat"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    os.makedirs(SHEETS, exist_ok=True)
    state_p = os.path.join(OUT, "state.json")
    state = load_json(state_p, {"done": {}, "hash": {}})
    rows_p = os.path.join(OUT, "samples_held.csv")
    eps_p = os.path.join(OUT, "episodes_held.csv")
    rows = list(csv.DictReader(open(rows_p))) if os.path.exists(rows_p) else []
    datasets = sorted(os.path.basename(p.rstrip("/")) for p in glob.glob(f"{DATA}/*/") if glob.glob(f"{p}meta/episodes/chunk-*/file-*.parquet"))
    per_file = {}
    ep_status = []
    for ds in datasets:
        root = f"{DATA}/{ds}"
        ep = pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(f"{root}/meta/episodes/chunk-*/file-*.parquet"))], ignore_index=True)
        d = pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(f"{root}/data/chunk-*/file-*.parquet"))], ignore_index=True)
        for _, r in ep.iterrows():
            e = int(r["episode_index"])
            sub = d[d["episode_index"] == e]
            S = np.stack(sub["observation.state"].to_numpy()).astype(np.float64)
            h = hashlib.sha256(S.tobytes()).hexdigest()[:16]
            ds8 = ds[:8]
            if h in state["hash"] and state["hash"][h] != f"{ds8}_e{e:02d}":
                ep_status.append({"dataset": ds, "episode": e, "status": "duplicate_of " + state["hash"][h]})
                continue
            state["hash"].setdefault(h, f"{ds8}_e{e:02d}")
            for cam in CAMS:
                key = f"{ds8}_e{e:02d}_{SERIAL[cam]}"
                if key in state["done"]:
                    ep_status.append({"dataset": ds, "episode": e, "status": "done " + SERIAL[cam]})
                    continue
                cands = candidates(S)
                cr = int(r[f"videos/observation.images.{cam}/chunk_index"])
                fi = int(r[f"videos/observation.images.{cam}/file_index"])
                base = int(round(float(r[f"videos/observation.images.{cam}/from_timestamp"]) * FPS))
                path = f"{root}/videos/observation.images.{cam}/chunk-{cr:03d}/file-{fi:03d}.mp4"
                per_file.setdefault(path, []).append({"key": key, "ds": ds, "ds8": ds8, "ep": e, "cam": cam,
                                                      "base": base, "length": len(S), "cands": cands,
                                                      "hits": {}})
    # group pending episodes by video file and stream each file once
    for path, jobs in per_file.items():
        want = {}
        for j in jobs:
            for f in j["cands"]:
                want.setdefault(j["base"] + int(f), []).append((j, int(f)))
        if not want:
            for j in jobs:
                ep_status.append({"dataset": j["ds"], "episode": j["ep"], "status": f"no_candidates {SERIAL[j['cam']]}"})
                state["done"][j["key"]] = 0
            continue
        last = max(want)
        for i, img in stream(path):
            if i > last:
                break
            for j, f in want.get(i, []):
                det = detect(img, j["cam"])
                if det is None:
                    continue
                x, y, w, hh = det["bbox"]
                crop = square_crop(img, x, y, w, hh)
                ok, enc = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 98])
                j["hits"][f] = {"score": lapvar(crop), "det": det, "crop": enc}
        for j in jobs:
            hits = j["hits"]
            chosen = []
            for f in sorted(hits, key=lambda q: -hits[q]["score"]):
                if len(chosen) >= PER_EP:
                    break
                if all(abs(f - c) >= SEP for c in chosen):
                    chosen.append(f)
            chosen = sorted(chosen)
            for k, f in enumerate(chosen, start=1):
                hit = hits[f]
                sid = f"{j['ds8']}_e{j['ep']:02d}_{SERIAL[j['cam']]}_{k}"
                raw, gw, mn = variants(cv2.imdecode(hit["crop"], cv2.IMREAD_COLOR), hit["det"]["mat"])
                cv2.imwrite(os.path.join(OUT, "crops", "raw", f"{sid}.png"), raw)
                cv2.imwrite(os.path.join(OUT, "crops", "gw", f"{sid}.png"), gw)
                cv2.imwrite(os.path.join(OUT, "crops", "mat", f"{sid}.png"), mn)
                det = hit["det"]
                flag = "tape?" if det["tape"] > 0.05 else ""
                sugg = "NOT-2x2" if det["aspect"] >= ASPECT_NOT2x2 else "2x2"
                rows.append({"sample_id": sid, "dataset": j["ds"], "episode": j["ep"], "camera": ROLE[j["cam"]],
                             "serial": SERIAL[j["cam"]], "ep_frame": f, "t_s": round(f / FPS, 2),
                             "score_lapvar": round(hit["score"], 1), "area": det["area"],
                             "aspect": round(det["aspect"], 3), "tape_share": round(det["tape"], 4),
                             "jaw_dist": round(det["dist"], 1), "suggested": sugg, "flag": flag})
            state["done"][j["key"]] = len(chosen)
            ep_status.append({"dataset": j["ds"], "episode": j["ep"], "status": f"{SERIAL[j['cam']]} cands={len(j['cands'])} detected={len(hits)} selected={len(chosen)}"})
    # write state, rows and episode status
    json.dump(state, open(state_p, "w"), indent=1)
    cols = ["sample_id", "dataset", "episode", "camera", "serial", "ep_frame", "t_s", "score_lapvar", "area",
            "aspect", "tape_share", "jaw_dist", "suggested", "flag"]
    with open(rows_p, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(sorted(rows, key=lambda r: r["sample_id"]))
    with open(eps_p, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["dataset", "episode", "status"])
        wr.writeheader()
        wr.writerows(ep_status)
    print("samples", len(rows), "episodes_status", len(ep_status))


if __name__ == "__main__":
    main()
