#!/usr/bin/env python3
"""P2-B-ANOM-REAL phase A (OD-9): placed-brick samples at the inspection spot from Studio dataset copies.

Run isolated with the data env:  python3 -I spot_samples.py [--out DIR]

Per episode (gripper = observation.state[5], arm joints = observation.state[0:4]):
  grasp   = first frame, after an open value (>= GRASP_OPEN), where the gripper stays < GRASP_CLOSED for 1 s
  release = first frame, after the grasp, where the gripper stays >= RELEASE_V for 0.5 s
  pre     = episode frame 0 (arm home, spot empty)
  post    = latest frame p <= end - 0.5 s with p >= release + 0.5 s and the arm more than AWAY_UNITS
            away from the release pose on joints 0-3
Per RealSense key: the placed brick = largest blob that became brighter between pre and post, not red
(jaws), not touching the image border. Crop square with margin, resize 256, three colour variants.
Recomputes all episodes on each run (cheap, 11 episodes); sample ids are stable.
Video is AV1: decoded with the ffmpeg CLI (libdav1d), because OpenCV cannot decode it here.
"""
import csv
import glob
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = "/home/intel/intel-junction-2026/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af"
DS_ID = "6d6bdce3"
OUT = "/home/intel/intel-junction-2026/data/anomaly/real/spot"
FPS = 30
W, H = 640, 480
CAMS = ["realsense 243722064316", "realsense 243622061421"]
CAM_ROLE = {"realsense 243722064316": "close", "realsense 243622061421": "top"}
GRASP_OPEN, GRASP_CLOSED, RELEASE_V = 8.0, 5.0, 6.0
HALF_S = 0.5
AWAY_UNITS = 15.0
MIN_AREA_CAM = {"realsense 243722064316": 600, "realsense 243622061421": 400}
PRE_DARK, POST_BRIGHT, DIFF_T = 90, 120, 45
CROP = 256


def red_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((h < 10) | (h > 170)) & (s > 110) & (v > 60)


def strong_red(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((h <= 6) | (h >= 174)) & (s > 120) & (v > 60)


def analyse(sub):
    S = np.stack(sub["observation.state"].to_numpy())
    g, J, L = S[:, 5], S[:, :4], len(S)
    res = {"len": L, "grasp": "", "release": "", "post": "", "spot_pose": "", "usable": 0, "reason": ""}
    above_open = np.where(g >= GRASP_OPEN)[0]
    if above_open.size == 0:
        res["reason"] = "never_open"
        return res, None
    closed = g < GRASP_CLOSED
    grasp = None
    for i in range(int(above_open[0]), L - 30):
        if closed[i] and closed[i:i + 30].all():
            grasp = i
            break
    if grasp is None:
        res["reason"] = "no_grasp_after_open (closed at start or never closed)"
        return res, None
    above = g >= RELEASE_V
    rel = None
    for i in range(grasp + 30, L - 15):
        if above[i:i + 15].all():
            rel = i
            break
    if rel is None:
        res["grasp"] = grasp
        res["reason"] = "held_to_end (no release before episode end)"
        return res, None
    cand = [p for p in range(rel + 15, L - 15 + 1) if np.abs(J[p] - J[rel]).max() > AWAY_UNITS]
    res["grasp"], res["release"] = grasp, rel
    res["spot_pose"] = " ".join(f"{v:.1f}" for v in J[rel])
    if not cand:
        res["reason"] = "no_arm_away_frame_after_release"
        return res, None
    post = max(cand)
    res["post"], res["usable"] = post, 1
    return res, post


def decode(path, idxs):
    idxs = sorted(set(idxs))
    sel = "+".join(f"eq(n\\,{i})" for i in idxs)
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"select='{sel}'", "-vsync", "0",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(raw) // (W * H * 3)
    if n != len(idxs):
        raise RuntimeError(f"decoded {n} frames, expected {len(idxs)} from {path}")
    arr = np.frombuffer(raw, np.uint8)[: n * W * H * 3].reshape(n, H, W, 3)
    return {i: arr[k].copy() for k, i in enumerate(idxs)}


def find_changed(pre, post, min_area):
    pg = cv2.GaussianBlur(pre, (5, 5), 0).astype(np.int16)
    qg = cv2.GaussianBlur(post, (5, 5), 0).astype(np.int16)
    d = np.abs(qg - pg).max(axis=2)
    hp = cv2.cvtColor(pre, cv2.COLOR_BGR2HSV).astype(np.int16)
    hq = cv2.cvtColor(post, cv2.COLOR_BGR2HSV).astype(np.int16)
    vpre, vpost = hp[..., 2], hq[..., 2]
    hd = np.abs(hq[..., 0] - hp[..., 0])
    hd = np.minimum(hd, 180 - hd)
    gained = (vpost > POST_BRIGHT) & ((vpre < PRE_DARK) | ((hq[..., 1] > 80) & (hd > 18)))
    m = ((d > DIFF_T) & gained).astype(np.uint8) * 255
    static_red = strong_red(pre) & strong_red(post)
    static_red = cv2.dilate(static_red.astype(np.uint8) * 255, np.ones((9, 9), np.uint8)) > 0
    m[static_red] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    cands = []
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if a < min_area or x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2:
            continue
        cands.append((int(a), k, int(x), int(y), int(w), int(h)))
    cands.sort(reverse=True)
    if not cands:
        return None, 0, m
    a, k, x, y, w, h = cands[0]
    comp = (lab == k).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (rw, rh) = cv2.minAreaRect(max(cnts, key=cv2.contourArea))[1]
    aspect = max(rw, rh) / max(min(rw, rh), 1e-6)
    info = {"x": x, "y": y, "w": w, "h": h, "area": a, "aspect": round(float(aspect), 3),
            "cx": round(x + w / 2, 1), "cy": round(y + h / 2, 1), "n_cands": len(cands),
            "second_area": cands[1][0] if len(cands) > 1 else 0}
    return info, len(cands), m


def square_crop(img, x, y, w, h, margin=0.15):
    cx, cy = x + w / 2, y + h / 2
    s = max(w, h) * (1 + 2 * margin)
    x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
    x1, y1 = x0 + int(round(s)), y0 + int(round(s))
    pad = max(0, -x0, -y0, x1 - W, y1 - H)
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    crop = padded[y0 + pad:y1 + pad, x0 + pad:x1 + pad]
    return cv2.resize(crop, (CROP, CROP), interpolation=cv2.INTER_AREA)


def colour_variants(crop, pre):
    c = crop.astype(np.float32)
    g = c.reshape(-1, 3).mean(0)
    gw = np.clip(c * (g.mean() / np.maximum(g, 1e-3)), 0, 255).astype(np.uint8)
    V = cv2.cvtColor(pre, cv2.COLOR_BGR2HSV)[..., 2]
    dark = pre[V <= 40]
    mat = dark.reshape(-1, 3).astype(np.float32).mean(0) if dark.size else g
    mn = np.clip(c * (mat.mean() / np.maximum(mat, 1e-3)), 0, 255).astype(np.uint8)
    return crop, gw, mn


def overview(pre_img, post_img, info, path):
    tiles = []
    for im, label in ((pre_img, "pre"), (post_img, "post")):
        t = cv2.resize(im.copy(), (320, 240))
        cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(t)
    if info:
        sc = 0.5
        cv2.rectangle(tiles[1], (int(info["x"] * sc), int(info["y"] * sc)),
                      (int((info["x"] + info["w"]) * sc), int((info["y"] + info["h"]) * sc)), (0, 255, 0), 2)
    return np.hstack(tiles)


def main():
    out = OUT
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
    for sub in ("crops/raw", "crops/gw", "crops/mat", "overview"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    ep = pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(f"{ROOT}/meta/episodes/chunk-*/file-*.parquet"))], ignore_index=True)
    d = pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(f"{ROOT}/data/chunk-*/file-*.parquet"))], ignore_index=True)
    ep_rows, requests = [], {}
    plan = {}
    for _, r in ep.iterrows():
        e = int(r["episode_index"])
        sub = d[d["episode_index"] == e]
        res, post = analyse(sub)
        row = {"episode": e, "task": str(r["tasks"][0]), "length": int(r["length"]), **res}
        ep_rows.append(row)
        print("EP", e, json.dumps({k: row[k] for k in ("length", "grasp", "release", "post", "usable", "reason", "spot_pose")}))
        if not res["usable"]:
            continue
        plan[e] = (int(post), int(res["release"]))
        for cam in CAMS:
            cr = r[f"videos/observation.images.{cam}/chunk_index"]
            fi = r[f"videos/observation.images.{cam}/file_index"]
            base = int(round(float(r[f"videos/observation.images.{cam}/from_timestamp"]) * FPS))
            path = f"{ROOT}/videos/observation.images.{cam}/chunk-{int(cr):03d}/file-{int(fi):03d}.mp4"
            requests.setdefault(path, []).extend([(e, cam, "pre", base + 0), (e, cam, "post", base + post)])
    frames = {}
    for path, reqs in requests.items():
        got = decode(path, [q[3] for q in reqs])
        for e, cam, role, fidx in reqs:
            frames[(e, cam, role)] = got[fidx]
    sample_rows = []
    for e, (post, rel) in plan.items():
        for cam in CAMS:
            sid = f"{DS_ID}_e{e:02d}_{cam.split()[-1]}"
            pre_img, post_img = frames[(e, cam, "pre")], frames[(e, cam, "post")]
            info, ncand, mask = find_changed(pre_img, post_img, MIN_AREA_CAM[cam])
            cv2.imwrite(os.path.join(out, "overview", f"{sid}_pre_post.jpg"), overview(pre_img, post_img, info, None))
            if info is None:
                sample_rows.append({"sample_id": sid, "episode": e, "camera": CAM_ROLE[cam], "usable": 0, "flag": "no_changed_blob"})
                continue
            crop = square_crop(post_img, info["x"], info["y"], info["w"], info["h"])
            raw, gw, mn = colour_variants(crop, pre_img)
            cv2.imwrite(os.path.join(out, "crops", "raw", f"{sid}.png"), raw)
            cv2.imwrite(os.path.join(out, "crops", "gw", f"{sid}.png"), gw)
            cv2.imwrite(os.path.join(out, "crops", "mat", f"{sid}.png"), mn)
            flag = []
            if info["n_cands"] > 1:
                flag.append(f"multi_blob({info['n_cands']})")
            if info["area"] < 3000:
                flag.append("small_area")
            sugg = "NOT-2x2" if info["aspect"] >= 1.6 else "2x2"
            sample_rows.append({"sample_id": sid, "episode": e, "camera": CAM_ROLE[cam], "usable": 1,
                                "pre_frame": 0, "post_frame": post, "release_frame": rel,
                                "bbox_x": info["x"], "bbox_y": info["y"], "bbox_w": info["w"], "bbox_h": info["h"],
                                "cx": info["cx"], "cy": info["cy"], "area": info["area"], "aspect": info["aspect"],
                                "n_cands": info["n_cands"], "second_area": info["second_area"],
                                "suggested": sugg, "suggest_conf": "low" if cam == CAMS[0] else "med",
                                "flag": ";".join(flag)})
            print("SAMPLE", sid, json.dumps({k: sample_rows[-1][k] for k in ("bbox_w", "bbox_h", "cx", "cy", "area", "aspect", "suggested", "flag")}))
    with open(os.path.join(out, "samples.csv"), "w", newline="") as fh:
        cols = sorted({k for rr in sample_rows for k in rr}, key=lambda c: (c not in ("sample_id", "episode", "camera", "usable"), c))
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(sample_rows)
    with open(os.path.join(out, "episodes.csv"), "w", newline="") as fh:
        cols = ["episode", "task", "length", "grasp", "release", "post", "spot_pose", "usable", "reason"]
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for rr in ep_rows:
            wr.writerow({k: rr[k] for k in cols})
    print("done samples", sum(1 for s in sample_rows if s.get("usable")), "episodes_usable", len(plan))


if __name__ == "__main__":
    main()
