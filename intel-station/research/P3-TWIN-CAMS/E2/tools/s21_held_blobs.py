#!/usr/bin/env python3
"""P3-TWIN-CAMS E2 step 21: held-brick colour blobs in the top video (read-only on the recordings).

Held interval per episode (joint time): gripper pos5 in [1.0, 4.5] (closed on a brick; empty closes read about 0.6)
inside [t_start - 1, t_end + 1] from episode.json. A top frame with pts[i] (out/motion_top.npz, absolute epoch) is kept
when pts[i] - OFF lies in a held interval, OFF = +0.0333 s (stage-2 top offset: video_time = joint_time + OFF). For
kept frames, colour blobs of the episode colour are extracted (HSV rules below; 150..6000 px components). The file is
decoded once, sequentially (grab, retrieve only for kept frames).
Writes out/s21_held_top.json (per frame) and out/s21_held_intervals.json; overlays/s21_carry_samples.jpg for viewing.
Run: envs/data-env/bin/python -I tools/s21_held_blobs.py
"""
import glob
import json
import os

import cv2
import numpy as np

R = "/home/intel/intel-junction-2026"
E2 = f"{R}/work/P3-TWIN-CAMS/E2"
E1 = f"{R}/work/P3-TWIN-CAMS/E1"
OFF = 0.0333
VIDEO = f"{R}/logs/monitor/top_20261114T170351Z.mkv"


def colour_mask(hsv, name):
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if name == "yellow":
        return (H >= 18) & (H <= 35) & (S >= 90) & (V >= 110)
    if name == "orange":
        return (H >= 5) & (H <= 18) & (S >= 130) & (V >= 110)
    if name == "red":
        return ((H <= 6) | (H >= 168)) & (S >= 120) & (V >= 70)
    if name == "white":
        return (S <= 55) & (V >= 175)
    return None


def held_runs(t, pos5, t0, t1):
    held = (pos5 >= 1.0) & (pos5 <= 4.5) & (t >= t0 - 1.0) & (t <= t1 + 1.0)
    p = np.concatenate([[0], held.astype(int), [0]])
    b = np.flatnonzero(np.diff(p))
    out = []
    for a, e in b.reshape(-1, 2):
        if e - a >= 15:
            out.append((float(t[a]), float(t[e - 1])))
    return out


def load_episodes():
    eps = []
    for ej in sorted(glob.glob(f"{R}/logs/sort/any_*/ep_*/episode.json")):
        d = json.load(open(ej))
        folder = os.path.dirname(ej)
        rel = os.path.relpath(folder, f"{R}/logs/sort")
        if not d.get("lift_q") or not d.get("colour") or d.get("t_start") is None:
            continue
        jf = os.path.join(folder, "joints.csv")
        if not os.path.exists(jf):
            continue
        arr = np.genfromtxt(jf, delimiter=",", names=True)
        eps.append({"ep": rel, "colour": d["colour"], "t_start": d["t_start"], "t_end": d["t_end"],
                    "t": arr["t"], "pos5": arr["pos5"], "success": d.get("success")})
    return eps


def main():
    pts = np.load(f"{E1}/out/motion_top.npz")["pts"]
    eps = load_episodes()
    intervals, want = [], {}
    for e in eps:
        runs = held_runs(e["t"], e["pos5"], e["t_start"], e["t_end"])
        for a, b in runs:
            lo = np.searchsorted(pts, a + OFF, side="left")
            hi = np.searchsorted(pts, b + OFF, side="right")
            for i in range(lo, hi):
                want[int(i)] = e["ep"]
            intervals.append({"ep": e["ep"], "colour": e["colour"], "joint_t0": round(a, 4), "joint_t1": round(b, 4),
                              "frames": int(hi - lo), "success": e["success"]})
    print("held intervals", len(intervals), "kept frames", len(want), flush=True)
    cap = cv2.VideoCapture(VIDEO)
    colour_of = {e["ep"]: e["colour"] for e in eps}
    frames, samples = [], {}
    last = max(want) if want else -1
    i = 0
    while i <= last:
        if not cap.grab():
            break
        if i in want:
            ok, img = cap.retrieve()
            if ok:
                ep = want[i]
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                m = colour_mask(hsv, colour_of[ep])
                blobs = []
                if m is not None:
                    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
                    for k in range(1, n):
                        a = int(st[k, cv2.CC_STAT_AREA])
                        if 150 <= a <= 6000:
                            blobs.append({"u": round(float(cen[k][0]), 1), "v": round(float(cen[k][1]), 1),
                                          "area": a, "w": int(st[k, cv2.CC_STAT_WIDTH]),
                                          "h": int(st[k, cv2.CC_STAT_HEIGHT]),
                                          "x": int(st[k, cv2.CC_STAT_LEFT]), "y": int(st[k, cv2.CC_STAT_TOP])})
                frames.append({"i": i, "t_video": round(float(pts[i]), 4), "ep": ep, "colour": colour_of[ep],
                               "blobs": blobs})
                if ep not in samples and len(blobs) >= 1:
                    samples[ep] = (img.copy(), blobs)
        i += 1
    cap.release()
    print("decoded up to frame", i - 1, "frames with data", len(frames), flush=True)
    json.dump({"video": VIDEO, "offset_s": OFF, "frames": frames},
              open(f"{E2}/out/s21_held_top.json", "w"))
    json.dump(intervals, open(f"{E2}/out/s21_held_intervals.json", "w"), indent=1)
    keys = list(samples.keys())[:3]
    tiles = []
    for k in keys:
        img, blobs = samples[k]
        t = img.copy()
        for b in blobs:
            cv2.rectangle(t, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), (255, 0, 255), 2)
            cv2.circle(t, (int(b["u"]), int(b["v"])), 4, (0, 255, 0), -1)
        cv2.putText(t, k, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(cv2.resize(t, (480, 360)))
    if tiles:
        while len(tiles) < 3:
            tiles.append(np.zeros((360, 480, 3), np.uint8))
        cv2.imwrite(f"{E2}/overlays/s21_carry_samples.jpg", np.hstack(tiles))
    print("samples", keys, flush=True)


if __name__ == "__main__":
    main()
