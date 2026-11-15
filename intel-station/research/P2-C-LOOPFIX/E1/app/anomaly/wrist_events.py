#!/usr/bin/env python3
"""P2-B-ANOM-REAL phase A: grasp events from the continuous wrist recording.

Run with the data env, isolated:  python3 -I wrist_events.py extract [--out DIR] [--dry]
Incremental: segments listed in state.json are skipped. Event ids depend only on the
closure-onset UTC time, so re-runs keep them stable. The newest two segments are never
processed (the newest is being written; its successor must be complete for the overlap).
"""
import csv
import datetime as dt
import json
import os
import re
import sys

import cv2
import numpy as np
import pandas as pd

REC = "/home/intel/intel-junction-2026/data/wrist/raw"
OUT = "/home/intel/intel-junction-2026/data/anomaly/real"
FPS = 30.0
W, H = 640, 480
SEG_LEN = 60.0
SEG_RE = re.compile(r"^w_(\d{8}T\d{6})Z\.mkv$")
MIN_RED = 1500      # px of red jaw per half-frame (lower half) for a valid gap
CLOSE_V = 35.0      # px/s, jaw gap closing speed
MIN_RUN_S = 0.15    # closing must persist this long
MIN_DROP = 20.0     # px total closing in one run
SKIN_T = 0.05       # median skin share in the event window above this = hand present
PRE_S, POST_S = 0.3, 1.5
K_LOOK = 6          # frames used for the closing-speed slope
MERGE_S = 1.0       # closing bursts closer than this are one event


def seg_start(name):
    m = SEG_RE.match(name)
    return dt.datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=dt.timezone.utc).timestamp()


def read_seg(name):
    cap = cv2.VideoCapture(os.path.join(REC, name), cv2.CAP_FFMPEG)
    i = 0
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield i, f
            i += 1
    finally:
        cap.release()


def frame_feats(f):
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((h < 10) | (h > 170)) & (s > 110) & (v > 60)
    red[: H // 2, :] = False
    rl, rr = red[:, : W // 2], red[:, W // 2:]
    nL, nR = int(rl.sum()), int(rr.sum())
    gap = np.nan
    if nL >= MIN_RED and nR >= MIN_RED:
        cl = np.nonzero(rl.sum(axis=0) >= 3)[0]
        cr = np.nonzero(rr.sum(axis=0) >= 3)[0]
        if cl.size and cr.size:
            gap = float((W // 2 + cr.min()) - cl.max())
    skin = ((h <= 22) & (s >= 60) & (s <= 180) & (v >= 90) & ~red).mean()
    return gap, nL, nR, float(skin)


def detect(gaps):
    s = pd.Series(gaps).rolling(5, center=True, min_periods=3).median()
    s = s.interpolate(limit=10, limit_area="inside")
    v = ((s - s.shift(K_LOOK)) / (K_LOOK / FPS)).to_numpy()
    s = s.to_numpy()
    closing = v < -CLOSE_V
    runs, i, n = [], 0, len(closing)
    while i < n:
        if closing[i]:
            j = i
            while j + 1 < n and closing[j + 1]:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1
    evs = []
    for i, j in runs:
        if (j - i + 1) / FPS < MIN_RUN_S:
            continue
        on = max(0, i - K_LOOK)
        pre, post = s[on], s[j]
        if not (np.isfinite(pre) and np.isfinite(post)):
            continue
        drop = float(pre - post)
        if drop < MIN_DROP:
            continue
        if evs and (on - evs[-1]["end"]) / FPS < MERGE_S:
            evs[-1]["end"], evs[-1]["post"] = j, post
            evs[-1]["drop"] = evs[-1]["pre"] - post
            continue
        evs.append({"on": on, "end": j, "pre": float(pre), "post": float(post), "drop": drop})
    return evs


def iso_ms(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(round((t % 1) * 1000)) % 1000:03d}Z"


def extract(out, dry):
    names = sorted(n for n in os.listdir(REC) if SEG_RE.match(n))
    print("segments_total", len(names), "newest_skipped", names[-1] if names else None)
    os.makedirs(out, exist_ok=True)
    st_p = os.path.join(out, "state.json")
    state = json.load(open(st_p)) if os.path.exists(st_p) and not dry else {"processed": []}
    done = set(state["processed"])
    ev_p, fr_p = os.path.join(out, "events.csv"), os.path.join(out, "frames.csv")
    ev_cols = ["event_id", "segment", "t_in_segment", "utc", "closure_onset", "closure_end", "dur_s",
               "gap_pre", "gap_post", "drop_px", "skin_med", "hand_present", "lapvar_held", "n_frames"]
    fr_cols = ["event_id", "k", "t_rel", "gap", "skin", "lapvar"]
    for i, cur in enumerate(names):
        if i + 2 >= len(names):
            break
        if cur in done:
            continue
        nxt = names[i + 1]
        t0 = seg_start(cur)
        seq_t, seq_gap, seq_skin = [], [], []
        for name in (cur, nxt):
            s0 = seg_start(name)
            for idx, f in read_seg(name):
                g, _, _, sk = frame_feats(f)
                seq_t.append(s0 + idx / FPS)
                seq_gap.append(g)
                seq_skin.append(sk)
        ts = np.array(seq_t)
        gaps = np.array(seq_gap, dtype=float)
        skins = np.array(seq_skin, dtype=float)
        valid = float(np.isfinite(gaps[: int(SEG_LEN * FPS)]).mean()) if gaps.size else 0.0
        evs = [e for e in detect(gaps) if t0 <= ts[e["on"]] < t0 + SEG_LEN]
        for e in evs:
            e["t_on"], e["t_end"] = float(ts[e["on"]]), float(ts[e["end"]])
            e["w0"], e["w1"] = e["t_on"] - PRE_S, e["t_on"] + POST_S
            e["eid"] = "ev_" + dt.datetime.fromtimestamp(e["t_on"], dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%f")[:-3]
        print(f"SEG {cur} frames_seq={ts.size} valid_gap_share={valid:.3f} events={len(evs)}")
        for e in evs:
            print(f"  {e['eid']} t_in_seg={e['t_on'] - t0:.2f} dur={e['t_end'] - e['t_on']:.2f} "
                  f"pre={e['pre']:.1f} post={e['post']:.1f} drop={e['drop']:.1f}")
        if dry:
            continue
        frame_rows = []
        if evs:
            for e in evs:
                os.makedirs(os.path.join(out, "frames", e["eid"]), exist_ok=True)
            j = 0
            for name in (cur, nxt):
                s0 = seg_start(name)
                for idx, f in read_seg(name):
                    tt = s0 + idx / FPS
                    for e in evs:
                        if e["w0"] - 1e-6 <= tt <= e["w1"] + 1e-6:
                            k = int(round((tt - e["w0"]) * FPS))
                            cv2.imwrite(os.path.join(out, "frames", e["eid"], f"f{k:02d}.jpg"), f,
                                        [cv2.IMWRITE_JPEG_QUALITY, 95])
                            lap = float(cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                            frame_rows.append({"event_id": e["eid"], "k": k, "t_rel": round(k / FPS - PRE_S, 4),
                                               "gap": gaps[j], "skin": skins[j], "lapvar": lap, "t": tt,
                                               "_e": e["eid"]})
                    j += 1
        ev_rows, fr_out = [], []
        for e in evs:
            m = (ts >= e["w0"]) & (ts <= e["w1"])
            sk = skins[m]
            e["skin_med"] = float(np.median(sk)) if sk.size else float("nan")
            hold = [r["lapvar"] for r in frame_rows if r["_e"] == e["eid"] and e["t_end"] + 0.3 <= r["t"] <= e["t_end"] + 1.2]
            lap_held = float(np.median(hold)) if hold else float("nan")
            n_fr = sum(1 for r in frame_rows if r["_e"] == e["eid"])
            ev_rows.append({"event_id": e["eid"], "segment": cur, "t_in_segment": round(e["t_on"] - t0, 3),
                            "utc": iso_ms(e["t_on"]), "closure_onset": round(e["t_on"], 3),
                            "closure_end": round(e["t_end"], 3), "dur_s": round(e["t_end"] - e["t_on"], 3),
                            "gap_pre": round(e["pre"], 1), "gap_post": round(e["post"], 1),
                            "drop_px": round(e["drop"], 1), "skin_med": round(e["skin_med"], 4),
                            "hand_present": int(e["skin_med"] > SKIN_T), "lapvar_held": round(lap_held, 2),
                            "n_frames": n_fr})
        for r in frame_rows:
            fr_out.append({"event_id": r["event_id"], "k": r["k"], "t_rel": r["t_rel"],
                           "gap": "" if not np.isfinite(r["gap"]) else round(float(r["gap"]), 1),
                           "skin": round(float(r["skin"]), 4), "lapvar": round(r["lapvar"], 2)})
        for path, cols, rows in ((ev_p, ev_cols, ev_rows), (fr_p, fr_cols, fr_out)):
            new = not os.path.exists(path)
            with open(path, "a", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=cols)
                if new:
                    wr.writeheader()
                wr.writerows(rows)
        done.add(cur)
        state["processed"] = sorted(done)
        json.dump(state, open(st_p, "w"), indent=1)
        print(f"  wrote {len(ev_rows)} events for {cur}")
    print("done_segments", len(done))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] != "extract":
        print(__doc__)
        sys.exit(2)
    out = OUT
    if "--out" in args:
        out = args[args.index("--out") + 1]
    extract(out, dry="--dry" in args)
