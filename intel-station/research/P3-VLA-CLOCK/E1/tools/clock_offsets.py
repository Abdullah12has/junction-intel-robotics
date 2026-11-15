#!/usr/bin/env python3
"""P3-VLA-CLOCK E1: camera-to-joint clock offsets for recorded episodes (v2).

Read-only inputs: R/logs/sort/<run>/ep_*/joints.csv, R/logs/monitor/{top,close}_*.mkv,
R/data/wrist/raw/w_*.mkv. No camera device, serial port, Studio or other folder is opened.
Outputs (R/work/P3-VLA-CLOCK/E1): offsets.csv, metrics.json, analysis_details.json, plots/.

Modes
  inventory                   episodes, joint windows and the camera segments covering them
  analyze                     per (episode, camera) lag: cross-correlation plus event check
  visual RUN EP CAM LAG_MS    frame strip around the first joint motion onset of one episode

Camera energy: a frame pair (t_prev, t) gives one sample at the pair midpoint (t_prev + t) / 2,
because the motion between the two frames happened inside that interval.
Primary energy: mean absolute difference of blurred, mean-removed 160x120 gray frames.
Secondary energy: fraction of pixels whose difference exceeds 12 gray levels (robustness check).
Sign convention: lag = camera timestamp minus joint timestamp for the same physical motion.
"""
import csv
import glob
import json
import os
import sys
import traceback
from datetime import datetime, timezone

import av
import cv2
import numpy as np

cv2.setNumThreads(4)

ROOT = "/home/intel/intel-junction-2026"
WORK = os.path.join(ROOT, "work", "P3-VLA-CLOCK", "E1")
PLOTS = os.path.join(WORK, "plots")
SORT = os.path.join(ROOT, "logs", "sort")
CAM_DIR = {"wrist": os.path.join(ROOT, "data", "wrist", "raw"),
           "top": os.path.join(ROOT, "logs", "monitor"),
           "close": os.path.join(ROOT, "logs", "monitor")}
CAM_PREFIX = {"wrist": "w", "top": "top", "close": "close"}
CAMERAS = ("wrist", "top", "close")
CAM_COLOUR = {"wrist": (255, 0, 0), "top": (0, 160, 0), "close": (0, 0, 255)}
RUNS = ("any_1731", "any_1740")
GRID = 0.01
PAD = 0.8
MAXLAG_MS = 500
STEPS = MAXLAG_MS // 10
WRIST_SEG_S = 60.0
FRAME_MS = 1000.0 / 30.0
RELIABLE_PEAK = 0.4
COLS = ["episode", "camera", "lag_ms", "lag_frames", "peak", "n", "event_lag_ms", "r_zero_lag",
        "event_pairs", "agree_50ms", "clock_ref", "edge", "cam_frames", "cam_dt_ms", "max_gap_ms",
        "dup_frac", "lag_h1_ms", "lag_h2_ms", "lag_frac_ms", "peak_frac", "segments"]
P2_PREVIOUS = {"source": "work/P2-D-VLA1/E1/STATUS.md R4 (jaw-gap half-travel, different method)",
               "any_1740/ep_0017 wrist vs goal5, lag_ms": -103.4,
               "any_1740/ep_0017 wrist vs pos5, lag_ms": -160.7}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_seconds(stamp):
    return datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()


def fmt_utc(t):
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%H:%M:%S.%f")[:-3] + "Z"


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    return o


def segments(cam):
    prefix = CAM_PREFIX[cam]
    items = []
    for path in glob.glob(os.path.join(CAM_DIR[cam], prefix + "_*.mkv")):
        name = os.path.basename(path)
        name_t = utc_seconds(name[len(prefix) + 1:-4])
        c = av.open(path)
        try:
            start_us = c.start_time or 0
        finally:
            c.close()
        if start_us / 1e6 > 1e9:
            items.append({"name": name, "path": path, "start": start_us / 1e6, "base": 0.0, "relative": False})
        else:
            items.append({"name": name, "path": path, "start": name_t, "base": name_t, "relative": True})
    items.sort(key=lambda s: s["start"])
    for k, s in enumerate(items):
        if k + 1 < len(items):
            s["end"] = items[k + 1]["start"]
        elif cam == "wrist":
            s["end"] = s["start"] + WRIST_SEG_S
        else:
            s["end"] = float("inf")
    return items


def episodes():
    out = []
    for run in RUNS:
        for d in sorted(glob.glob(os.path.join(SORT, run, "ep_*"))):
            if os.path.isdir(d):
                out.append((run, os.path.basename(d), d))
    return out


def read_meta(d):
    p = os.path.join(d, "episode.json")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def read_joints(d):
    p = os.path.join(d, "joints.csv")
    if not os.path.isfile(p):
        return None
    ts, ps = [], []
    with open(p, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                tv = float(row["t"])
                pv = [float(row["pos%d" % j]) for j in range(6)]
            except (KeyError, TypeError, ValueError):
                continue
            ts.append(tv)
            ps.append(pv)
    if len(ts) < 50:
        return None
    t = np.asarray(ts, dtype=np.float64)
    pos = np.asarray(ps, dtype=np.float64)
    order = np.argsort(t, kind="stable")
    t, pos = t[order], pos[order]
    keep = np.concatenate([[True], np.diff(t) > 0])
    return t[keep], pos[keep]


def joint_speed(t, pos):
    v = np.gradient(pos, t, axis=0)
    s = np.sqrt(np.sum(v * v, axis=1))
    return np.convolve(np.pad(s, 1, mode="edge"), np.ones(3) / 3.0, mode="valid")


def smooth3(v):
    if v.size < 3:
        return v
    return np.convolve(np.pad(v, 1, mode="edge"), np.ones(3) / 3.0, mode="valid")


def on_grid(t_src, v_src, grid):
    inside = (grid >= t_src[0]) & (grid <= t_src[-1])
    return np.interp(grid, t_src, v_src), inside


def iter_frames(seg, a, b):
    c = av.open(seg["path"])
    try:
        st = c.streams.video[0]
        st.thread_count = 2
        tb = float(st.time_base)
        base = seg["base"]
        target = int(round((a - base) / tb))
        if target > 0:
            try:
                c.seek(target, stream=st, backward=True)
            except av.error.FFmpegError:
                pass
        last = None
        for pkt in c.demux(st):
            if pkt.size == 0 or pkt.pts is None:
                continue
            try:
                frames = pkt.decode()
            except av.error.FFmpegError:
                continue
            for fr in frames:
                if fr.pts is None:
                    continue
                t = fr.pts * tb + base
                if t >= b:
                    return
                if last is not None and t <= last:
                    continue
                last = t
                yield t, fr
    finally:
        c.close()


def energies(seg, a, b):
    times, e1, e2 = [], [], []
    prev, prev_t = None, None
    for t, fr in iter_frames(seg, a, b):
        g = fr.reformat(width=160, height=120, format="gray", interpolation="AREA").to_ndarray().astype(np.float32)
        g = cv2.GaussianBlur(g, (3, 3), 0.0)
        g -= g.mean()
        if prev is not None:
            mid = 0.5 * (prev_t + t)
            if mid >= a:
                d = np.abs(g - prev)
                times.append(mid)
                e1.append(float(d.mean()))
                e2.append(float((d > 12.0).mean()))
        prev, prev_t = g, t
    return times, e1, e2


def camera_series(segs, a, b):
    T, E1, E2, used = [], [], [], []
    for s in segs:
        if s["start"] >= b or s["end"] <= a:
            continue
        ts, v1, v2 = energies(s, max(a, s["start"]), min(b, s["end"]))
        if ts:
            used.append(s["name"])
            T.extend(ts)
            E1.extend(v1)
            E2.extend(v2)
    T = np.asarray(T, dtype=np.float64)
    E1 = np.asarray(E1, dtype=np.float64)
    E2 = np.asarray(E2, dtype=np.float64)
    order = np.argsort(T, kind="stable")
    T, E1, E2 = T[order], E1[order], E2[order]
    if T.size:
        keep = np.concatenate([[True], np.diff(T) > 0])
        T, E1, E2 = T[keep], E1[keep], E2[keep]
    return T, E1, E2, used


def xcorr(sj, mj, sc, mc):
    idx = np.nonzero(mj)[0]
    lags = np.arange(-STEPS, STEPS + 1)
    r = np.full(lags.size, np.nan)
    n = np.zeros(lags.size, dtype=np.int64)
    for k, m in enumerate(lags):
        j2 = idx + m
        ok = (j2 >= 0) & (j2 < sc.size)
        i1 = idx[ok]
        j2 = j2[ok]
        good = mc[j2]
        x = sj[i1[good]]
        y = sc[j2[good]]
        if x.size < 200:
            continue
        x = x - x.mean()
        y = y - y.mean()
        den = np.sqrt(np.sum(x * x) * np.sum(y * y))
        if den > 0:
            r[k] = np.sum(x * y) / den
            n[k] = x.size
    return lags, r, n


def refine_peak(lags, r):
    k = int(np.nanargmax(r))
    delta = 0.0
    if 0 < k < r.size - 1 and np.all(np.isfinite(r[k - 1:k + 2])):
        y0, y1, y2 = r[k - 1], r[k], r[k + 1]
        den = y0 - 2.0 * y1 + y2
        if den != 0:
            delta = 0.5 * (y0 - y2) / den
    return (lags[k] + delta) * GRID * 1000.0, float(r[k]), k


def thin(times, gap):
    kept = []
    for v in times:
        if not kept or v - kept[-1] >= gap:
            kept.append(float(v))
    return np.asarray(kept, dtype=np.float64)


def crossings(x, t, frac=0.3, gap=0.25):
    if x.size < 20:
        return np.zeros(0), np.zeros(0)
    lo, hi = np.percentile(x, 10), np.percentile(x, 95)
    thr = lo + frac * (hi - lo)
    up = np.nonzero((x[1:] >= thr) & (x[:-1] < thr))[0] + 1
    dn = np.nonzero((x[1:] < thr) & (x[:-1] >= thr))[0] + 1
    return thin(t[up], gap), thin(t[dn], gap)


def paired_diffs(ref, other, win=0.5):
    out = []
    if other.size == 0:
        return out
    for v in ref:
        k = int(np.argmin(np.abs(other - v)))
        if abs(other[k] - v) <= win:
            out.append(float(other[k] - v))
    return out


def analyze_pair(t, pos, cam_T, e1, e2):
    s = joint_speed(t, pos)
    grid = np.arange(t[0] - 0.6, t[-1] + 0.6, GRID)
    sj, mj = on_grid(t, s, grid)
    sj = np.where(mj, sj, 0.0)
    sc1, mc = on_grid(cam_T, e1, grid)
    sc2, _ = on_grid(cam_T, e2, grid)
    lags, r1, n1 = xcorr(sj, mj, sc1, mc)
    if not np.any(np.isfinite(r1)):
        raise RuntimeError("no valid correlation (primary energy)")
    lag_ms, peak, k = refine_peak(lags, r1)
    _, r2, _ = xcorr(sj, mj, sc2, mc)
    lag_frac, peak_frac = float("nan"), float("nan")
    if np.any(np.isfinite(r2)):
        lag_frac, peak_frac, _ = refine_peak(lags, r2)
    mid = 0.5 * (t[0] + t[-1])
    _, rh1, _ = xcorr(sj, mj & (grid <= mid), sc1, mc)
    _, rh2, _ = xcorr(sj, mj & (grid > mid), sc1, mc)
    lag_h1 = refine_peak(lags, rh1)[0] if np.any(np.isfinite(rh1)) else float("nan")
    lag_h2 = refine_peak(lags, rh2)[0] if np.any(np.isfinite(rh2)) else float("nan")
    idx = np.nonzero(mj)[0]
    tg = grid[idx]
    xj = sj[idx]
    valid = mc[idx]
    tcg = tg[valid]
    xc = sc1[idx][valid]
    j_on, j_off = crossings(xj, tg)
    c_on, c_off = crossings(xc, tcg)
    diffs = paired_diffs(j_on, c_on) + paired_diffs(j_off, c_off)
    ev = float(np.median(diffs)) * 1000.0 if len(diffs) >= 3 else float("nan")
    if np.isfinite(ev):
        agree = "yes" if abs(ev - lag_ms) <= 50.0 else "no"
    else:
        agree = "na"
    res = {"lag_ms": float(lag_ms), "lag_frames": float(lag_ms) * 30.0 / 1000.0, "peak": peak,
           "n": int(n1[k]), "event_lag_ms": ev, "r_zero_lag": float(r1[STEPS]), "event_pairs": len(diffs),
           "agree_50ms": agree, "edge": bool(abs(lag_ms) >= MAXLAG_MS - 15), "lag_h1_ms": float(lag_h1),
           "lag_h2_ms": float(lag_h2), "lag_frac_ms": float(lag_frac), "peak_frac": float(peak_frac)}
    return res, (lags * GRID * 1000.0, r1)


def inventory():
    segs = {c: segments(c) for c in CAMERAS}
    for c in CAMERAS:
        rel = sum(1 for s in segs[c] if s["relative"])
        print("SEGMENTS %s count=%d relative=%d absolute=%d" % (c, len(segs[c]), rel, len(segs[c]) - rel), flush=True)
    for run, ep, d in episodes():
        meta = read_meta(d)
        j = read_joints(d)
        line = "EP %s/%s json=%s joints=%s" % (run, ep, meta is not None, j is not None)
        if meta:
            line += " success=%s plate=%s t_start=%s t_end=%s task=%r" % (
                meta.get("success"), meta.get("plate"), meta.get("t_start"), meta.get("t_end"), meta.get("task"))
        print(line, flush=True)
        if j is None:
            continue
        t, pos = j
        print("   joints n=%d t0=%.4f (%s) t1=%.4f (%s) span=%.2f s hz=%.2f" % (
            t.size, t[0], fmt_utc(t[0]), t[-1], fmt_utc(t[-1]), t[-1] - t[0], 1.0 / np.median(np.diff(t))), flush=True)
        for cam in CAMERAS:
            cov = [s for s in segs[cam] if s["start"] < t[-1] + PAD and s["end"] > t[0] - PAD]
            desc = ", ".join("%s(%s)" % (s["name"], "rel" if s["relative"] else "abs") for s in cov)
            print("   %s cover=%d: %s" % (cam, len(cov), desc), flush=True)


def analyze():
    segs = {c: segments(c) for c in CAMERAS}
    rows, curves, detail, skipped = [], {c: [] for c in CAMERAS}, {}, []
    for run, ep, d in episodes():
        key = "%s/%s" % (run, ep)
        j = read_joints(d)
        if j is None:
            skipped.append("%s: joints.csv missing or unreadable" % key)
            continue
        t, pos = j
        meta = read_meta(d) or {}
        detail[key] = {"joint_samples": int(t.size), "t0": float(t[0]), "t1": float(t[-1]),
                       "joint_hz": float(1.0 / np.median(np.diff(t))), "success": meta.get("success"),
                       "plate": meta.get("plate"), "task": meta.get("task"),
                       "t_start": meta.get("t_start"), "t_end": meta.get("t_end"), "cams": {}}
        for cam in CAMERAS:
            cam_T, e1, e2, used = camera_series(segs[cam], t[0] - PAD, t[-1] + PAD)
            if cam_T.size < 100:
                skipped.append("%s %s: camera coverage %d frames" % (key, cam, cam_T.size))
                continue
            rel = any(s["relative"] for s in segs[cam] if s["name"] in used)
            dup_frac = float(np.mean(e1 < 0.05 * np.median(e1)))
            max_gap_ms = float(np.max(np.diff(cam_T)) * 1000.0)
            try:
                res, curve = analyze_pair(t, pos, cam_T, smooth3(e1), smooth3(e2))
            except Exception as exc:
                skipped.append("%s %s: %s" % (key, cam, exc))
                traceback.print_exc()
                continue
            res.update({"episode": key, "camera": cam, "clock_ref": "relative" if rel else "absolute",
                        "cam_frames": int(cam_T.size), "cam_dt_ms": float(np.median(np.diff(cam_T)) * 1000.0),
                        "max_gap_ms": max_gap_ms, "dup_frac": dup_frac, "segments": ";".join(used)})
            rows.append(res)
            curves[cam].append((key, curve[0], curve[1]))
            detail[key]["cams"][cam] = {"lag_ms": res["lag_ms"], "segments": used, "relative": rel}
            print("%s %s lag=%.1f peak=%.3f n=%d h1=%.1f h2=%.1f frac_lag=%.1f frac_peak=%.3f ev=%.1f pairs=%d "
                  "dup=%.4f maxgap=%.1f clock=%s" % (
                      key, cam, res["lag_ms"], res["peak"], res["n"], res["lag_h1_ms"], res["lag_h2_ms"],
                      res["lag_frac_ms"], res["peak_frac"], res["event_lag_ms"], res["event_pairs"],
                      dup_frac, max_gap_ms, res["clock_ref"]), flush=True)
    write_outputs(rows, curves, detail, skipped)


def write_outputs(rows, curves, detail, skipped):
    os.makedirs(PLOTS, exist_ok=True)
    with open(os.path.join(WORK, "offsets.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        for r in rows:
            w.writerow([r["episode"], r["camera"], "%.2f" % r["lag_ms"], "%.3f" % r["lag_frames"],
                        "%.4f" % r["peak"], r["n"], num(r["event_lag_ms"], 2), "%.4f" % r["r_zero_lag"],
                        r["event_pairs"], r["agree_50ms"], r["clock_ref"], "yes" if r["edge"] else "no",
                        r["cam_frames"], "%.2f" % r["cam_dt_ms"], "%.2f" % r["max_gap_ms"],
                        "%.4f" % r["dup_frac"], num(r["lag_h1_ms"], 2), num(r["lag_h2_ms"], 2),
                        num(r["lag_frac_ms"], 2), num(r["peak_frac"], 4), r["segments"]])
    with open(os.path.join(WORK, "analysis_details.json"), "w", encoding="utf-8") as fh:
        json.dump(clean({"episodes": detail, "skipped": skipped}), fh, indent=1)
    delta = {}
    for r in rows:
        if r["episode"] == "any_1740/ep_0017" and r["camera"] == "wrist":
            delta["any_1740/ep_0017 wrist lag minus P2 goal5 value, ms"] = (
                r["lag_ms"] - P2_PREVIOUS["any_1740/ep_0017 wrist vs goal5, lag_ms"])
            delta["any_1740/ep_0017 wrist lag minus P2 pos5 value, ms"] = (
                r["lag_ms"] - P2_PREVIOUS["any_1740/ep_0017 wrist vs pos5, lag_ms"])
    doc = {"stage": "clock-offsets-v1", "utc": utc_now(), "metrics": summarise(rows),
           "previous": P2_PREVIOUS, "delta": delta}
    with open(os.path.join(WORK, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(clean(doc), fh, indent=1)
    make_plots(rows, curves)
    print("WROTE offsets.csv rows=%d skipped=%d" % (len(rows), len(skipped)), flush=True)
    for s in skipped:
        print("SKIPPED", s, flush=True)


def num(v, nd):
    return ("%." + str(nd) + "f") % v if np.isfinite(v) else "nan"


def summarise(rows):
    out = {"rule": "reliable = absolute clock, peak >= %.1f, lag not at search edge" % RELIABLE_PEAK,
           "frame_ms": FRAME_MS, "cameras": {}}
    by_key = {}
    for cam in CAMERAS:
        sel = [r for r in rows if r["camera"] == cam and r["clock_ref"] == "absolute"]
        rel_ok = [r for r in sel if r["peak"] >= RELIABLE_PEAK and not r["edge"]]
        for r in rel_ok:
            by_key[(r["episode"], cam)] = r["lag_ms"]
        entry = {"pairs_all": len(sel), "pairs_reliable": len(rel_ok),
                 "pairs_unreliable_episodes": sorted(r["episode"] for r in sel if r not in rel_ok)}
        if sel:
            al = np.array([r["lag_ms"] for r in sel])
            entry.update({"all_median_lag_ms": float(np.median(al)), "all_min_lag_ms": float(al.min()),
                          "all_max_lag_ms": float(al.max())})
        if rel_ok:
            lags = np.array([r["lag_ms"] for r in rel_ok])
            q25, q50, q75 = np.percentile(lags, [25, 50, 75])
            entry.update({"median_lag_ms": float(q50), "p25_lag_ms": float(q25), "p75_lag_ms": float(q75),
                          "iqr_lag_ms": float(q75 - q25), "min_lag_ms": float(lags.min()),
                          "max_lag_ms": float(lags.max()),
                          "std_lag_ms": float(lags.std(ddof=1)) if lags.size > 1 else None,
                          "within_1_frame_of_median": int(np.sum(np.abs(lags - q50) <= FRAME_MS)),
                          "median_peak": float(np.median([r["peak"] for r in rel_ok])),
                          "split_half_abs_diff_median_ms": float(np.median(
                              [abs(r["lag_h1_ms"] - r["lag_h2_ms"]) for r in rel_ok])),
                          "event_agree_50ms": int(sum(r["agree_50ms"] == "yes" for r in rel_ok)),
                          "event_with_estimate": int(sum(r["agree_50ms"] != "na" for r in rel_ok)),
                          "frac_variant_median_lag_ms": float(np.median([r["lag_frac_ms"] for r in rel_ok])),
                          "dup_frac_median": float(np.median([r["dup_frac"] for r in sel]))})
        out["cameras"][cam] = entry
    diffs = {}
    for other in ("top", "close"):
        pairs = [(ep, v - by_key[(ep, "wrist")]) for (ep, c), v in by_key.items()
                 if c == other and (ep, "wrist") in by_key]
        vals = np.array([p[1] for p in pairs]) if pairs else np.zeros(0)
        diffs[other + "_minus_wrist"] = {
            "episodes": len(pairs),
            "median_ms": float(np.median(vals)) if vals.size else None,
            "std_ms": float(vals.std(ddof=1)) if vals.size > 1 else None,
            "per_episode_ms": {p[0]: round(float(p[1]), 2) for p in sorted(pairs)}}
    out["paired_differences_reliable"] = diffs
    return out


def draw_plot(title, curves, xlim, ylim, vlines=(), points=(), xlabel="", ylabel="", w=900, h=440):
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    left, right, top, bottom = 70, w - 20, 40, h - 50
    if ylim[1] <= ylim[0]:
        ylim = (ylim[0] - 1.0, ylim[0] + 1.0)

    def px(x):
        return left + (np.asarray(x, dtype=np.float64) - xlim[0]) / (xlim[1] - xlim[0]) * (right - left)

    def py(y):
        return bottom - (np.asarray(y, dtype=np.float64) - ylim[0]) / (ylim[1] - ylim[0]) * (bottom - top)

    cv2.rectangle(canvas, (left, top), (right, bottom), (0, 0, 0), 1)
    for k in range(6):
        xv = xlim[0] + k * (xlim[1] - xlim[0]) / 5.0
        x_px = int(px(xv))
        cv2.line(canvas, (x_px, bottom), (x_px, bottom + 5), (0, 0, 0), 1)
        cv2.putText(canvas, "%g" % round(xv, 2), (x_px - 18, bottom + 20), FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        yv = ylim[0] + k * (ylim[1] - ylim[0]) / 5.0
        y_px = int(py(yv))
        cv2.line(canvas, (left - 5, y_px), (left, y_px), (0, 0, 0), 1)
        cv2.putText(canvas, "%.2f" % yv, (2, y_px + 5), FONT, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    for xs, ys, colour, thick in curves:
        pts = np.stack([px(xs), py(ys)], axis=1)
        pts = np.clip(pts, [0, 0], [w - 1, h - 1]).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], False, colour, thick, cv2.LINE_AA)
    for xv, colour in vlines:
        x_px = int(px(xv))
        for y0 in range(top, bottom, 8):
            cv2.line(canvas, (x_px, y0), (x_px, min(y0 + 4, bottom)), colour, 1)
    for xv, yv, colour in points:
        cv2.circle(canvas, (int(px(xv)), int(py(yv))), 4, colour, -1, cv2.LINE_AA)
    cv2.putText(canvas, title, (left, 22), FONT, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, xlabel, (left + (right - left) // 4, h - 12), FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, ylabel, (2, top - 10), FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def make_plots(rows, curves):
    for cam in CAMERAS:
        cs = curves[cam]
        if not cs:
            continue
        stack = np.stack([c[2] for c in cs])
        lo, hi = float(np.nanmin(stack)), float(np.nanmax(stack))
        lines = [(c[1], c[2], (200, 200, 200), 1) for c in cs]
        lines.append((cs[0][1], np.nanmedian(stack, axis=0), (0, 0, 0), 3))
        img = draw_plot("xcorr %s: grey = each episode, black = median" % cam, lines,
                        (-MAXLAG_MS, MAXLAG_MS), (lo - 0.02, hi + 0.02),
                        vlines=[(0.0, (0, 0, 255))], xlabel="lag ms (camera minus joint)", ylabel="Pearson r")
        cv2.imwrite(os.path.join(PLOTS, "xcorr_%s.png" % cam), img)
    keys = sorted(set(r["episode"] for r in rows))
    pts = []
    for r in rows:
        x = keys.index(r["episode"]) + {"wrist": -0.2, "top": 0.0, "close": 0.2}[r["camera"]]
        pts.append((x, r["lag_ms"], CAM_COLOUR[r["camera"]]))
    if keys:
        guides = [([-0.5, len(keys) - 0.5], [FRAME_MS, FRAME_MS], (160, 160, 160), 1),
                  ([-0.5, len(keys) - 0.5], [-FRAME_MS, -FRAME_MS], (160, 160, 160), 1)]
        img = draw_plot("lag per episode (wrist blue, top green, close red; grey = +/-1 frame)", guides,
                        (-0.5, len(keys) - 0.5), (-520, 520), vlines=[], points=pts,
                        xlabel="episode index: " + ", ".join("%d=%s" % (i, k.replace("any_", "")) for i, k in
                                                             enumerate(keys))[:150], ylabel="lag ms")
        cv2.imwrite(os.path.join(PLOTS, "lag_by_episode.png"), img)


def visual(run, ep, cam, lag_ms):
    d = os.path.join(SORT, run, ep)
    t, pos = read_joints(d)
    grid = np.arange(t[0], t[-1], GRID)
    sg, _ = on_grid(t, joint_speed(t, pos), grid)
    top_speed = np.percentile(sg, 95)
    on_t, _ = crossings(sg, grid)
    t_on = None
    for x in on_t:
        if x <= t[0] + 0.5:
            continue
        k0 = int(np.searchsorted(grid, x))
        k1 = int(np.searchsorted(grid, x + 0.5))
        if sg[k0:k1].max() >= 0.5 * top_speed:
            t_on = float(x)
            break
    if t_on is None:
        print("no joint onset found for", run, ep)
        return
    a, b = t_on - 0.4, t_on + 0.9
    tiles, e_t, e_v = [], [], []
    for sgm in segments(cam):
        if sgm["start"] >= b or sgm["end"] <= a:
            continue
        lo_w, hi_w = max(a, sgm["start"]), min(b, sgm["end"])
        for tt, fr in iter_frames(sgm, lo_w, hi_w):
            if tt >= a:
                tiles.append((tt, fr.reformat(width=240, height=180, format="rgb24",
                                              interpolation="AREA").to_ndarray()))
        et, ev, _ = energies(sgm, lo_w, hi_w)
        e_t += et
        e_v += ev
    sel = [(tt, im) for tt, im in tiles if t_on - 0.2 <= tt <= t_on + 0.6][:16]
    e_t = np.asarray(e_t, dtype=np.float64)
    e_v = np.asarray(e_v, dtype=np.float64)
    win = (e_t >= t_on - 0.2) & (e_t <= t_on + 0.6)
    e_on = float("nan")
    if win.sum() > 3:
        base_e, top_e = np.percentile(e_v[win], 10), np.percentile(e_v[win], 95)
        cand = e_t[win][e_v[win] >= base_e + 0.3 * (top_e - base_e)]
        if cand.size:
            e_on = float(cand[0])
    tw, th, cols, rws, panel_h = 240, 180, 8, 2, 240
    grid_img = np.full((rws * th, cols * tw, 3), 255, dtype=np.uint8)
    for k, (tt, im) in enumerate(sel):
        r_, c_ = divmod(k, cols)
        grid_img[r_ * th:(r_ + 1) * th, c_ * tw:(c_ + 1) * tw] = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.putText(grid_img, "%+d ms" % round((tt - t_on) * 1000.0), (c_ * tw + 4, r_ * th + 18),
                    FONT, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    w_win = (grid >= t_on - 0.2) & (grid <= t_on + 0.6)
    xs_j = (grid[w_win] - t_on) * 1000.0
    ys_j = sg[w_win]
    ys_j = (ys_j - ys_j.min()) / max(ys_j.max() - ys_j.min(), 1e-9)
    xs_e = (e_t[win] - t_on) * 1000.0
    ys_e = e_v[win]
    ys_e = (ys_e - ys_e.min()) / max(ys_e.max() - ys_e.min(), 1e-9)
    vlines = [(0.0, (255, 0, 0)), (float(lag_ms), (0, 0, 255))]
    if np.isfinite(e_on):
        vlines.append(((e_on - t_on) * 1000.0, (255, 0, 255)))
    panel = draw_plot("joint speed (blue; 0 ms = joint onset) and camera energy (green); red = predicted image onset",
                      [(xs_j, ys_j, (255, 0, 0), 2), (xs_e, ys_e, (0, 160, 0), 2)],
                      (-200.0, 600.0), (-0.05, 1.05), vlines=vlines, xlabel="ms from joint onset",
                      ylabel="normalised", w=cols * tw, h=panel_h)
    canvas = np.vstack([grid_img, panel])
    out_png = os.path.join(PLOTS, "visual_%s_%s_%s.png" % (run.replace("any_", "any"), ep, cam))
    cv2.imwrite(out_png, canvas)
    print("VISUAL %s/%s %s joint_onset_epoch=%.4f (%s) lag_ms=%.2f predicted_image_onset_ms=%.1f" % (
        run, ep, cam, t_on, fmt_utc(t_on), lag_ms, lag_ms), flush=True)
    print("  energy_onset_ms=%s" % ("%.1f" % ((e_on - t_on) * 1000.0) if np.isfinite(e_on) else "nan"), flush=True)
    print("  tiles_ms=%s" % ",".join("%+d" % round((tt - t_on) * 1000.0) for tt, _ in sel), flush=True)
    print("  png=%s" % out_png, flush=True)


def visual2(run, ep, cam, lag_ms):
    d = os.path.join(SORT, run, ep)
    t, pos = read_joints(d)
    grid = np.arange(t[0], t[-1], GRID)
    sg, _ = on_grid(t, joint_speed(t, pos), grid)
    on_t, _ = crossings(sg, grid)
    best, best_peak = None, -1.0
    for x in on_t:
        if x <= t[0] + 0.5 or x >= t[-1] - 0.6:
            continue
        k0 = int(np.searchsorted(grid, x))
        k1 = int(np.searchsorted(grid, x + 0.5))
        pk = float(sg[k0:k1].max())
        if pk > best_peak:
            best, best_peak = float(x), pk
    if best is None:
        print("no joint onset found for", run, ep)
        return
    t_on = best
    a, b = t_on - 0.4, t_on + 0.9
    tiles, e_t, e_v = [], [], []
    for sgm in segments(cam):
        if sgm["start"] >= b or sgm["end"] <= a:
            continue
        lo_w, hi_w = max(a, sgm["start"]), min(b, sgm["end"])
        for tt, fr in iter_frames(sgm, lo_w, hi_w):
            if tt >= a:
                tiles.append((tt, fr.reformat(width=240, height=180, format="rgb24",
                                              interpolation="AREA").to_ndarray()))
        et, ev, _ = energies(sgm, lo_w, hi_w)
        e_t += et
        e_v += ev
    idx_sel = [i for i, (tt, _) in enumerate(tiles) if t_on - 0.2 <= tt <= t_on + 0.6][:16]
    tw, th, cols = 240, 180, 8
    rgb_img = np.full((2 * th, cols * tw, 3), 255, dtype=np.uint8)
    dif_img = np.full((2 * th, cols * tw, 3), 255, dtype=np.uint8)
    diff_means = []
    for k, i in enumerate(idx_sel):
        tt, im = tiles[i]
        r_, c_ = divmod(k, cols)
        rgb_img[r_ * th:(r_ + 1) * th, c_ * tw:(c_ + 1) * tw] = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.putText(rgb_img, "%+d ms" % round((tt - t_on) * 1000.0), (c_ * tw + 4, r_ * th + 18),
                    FONT, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        if i == 0:
            continue
        t_prev, im_prev = tiles[i - 1]
        g1 = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.int16)
        g0 = cv2.cvtColor(im_prev, cv2.COLOR_RGB2GRAY).astype(np.int16)
        dd = np.abs(g1 - g0)
        mid_ms = round((0.5 * (tt + t_prev) - t_on) * 1000.0)
        diff_means.append((mid_ms, float(dd.mean())))
        dimg = cv2.cvtColor(np.clip(dd * 4, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        dif_img[r_ * th:(r_ + 1) * th, c_ * tw:(c_ + 1) * tw] = dimg
        cv2.putText(dif_img, "d%+d ms" % mid_ms, (c_ * tw + 4, r_ * th + 18),
                    FONT, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    e_t = np.asarray(e_t, dtype=np.float64)
    e_v = np.asarray(e_v, dtype=np.float64)
    win = (e_t >= t_on - 0.2) & (e_t <= t_on + 0.6)
    e_on = float("nan")
    if win.sum() > 3:
        base_e, top_e = np.percentile(e_v[win], 10), np.percentile(e_v[win], 95)
        cand = e_t[win][e_v[win] >= base_e + 0.3 * (top_e - base_e)]
        if cand.size:
            e_on = float(cand[0])
    w_win = (grid >= t_on - 0.2) & (grid <= t_on + 0.6)
    xs_j = (grid[w_win] - t_on) * 1000.0
    ys_j = sg[w_win]
    ys_j = (ys_j - ys_j.min()) / max(ys_j.max() - ys_j.min(), 1e-9)
    curves = [(xs_j, ys_j, (255, 0, 0), 2)]
    if win.sum() > 1:
        xs_e = (e_t[win] - t_on) * 1000.0
        ys_e = e_v[win]
        ys_e = (ys_e - ys_e.min()) / max(ys_e.max() - ys_e.min(), 1e-9)
        curves.append((xs_e, ys_e, (0, 160, 0), 2))
    vlines = [(0.0, (255, 0, 0)), (float(lag_ms), (0, 0, 255))]
    if np.isfinite(e_on):
        vlines.append(((e_on - t_on) * 1000.0, (255, 0, 255)))
    panel = draw_plot("joint speed (blue; 0 ms = joint onset) and camera energy (green); red = predicted image onset",
                      curves, (-200.0, 600.0), (-0.05, 1.05), vlines=vlines, xlabel="ms from joint onset",
                      ylabel="normalised", w=cols * tw, h=240)
    canvas = np.vstack([rgb_img, dif_img, panel])
    out_png = os.path.join(PLOTS, "visual2_%s_%s_%s.png" % (run.replace("any_", "any"), ep, cam))
    cv2.imwrite(out_png, canvas)
    print("VISUAL2 %s/%s %s joint_onset_epoch=%.4f (%s) joint_peak_speed=%.3f lag_ms=%.2f" % (
        run, ep, cam, t_on, fmt_utc(t_on), best_peak, lag_ms), flush=True)
    print("  energy_onset_ms=%s" % ("%.1f" % ((e_on - t_on) * 1000.0) if np.isfinite(e_on) else "nan"), flush=True)
    print("  rgb_tiles_ms=%s" % ",".join("%+d" % round((tt - t_on) * 1000.0) for tt, _ in
                                       [tiles[i] for i in idx_sel]), flush=True)
    print("  diff_mean_by_pair_midpoint_ms=%s" % ",".join("%+d:%.2f" % (m, v) for m, v in diff_means), flush=True)
    print("  png=%s" % out_png, flush=True)


def main(argv):
    mode = argv[1] if len(argv) > 1 else ""
    os.makedirs(PLOTS, exist_ok=True)
    if mode == "inventory":
        inventory()
    elif mode == "analyze":
        analyze()
    elif mode == "visual2" and len(argv) == 6:
        visual2(argv[2], argv[3], argv[4], float(argv[5]))
    elif mode == "visual" and len(argv) == 6:
        visual(argv[2], argv[3], argv[4], float(argv[5]))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
