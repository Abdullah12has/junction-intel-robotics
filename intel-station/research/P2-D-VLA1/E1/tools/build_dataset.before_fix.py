#!/usr/bin/env python3
# Build a LeRobot v3.0 dataset (Studio layout) from finished sort-cycle episodes.
# Row i of an episode = time t_first + i / 30 on a uniform 30 fps grid over the joints.csv samples with t_start <= t <= t_end.
#   action = goal0..goal5 and observation.state = pos0..pos5, linearly interpolated to the grid; timestamp = i / 30.
# Camera frame of row i = decoded camera frame nearest to grid time t_first + i / 30, re-encoded as AV1 with the team's settings.
# variant team: realsense close + top (the team's keys). variant full: + wrist (key observation.images.wrist, new).
# Writes OUT (dataset root) and OUT.build_report.json next to it. Idempotent: OUT is removed and rebuilt.

import argparse
import copy
import csv
import glob
import json
import os
import shutil
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.configs.video import VideoEncoderConfig
from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats, sample_indices

ROOT = "/home/intel/intel-junction-2026"
TEMPLATE = "data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af"
FPS = 30
PAD = 0.5
WRIST_SEG_S = 60.0
CAM_KEY = {
    "close": "observation.images.realsense 243722064316",
    "top": "observation.images.realsense 243622061421",
    "wrist": "observation.images.wrist",
}
ACTION_COLS = ["goal%d" % j for j in range(6)]
STATE_COLS = ["pos%d" % j for j in range(6)]
STAT_ORDER = ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]
FEATURE_ORDER = ["action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index"]


def utc_seconds(stamp):
    return datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()


def list_media(folder, prefix):
    found = []
    for path in glob.glob(os.path.join(folder, prefix + "_*.mkv")):
        name = os.path.basename(path)
        found.append((utc_seconds(name[len(prefix) + 1:-4]), path))
    return sorted(found)


def candidates(files, a, b, span):
    out = []
    for k, (ts, path) in enumerate(files):
        end = files[k + 1][0] if k + 1 < len(files) else float("inf")
        if span is not None:
            end = min(end, ts + span)
        if ts <= b and end > a:
            out.append((path, ts, end))
    return out


def open_video(path):
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    relative = (container.start_time or 0) / 1e6 < 1e9
    return container, stream, relative


def has_packet_at_or_after(path, ts, t):
    try:
        container, stream, relative = open_video(path)
    except av.error.FFmpegError:
        return False
    try:
        base = ts if relative else 0.0
        tb = float(stream.time_base)
        target = int(round((t - base) / tb))
        if target > 0:
            try:
                container.seek(target, stream=stream, backward=True)
            except av.error.FFmpegError:
                return False
        for pkt in container.demux(stream):
            if pkt.pts is not None and pkt.pts * tb + base >= t:
                return True
        return False
    finally:
        container.close()


def decode_window(files, a, b):
    for path, ts, end in files:
        container, stream, relative = open_video(path)
        base = ts if relative else 0.0
        tb = float(stream.time_base)
        stop = False
        past = False
        try:
            target = int(round((a - base) / tb))
            if target > 0:
                try:
                    container.seek(target, stream=stream, backward=True)
                except av.error.FFmpegError:
                    pass
            for pkt in container.demux(stream):
                if pkt.pts is None:
                    continue
                try:
                    decoded = pkt.decode()
                except av.error.FFmpegError:
                    continue
                for fr in decoded:
                    if fr.pts is None:
                        continue
                    t = fr.pts * tb + base
                    if t > b:
                        past = True
                        stop = True
                        break
                    if t >= end:
                        stop = True
                        break
                    if t >= a:
                        yield t, fr
                if stop:
                    break
        finally:
            container.close()
        if past:
            return


def nearest_frames(times, frames):
    it = iter(frames)
    cur = next(it, None)
    if cur is None:
        return
    nxt = next(it, None)
    for i, ti in enumerate(times):
        while nxt is not None and abs(nxt[0] - ti) <= abs(cur[0] - ti):
            cur, nxt = nxt, next(it, None)
        yield i, ti, cur[0], cur[1]


def encode_camera(job):
    cam, out_path, width, height, options, episodes = job
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    container = av.open(out_path, mode="w")
    stream = container.add_stream("libsvtav1", rate=FPS, options=options)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    results = []
    for ep in episodes:
        times = ep["times"]
        n = len(times)
        keep = set(int(k) for k in sample_indices(n))
        small = []
        dts = np.zeros(n, dtype=np.float64)
        frames = decode_window(ep["files"], float(times[0]) - PAD, float(times[-1]) + PAD)
        count = 0
        for i, ti, tc, fr in nearest_frames(times, frames):
            out = fr.reformat(format="yuv420p")
            out.pts = ep["row0"] + i
            out.time_base = Fraction(1, FPS)
            for pkt in stream.encode(out):
                container.mux(pkt)
            if i in keep:
                small.append(fr.reformat(width=160, height=120, format="rgb24", interpolation="AREA").to_ndarray())
            dts[i] = tc - ti
            count += 1
        frames.close()
        if count != n:
            raise RuntimeError("%s %s: encoded %d of %d rows, video does not cover the episode" % (cam, ep["name"], count, n))
        pixels = np.ascontiguousarray(np.stack(small).transpose(0, 3, 1, 2))
        st = get_feature_stats(pixels, axis=(0, 2, 3), keepdims=True)
        st = {k: (v if k == "count" else np.squeeze(v / 255.0, axis=0)) for k, v in st.items()}
        results.append({"name": ep["name"], "n": n, "dt_max_s": float(np.abs(dts).max()),
                        "dt_mean_s": float(np.abs(dts).mean()),
                        "rows_dt_over_1_frame": int(np.sum(np.abs(dts) > 1.0 / FPS + 1e-9)), "stats": st})
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    return {"cam": cam, "mp4": out_path, "episodes": results}


def load_episodes(run_dirs, only_success, skipped):
    episodes = []
    for run in run_dirs:
        run = os.path.abspath(run)
        run_name = os.path.basename(run.rstrip("/"))
        for ep_dir in sorted(glob.glob(os.path.join(run, "ep_*"))):
            name = os.path.basename(ep_dir)
            meta_path = os.path.join(ep_dir, "episode.json")
            joints_path = os.path.join(ep_dir, "joints.csv")
            if not (os.path.isfile(meta_path) and os.path.isfile(joints_path)):
                skipped.append({"episode": name, "run": run_name, "reason": "not finished (episode.json or joints.csv missing)"})
                continue
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            if only_success and meta.get("success") is not True:
                skipped.append({"episode": name, "run": run_name, "reason": "success is not true"})
                continue
            if meta.get("t_start") is None or meta.get("t_end") is None or not meta.get("task"):
                skipped.append({"episode": name, "run": run_name, "reason": "episode.json lacks t_start, t_end or task"})
                continue
            t_start, t_end = float(meta["t_start"]), float(meta["t_end"])
            with open(joints_path, newline="", encoding="utf-8") as fh:
                rows = [r for r in csv.DictReader(fh)]
            sel = [r for r in rows if t_start <= float(r["t"]) <= t_end]
            if len(sel) < 2:
                skipped.append({"episode": name, "run": run_name, "reason": "fewer than 2 joint rows in t_start..t_end"})
                continue
            t = np.array([float(r["t"]) for r in sel], dtype=np.float64)
            gaps = np.diff(t)
            if np.any(gaps <= 0):
                skipped.append({"episode": name, "run": run_name, "reason": "joint times are not strictly increasing"})
                continue
            grid = t[0] + np.arange(int(np.floor((t[-1] - t[0]) * FPS + 1e-9)) + 1) / FPS
            if len(grid) < 2:
                skipped.append({"episode": name, "run": run_name, "reason": "fewer than 2 grid rows at FPS"})
                continue
            src = np.array([[float(r[c]) for c in ACTION_COLS + STATE_COLS] for r in sel], dtype=np.float64)
            resampled = np.stack([np.interp(grid, t, src[:, j]) for j in range(src.shape[1])], axis=1).astype(np.float32)
            episodes.append({
                "name": name,
                "run": run_name,
                "task": str(meta["task"]),
                "t": grid,
                "action": resampled[:, :len(ACTION_COLS)],
                "state": resampled[:, len(ACTION_COLS):],
                "synthetic": bool(meta.get("synthetic", False)),
                "joint_dt_median_s": float(np.median(gaps)),
                "joint_gap_max_s": float(gaps.max()),
                "joint_dt_error_max_s": float(np.abs(gaps - 1.0 / FPS).max()),
            })
    return episodes


def write_fsl(path_values, width):
    arr = np.ascontiguousarray(path_values, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(arr.reshape(-1), type=pa.float32()), width)


def to_json_stats(stats):
    return {k: {kk: np.asarray(vv).tolist() for kk, vv in v.items()} for k, v in stats.items()}


def main():
    ap = argparse.ArgumentParser(description="Build a LeRobot v3.0 dataset from finished sort episodes.")
    ap.add_argument("--runs", nargs="+", required=True, help="run folders, e.g. logs/sort/RUN")
    ap.add_argument("--out", required=True, help="dataset root to (re)build, inside the project folder")
    ap.add_argument("--variant", choices=["team", "full"], required=True)
    ap.add_argument("--only-success", action="store_true")
    ap.add_argument("--zip", action="store_true", help="also write OUT.zip for Studio import (mp4 stored)")
    args = ap.parse_args()
    t_begin = time.time()
    out = os.path.abspath(args.out)
    if not out.startswith(ROOT + "/") or "/data/studio-datasets" in out or "/data/curated" in out:
        sys.exit("refusing --out %s: must be inside %s and not a Studio or curated folder" % (out, ROOT))
    cams = ["close", "top"] + (["wrist"] if args.variant == "full" else [])
    tpl = os.path.join(ROOT, TEMPLATE)
    with open(os.path.join(tpl, "meta/info.json"), encoding="utf-8") as fh:
        tpl_info = json.load(fh)
    hf_meta = (pq.read_schema(os.path.join(tpl, "data/chunk-000/file-000.parquet")).metadata or {}).get(b"huggingface")

    skipped = []
    episodes = load_episodes(args.runs, args.only_success, skipped)
    logs_dir = os.path.join(ROOT, "logs/monitor")
    wrist_dir = os.path.join(ROOT, "data/wrist/raw")
    media = {"top": list_media(logs_dir, "top"), "close": list_media(logs_dir, "close"), "wrist": list_media(wrist_dir, "w")}
    span = {"top": None, "close": None, "wrist": WRIST_SEG_S}
    ok = []
    for ep in episodes:
        t_first, t_last = float(ep["t"][0]), float(ep["t"][-1])
        files, reason = {}, None
        for cam in cams:
            cand = candidates(media[cam], t_first - PAD, t_last + PAD, span[cam])
            if not cand or cand[0][1] > t_first:
                reason = "no %s video covers the episode start" % cam
                break
            if not has_packet_at_or_after(cand[-1][0], cand[-1][1], t_last - 1.0 / FPS):
                reason = "%s video does not reach the episode end (still recording or gap)" % cam
                break
            files[cam] = cand
        if reason:
            skipped.append({"episode": ep["name"], "run": ep["run"], "reason": reason})
        else:
            ep["files"] = files
            ok.append(ep)
    if not ok:
        sys.exit("no usable episodes; skipped: %s" % json.dumps(skipped))

    tasks, row0 = [], 0
    for k, ep in enumerate(ok):
        ep["index"] = k
        ep["row0"] = row0
        row0 += len(ep["t"])
        if ep["task"] not in tasks:
            tasks.append(ep["task"])
    total, n_eps = row0, len(ok)
    if os.path.exists(out):
        shutil.rmtree(out)
    cam_list = [c for c in ["close", "top", "wrist"] if c in cams]
    feature_keys = FEATURE_ORDER + [CAM_KEY[c] for c in cam_list]
    height, width = int(tpl_info["features"][CAM_KEY["close"]]["shape"][0]), int(tpl_info["features"][CAM_KEY["close"]]["shape"][1])
    options = VideoEncoderConfig().get_codec_options(None, as_strings=True)
    jobs = []
    for cam in cam_list:
        mp4 = os.path.join(out, "videos", CAM_KEY[cam], "chunk-000", "file-000.mp4")
        eps_job = [{"name": ep["name"], "times": ep["t"], "row0": ep["row0"], "files": ep["files"][cam]} for ep in ok]
        jobs.append((cam, mp4, width, height, options, eps_job))

    t_enc = time.time()
    try:
        with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
            enc = {r["cam"]: r for r in pool.map(encode_camera, jobs)}
    except Exception:
        shutil.rmtree(out, ignore_errors=True)
        raise
    t_enc = time.time() - t_enc
    img_stats = {cam: {r["name"]: r for r in enc[cam]["episodes"]} for cam in cam_list}

    for sub in ("data/chunk-000", "meta/episodes/chunk-000"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    frame_index = np.concatenate([np.arange(len(ep["t"]), dtype=np.int64) for ep in ok])
    table = pa.table({
        "action": write_fsl(np.concatenate([ep["action"] for ep in ok]), 6),
        "observation.state": write_fsl(np.concatenate([ep["state"] for ep in ok]), 6),
        "timestamp": pa.array((frame_index.astype(np.float64) / FPS).astype(np.float32), type=pa.float32()),
        "frame_index": pa.array(frame_index),
        "episode_index": pa.array(np.concatenate([np.full(len(ep["t"]), ep["index"], dtype=np.int64) for ep in ok])),
        "index": pa.array(np.arange(total, dtype=np.int64)),
        "task_index": pa.array(np.concatenate([np.full(len(ep["t"]), tasks.index(ep["task"]), dtype=np.int64) for ep in ok])),
    })
    if hf_meta:
        table = table.replace_schema_metadata({b"huggingface": hf_meta})
    pq.write_table(table, os.path.join(out, "data/chunk-000/file-000.parquet"))
    pd.DataFrame({"task_index": np.arange(len(tasks), dtype=np.int64)},
                 index=pd.Index(tasks, name="task")).to_parquet(os.path.join(out, "meta/tasks.parquet"))

    ep_stats = []
    for ep in ok:
        n = len(ep["t"])
        feats = {
            "action": ep["action"],
            "observation.state": ep["state"],
            "timestamp": (np.arange(n, dtype=np.float64) / FPS).astype(np.float32),
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, ep["index"], dtype=np.int64),
            "index": np.arange(ep["row0"], ep["row0"] + n, dtype=np.int64),
            "task_index": np.full(n, tasks.index(ep["task"]), dtype=np.int64),
        }
        stats = {k: get_feature_stats(v, axis=0, keepdims=(v.ndim == 1)) for k, v in feats.items()}
        for cam in cam_list:
            stats[CAM_KEY[cam]] = img_stats[cam][ep["name"]]["stats"]
        ep["stats"] = stats
        ep_stats.append(stats)
    dataset_stats = aggregate_stats(ep_stats)
    with open(os.path.join(out, "meta/stats.json"), "w", encoding="utf-8") as fh:
        json.dump(to_json_stats(dataset_stats), fh, indent=4, sort_keys=True)

    names, types, cols = [], [], []

    def add(name, typ, values):
        names.append(name)
        types.append(typ)
        cols.append(values)

    add("episode_index", pa.int64(), [ep["index"] for ep in ok])
    add("tasks", pa.list_(pa.string()), [[ep["task"]] for ep in ok])
    add("length", pa.int64(), [len(ep["t"]) for ep in ok])
    add("data/chunk_index", pa.int64(), [0] * n_eps)
    add("data/file_index", pa.int64(), [0] * n_eps)
    add("dataset_from_index", pa.int64(), [ep["row0"] for ep in ok])
    add("dataset_to_index", pa.int64(), [ep["row0"] + len(ep["t"]) for ep in ok])
    for cam in cam_list:
        key = CAM_KEY[cam]
        add("videos/%s/chunk_index" % key, pa.int64(), [0] * n_eps)
        add("videos/%s/file_index" % key, pa.int64(), [0] * n_eps)
        add("videos/%s/from_timestamp" % key, pa.float64(), [ep["row0"] / FPS for ep in ok])
        add("videos/%s/to_timestamp" % key, pa.float64(), [(ep["row0"] + len(ep["t"])) / FPS for ep in ok])
    video_keys = {CAM_KEY[c] for c in cam_list}
    for fkey in feature_keys:
        for stat in STAT_ORDER:
            if stat == "count":
                typ = pa.list_(pa.int64())
            elif fkey in video_keys:
                typ = pa.list_(pa.list_(pa.list_(pa.float64())))
            else:
                typ = pa.list_(pa.float64())
            add("stats/%s/%s" % (fkey, stat), typ, [np.asarray(ep["stats"][fkey][stat]).tolist() for ep in ok])
    add("meta/episodes/chunk_index", pa.int64(), [0] * n_eps)
    add("meta/episodes/file_index", pa.int64(), [0] * n_eps)
    ep_table = pa.Table.from_arrays([pa.array(c, type=t) for c, t in zip(cols, types)], names=names)
    pq.write_table(ep_table, os.path.join(out, "meta/episodes/chunk-000/file-000.parquet"))

    features = {}
    for key, val in tpl_info["features"].items():
        if key.startswith("observation.images."):
            if not any(k.startswith("observation.images.") for k in features):
                for cam in cam_list:
                    features[CAM_KEY[cam]] = copy.deepcopy(tpl_info["features"][CAM_KEY["close"]])
            continue
        features[key] = copy.deepcopy(val)
    info = {
        "codebase_version": tpl_info["codebase_version"],
        "fps": FPS,
        "features": features,
        "total_episodes": n_eps,
        "total_frames": total,
        "total_tasks": len(tasks),
        "chunks_size": tpl_info["chunks_size"],
        "data_files_size_in_mb": tpl_info["data_files_size_in_mb"],
        "video_files_size_in_mb": tpl_info["video_files_size_in_mb"],
        "data_path": tpl_info["data_path"],
        "video_path": tpl_info["video_path"],
        "robot_type": tpl_info["robot_type"],
        "splits": {"train": "0:%d" % n_eps},
    }
    with open(os.path.join(out, "meta/info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=4)

    per_ep = []
    for ep in ok:
        per_ep.append({
            "episode": ep["name"], "run": ep["run"], "task": ep["task"], "rows": len(ep["t"]),
            "synthetic": ep["synthetic"],
            "joint_dt_median_s": round(ep["joint_dt_median_s"], 6),
            "joint_gap_max_s": round(ep["joint_gap_max_s"], 6),
            "joint_dt_error_max_s": round(ep["joint_dt_error_max_s"], 6),
            "video_nearest_frame_dt_max_s": {c: round(img_stats[c][ep["name"]]["dt_max_s"], 6) for c in cam_list},
            "video_nearest_frame_dt_mean_s": {c: round(img_stats[c][ep["name"]]["dt_mean_s"], 6) for c in cam_list},
            "video_rows_dt_over_1_frame": {c: img_stats[c][ep["name"]]["rows_dt_over_1_frame"] for c in cam_list},
        })
    report = {
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "variant": args.variant,
        "only_success": bool(args.only_success),
        "cams": {c: CAM_KEY[c] for c in cam_list},
        "fps": FPS,
        "episodes_built": n_eps,
        "frames_total": total,
        "tasks": tasks,
        "synthetic_episodes_built": sum(1 for ep in ok if ep["synthetic"]),
        "skipped": skipped,
        "max_nearest_frame_dt_s": {c: max(img_stats[c][ep["name"]]["dt_max_s"] for ep in ok) for c in cam_list},
        "seconds_encode": round(t_enc, 2),
        "seconds_total": round(time.time() - t_begin, 2),
        "stats_note": "per-episode image stats follow lerobot compute_episode_stats (sample_indices, /255); "
                      "frames downsampled to 160x120 with swscale AREA (lerobot uses [::4,::4])",
        "per_episode": per_ep,
    }
    report_path = os.path.join(os.path.dirname(out), os.path.basename(out) + ".build_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    if args.zip:
        zip_path = out + ".zip"
        if os.path.exists(zip_path):
            os.remove(zip_path)
        with zipfile.ZipFile(zip_path, "w") as zf:
            for dirpath, _, files_ in os.walk(out):
                for fn in sorted(files_):
                    full = os.path.join(dirpath, fn)
                    comp = zipfile.ZIP_STORED if fn.endswith(".mp4") else zipfile.ZIP_DEFLATED
                    zf.write(full, os.path.relpath(full, out), compress_type=comp)
    print("built %s: episodes=%d frames=%d tasks=%d variant=%s synthetic=%d skipped=%d" % (
        out, n_eps, total, len(tasks), args.variant, report["synthetic_episodes_built"], len(skipped)))
    print("max nearest-frame dt (s): %s" % json.dumps(report["max_nearest_frame_dt_s"]))
    print("encode %.1f s, total %.1f s; report %s" % (t_enc, report["seconds_total"], report_path))


if __name__ == "__main__":
    main()
