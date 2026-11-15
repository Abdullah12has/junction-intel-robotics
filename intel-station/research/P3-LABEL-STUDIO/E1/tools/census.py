#!/usr/bin/env python3
"""Census of R/data/studio-datasets (LeRobot v3 layout). Read-only on the dataset folders.

Writes out/datasets_census.json and out/episodes_census.jsonl; prints a compact table.
"""
import glob
import hashlib
import json
import os
import statistics
import subprocess

import pyarrow.parquet as pq

R = "/home/intel/intel-junction-2026"
SD = os.path.join(R, "data", "studio-datasets")
OUTD = os.path.join(R, "work", "P3-LABEL-STUDIO", "E1", "out")
os.makedirs(OUTD, exist_ok=True)
LAGS = range(-8, 9)


def ffprobe(path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
           "-show_entries",
           "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_read_packets,start_time,duration:"
           "format=duration,start_time,size",
           "-of", "json", path]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return {"error": p.stderr.strip()[:300]}
    j = json.loads(p.stdout or "{}")
    s = (j.get("streams") or [{}])[0]
    f = j.get("format") or {}
    return {"codec": s.get("codec_name"), "w": s.get("width"), "h": s.get("height"),
            "pix_fmt": s.get("pix_fmt"), "r_frame_rate": s.get("r_frame_rate"),
            "packets": s.get("nb_read_packets"), "stream_start": s.get("start_time"),
            "stream_duration": s.get("duration"), "format_duration": f.get("duration"),
            "format_start": f.get("start_time")}


def read_pq(path):
    t = pq.read_table(path)
    return t.column_names, t.to_pylist()


def mae(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def lag_scan(act, st):
    # state[t+lag] compared with action[t]; a positive best lag means the state trails the command
    n = len(act)
    res = {}
    for lag in LAGS:
        errs = [mae(act[t], st[t + lag]) for t in range(n) if 0 <= t + lag < n]
        if errs:
            res[lag] = sum(errs) / len(errs)
    best = min(res, key=res.get)
    return best, res[best], res.get(0)


def grip_events(g):
    lo, hi = min(g), max(g)
    rng = hi - lo
    if rng < 5:
        return [], rng
    mid, band = (lo + hi) / 2.0, 0.1 * rng
    state = "hi" if g[0] > mid else "lo"
    ev = []
    for i, v in enumerate(g):
        if state == "hi" and v < mid - band:
            state = "lo"
            ev.append([i, "to_lo"])
        elif state == "lo" and v > mid + band:
            state = "hi"
            ev.append([i, "to_hi"])
    return ev, rng


def col_range(arr, j):
    col = [row[j] for row in arr]
    return [round(min(col), 2), round(max(col), 2)]


def process(uuid):
    d = os.path.join(SD, uuid)
    info = json.load(open(os.path.join(d, "meta", "info.json")))
    feats, fps = info["features"], info["fps"]
    jn = feats["action"]["names"]
    cams = [k for k, v in feats.items() if v.get("dtype") == "video"]
    cam_meta = {k: {kk: feats[k]["info"].get(kk) for kk in
                    ("video.codec", "video.height", "video.width", "video.fps", "is_depth_map")}
                for k in cams}
    tcols, tasks = read_pq(os.path.join(d, "meta", "tasks.parquet"))
    meta_rows, meta_cols = [], None
    for f in sorted(glob.glob(os.path.join(d, "meta", "episodes", "chunk-*", "*.parquet"))):
        cols, rows = read_pq(f)
        if meta_cols is None:
            meta_cols = [c for c in cols if not c.startswith("stats/")]
        for r in rows:
            r = {k: v for k, v in r.items() if not k.startswith("stats/")}
            r["_meta_file"] = os.path.relpath(f, d)
            meta_rows.append(r)
    data_by_ep, data_files, dcols, total_rows = {}, [], None, 0
    for f in sorted(glob.glob(os.path.join(d, "data", "chunk-*", "*.parquet"))):
        cols, rows = read_pq(f)
        if dcols is None:
            dcols = cols
        data_files.append({"file": os.path.relpath(f, d), "rows": len(rows)})
        total_rows += len(rows)
        for r in rows:
            data_by_ep.setdefault(int(r["episode_index"]), []).append(r)
    videos = {}
    for k in cams:
        files = sorted(glob.glob(os.path.join(d, "videos", k, "chunk-*", "*.mp4")))
        videos[k] = [dict(file=os.path.relpath(f, d), size=os.path.getsize(f), **ffprobe(f))
                     for f in files]
    eps, issues = [], []
    for m in sorted(meta_rows, key=lambda r: int(r["episode_index"])):
        ep = int(m["episode_index"])
        rows = sorted(data_by_ep.get(ep, []), key=lambda r: int(r["frame_index"]))
        n = len(rows)
        rec = {"dataset": uuid, "episode_index": ep, "meta_length": m.get("length"),
               "data_rows": n, "meta_file": m["_meta_file"], "tasks": m.get("tasks"),
               "task_index_data": sorted({int(r["task_index"]) for r in rows}), "flags": []}
        seg = {}
        for k in cams:
            c = m.get("videos/%s/chunk_index" % k)
            fidx = m.get("videos/%s/file_index" % k)
            f0 = m.get("videos/%s/from_timestamp" % k)
            f1 = m.get("videos/%s/to_timestamp" % k)
            if c is None:
                seg[k] = None
                rec["flags"].append("no_video_ref_%s" % k[-4:])
                continue
            mp4 = os.path.join(d, "videos", k, "chunk-%03d" % int(c), "file-%03d.mp4" % int(fidx))
            ex = os.path.exists(mp4)
            span = round(f1 - f0, 4) if (f0 is not None and f1 is not None) else None
            seg[k] = {"mp4": os.path.relpath(mp4, d), "exists": ex, "from_ts": f0,
                      "to_ts": f1, "span_s": span}
            if not ex:
                rec["flags"].append("mp4_missing_%s" % k[-4:])
            elif span is not None and n and abs(span - n / fps) > 0.1:
                rec["flags"].append("video_span_vs_rows_%s=%.3f_vs_%.3f" % (k[-4:], span, n / fps))
        rec["video_segments"] = seg
        if n == 0:
            rec["flags"].append("no_data_rows")
            eps.append(rec)
            continue
        fi = [int(r["frame_index"]) for r in rows]
        ts = [float(r["timestamp"]) for r in rows]
        act = [[float(x) for x in r["action"]] for r in rows]
        st = [[float(x) for x in r["observation.state"]] for r in rows]
        if fi != list(range(n)):
            rec["flags"].append("frame_index_not_contiguous")
        if m.get("length") is not None and int(m["length"]) != n:
            rec["flags"].append("meta_length_ne_rows")
        dts = [ts[i + 1] - ts[i] for i in range(n - 1)]
        bad = sum(1 for x in dts if abs(x - 1.0 / fps) > 0.5 / fps)
        if bad:
            rec["flags"].append("ts_bad_steps=%d" % bad)
        nan = sum(1 for row in act + st for x in row if x != x)
        if nan:
            rec["flags"].append("nan=%d" % nan)
        astd = [round(statistics.pstdev([row[j] for row in act]), 2) for j in range(len(jn))]
        frozen = [jn[j] for j in range(len(jn)) if astd[j] < 0.5]
        if frozen:
            rec["flags"].append("frozen_action=" + "|".join(frozen))
        best, e_best, e0 = lag_scan(act, st)
        ev, grng = grip_events([row[5] for row in act])
        arng = [col_range(act, j) for j in range(len(jn))]
        rec.update({
            "n_frames": n, "dur_s": round(n / fps, 3), "t0": round(ts[0], 4),
            "t_last": round(ts[-1], 4), "lag_best": best, "mae_lag_best": round(e_best, 3),
            "mae_lag0": round(e0, 3) if e0 is not None else None,
            "action_range": arng, "grip_range": arng[5],
            "state_range": [col_range(st, j) for j in range(len(jn))],
            "action_std": astd,
            "action_first": [round(x, 2) for x in act[0]],
            "action_last": [round(x, 2) for x in act[-1]],
            "state_first": [round(x, 2) for x in st[0]],
            "state_last": [round(x, 2) for x in st[-1]],
            "grip_events": ev, "grip_events_s": [[round(i / fps, 2), s] for i, s in ev],
            "sig": hashlib.sha1(json.dumps([[round(x, 1) for x in row] for row in act])
                                .encode()).hexdigest()[:16]})
        eps.append(rec)
    sigs = {}
    for e in eps:
        if "sig" in e:
            sigs.setdefault(e["sig"], []).append(e["episode_index"])
    for e in eps:
        if "sig" in e and len(sigs[e["sig"]]) > 1:
            e["flags"].append("dup_action_with_eps=%s" % sigs[e["sig"]])
    ok = [e for e in eps if "action_range" in e]
    jmm_a = [[min(e["action_range"][j][0] for e in ok), max(e["action_range"][j][1] for e in ok)]
             for j in range(len(jn))] if ok else None
    jmm_s = [[min(e["state_range"][j][0] for e in ok), max(e["state_range"][j][1] for e in ok)]
             for j in range(len(jn))] if ok else None
    if info.get("total_episodes") != len(meta_rows):
        issues.append("info.total_episodes=%s vs meta rows=%d" % (info.get("total_episodes"), len(meta_rows)))
    if info.get("total_frames") != total_rows:
        issues.append("info.total_frames=%s vs data rows=%d" % (info.get("total_frames"), total_rows))
    if len(data_by_ep) != len(meta_rows):
        issues.append("episodes with data=%d vs meta=%d" % (len(data_by_ep), len(meta_rows)))
    if info.get("total_tasks") != len(tasks):
        issues.append("info.total_tasks=%s vs tasks rows=%d" % (info.get("total_tasks"), len(tasks)))
    issues.append("episodes with flags=%d of %d" % (sum(1 for e in eps if e["flags"]), len(eps)))
    summary = {
        "uuid": uuid, "path": "data/studio-datasets/" + uuid,
        "codebase_version": info.get("codebase_version"), "fps": fps,
        "robot_type": info.get("robot_type"), "total_episodes_info": info.get("total_episodes"),
        "total_frames_info": info.get("total_frames"), "total_tasks_info": info.get("total_tasks"),
        "meta_episode_rows": len(meta_rows), "data_episode_count": len(data_by_ep),
        "data_rows": total_rows, "joint_names": jn,
        "joint_minmax_action": jmm_a, "joint_minmax_state": jmm_s,
        "features": {k: {"dtype": v.get("dtype"), "shape": v.get("shape"), "names": v.get("names")}
                     for k, v in feats.items() if v.get("dtype") != "video"},
        "cameras": cam_meta, "tasks_columns": tcols, "tasks": tasks, "data_columns": dcols,
        "episode_meta_columns": meta_cols, "data_files": data_files, "videos": videos,
        "issues": issues, "episode_indices": [e["episode_index"] for e in eps]}
    return summary, eps


def main():
    all_sum, all_eps = [], []
    for uuid in sorted(os.listdir(SD)):
        if os.path.isdir(os.path.join(SD, uuid)):
            s, e = process(uuid)
            all_sum.append(s)
            all_eps.extend(e)
    with open(os.path.join(OUTD, "datasets_census.json"), "w") as fh:
        json.dump(all_sum, fh, indent=1)
    with open(os.path.join(OUTD, "episodes_census.jsonl"), "w") as fh:
        for e in all_eps:
            fh.write(json.dumps(e) + "\n")
    for s in all_sum:
        print("DATASET %s fps=%s robot=%s info_eps=%s meta=%s data=%s rows=%s info_frames=%s "
              "tasks=%d cams=%s" % (s["uuid"][:8], s["fps"], s["robot_type"], s["total_episodes_info"],
                                    s["meta_episode_rows"], s["data_episode_count"], s["data_rows"],
                                    s["total_frames_info"], len(s["tasks"]),
                                    [c[-12:] for c in sorted(s["cameras"].keys())]))
        print("  tasks:", json.dumps(s["tasks"]))
        print("  joint_minmax_action:", json.dumps(s["joint_minmax_action"]))
        print("  joint_minmax_state:", json.dumps(s["joint_minmax_state"]))
        print("  issues:", json.dumps(s["issues"]))
    print("EPISODES (dataset ep rows dur_s | task | lag_best mae_best mae_lag0 | grip_range | "
          "grip_events_s | flags)")
    for e in all_eps:
        print("%s %3d %4s %7s | %s | %s %s %s | %s | %s | %s" % (
            e["dataset"][:8], e["episode_index"], e["data_rows"], e.get("dur_s"),
            "; ".join(e.get("tasks") or []), e.get("lag_best"), e.get("mae_lag_best"),
            e.get("mae_lag0"), e.get("grip_range"), e.get("grip_events_s"), e["flags"]))


if __name__ == "__main__":
    main()
