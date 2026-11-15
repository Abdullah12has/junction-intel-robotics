#!/usr/bin/env python3
"""Curate episodes of dataset 6d6bdce3 and build LeRobot v3 datasets PP-BLUE, PP-GREEN, PP-BOTH (source read-only).

Usage: python3 -I build.py <src_dataset_dir> <analysis_json> <out_dir> <episodes_v1_csv_out>
Rules (report.md section 2): one grasp; released on a plate (pan > 0 BLUE, pan < -30 GREEN); starts within
10 units of the rest pose; ends within 10 units of the rest pose; idle head/tail trimmed to 0.5 s; text = real
destination. Hand and colour notes are manual visual checks (contact sheets) and are written as notes only.
"""
import csv
import glob
import json
import os
import shutil
import sys
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FPS = 30.0
REST_TOL = 10.0
LIVE_HOME = np.array([-6.6, -98.1, 99.4, 54.3, -53.1])  # live probe 14:13Z (shared/FACTS-LIVE.md)
CAMS = ["observation.images.realsense 243622061421", "observation.images.realsense 243722064316"]
TEXT = {
    "BLUE": "move the red brick to the blue plate",
    "GREEN": "move the red brick to the green plate",
}
DATASET_DEST = {"PP-BLUE": "BLUE", "PP-GREEN": "GREEN"}
NOTES = {
    11: "top view: hand only at top-left edge over the laptop, outside the mat; red 2x2 placed on blue plate",
    12: "top view: hand only at top-left edge; arm still approaching rest at last frame (9.9 units); red 2x2 on blue plate",
    15: "top view: hand only at top-left edge; arm still approaching rest at last frame (4.7 units); red 2x2 on green plate",
}


def read_glob(pattern):
    tables = [pq.read_table(p) for p in sorted(glob.glob(pattern))]
    return pa.concat_tables(tables)


def set_col(tbl, name, values):
    i = tbl.schema.get_field_index(name)
    field = tbl.schema.field(i)
    return tbl.set_column(i, field.name, pa.array(values, type=field.type))


def feature_stats(arr, template):
    full = {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }
    for q, key in ((1, "q01"), (10, "q10"), (50, "q50"), (90, "q90"), (99, "q99")):
        full[key] = np.percentile(arr, q, axis=0).tolist()
    return {k: v for k, v in full.items() if k in template}


def build_dataset(name, items, src, out_dir, data, fi_col, meta, tasks_schema, info, stats, per_ep_rows):
    d = os.path.join(out_dir, name)
    shutil.rmtree(d, ignore_errors=True)
    for sub in ("data/chunk-000", "meta/episodes/chunk-000", "videos"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    texts = []
    for it in items:
        if TEXT[it["dest"]] not in texts:
            texts.append(TEXT[it["dest"]])
    meta_by_ep = {int(r["episode_index"]): r for r in meta.to_pylist()}
    pieces, meta_rows, video_files = [], [], set()
    gstart = 0
    for new_ep, it in enumerate(items):
        ep = it["episode"]
        idx = per_ep_rows[ep]
        fi = fi_col[idx]
        sel = idx[(fi >= it["cut_start"]) & (fi < it["cut_end"])]
        length = len(sel)
        assert length == it["cut_end"] - it["cut_start"], (ep, length)
        sub = data.take(pa.array(sel))
        frames = np.arange(length, dtype=np.int64)
        sub = set_col(sub, "episode_index", np.full(length, new_ep, dtype=np.int64))
        sub = set_col(sub, "frame_index", frames)
        sub = set_col(sub, "timestamp", (frames / FPS).astype(np.float32))
        sub = set_col(sub, "index", np.arange(gstart, gstart + length, dtype=np.int64))
        sub = set_col(sub, "task_index", np.full(length, texts.index(TEXT[it["dest"]]), dtype=np.int64))
        pieces.append(sub)
        src_row = meta_by_ep[ep]
        row = dict(src_row)
        row["episode_index"] = new_ep
        row["tasks"] = [TEXT[it["dest"]]]
        row["length"] = length
        row["data/chunk_index"] = 0
        row["data/file_index"] = 0
        row["dataset_from_index"] = gstart
        row["dataset_to_index"] = gstart + length
        for cam in CAMS:
            f0 = float(src_row[f"videos/{cam}/from_timestamp"])
            row[f"videos/{cam}/from_timestamp"] = f0 + it["cut_start"] / FPS
            row[f"videos/{cam}/to_timestamp"] = f0 + it["cut_end"] / FPS
            video_files.add((cam, int(src_row[f"videos/{cam}/chunk_index"]), int(src_row[f"videos/{cam}/file_index"])))
        meta_rows.append(row)
        gstart += length

    data_out = pa.concat_tables(pieces)
    pq.write_table(data_out, os.path.join(d, "data/chunk-000/file-000.parquet"))
    pq.write_table(pa.Table.from_pylist(meta_rows, schema=meta.schema),
                   os.path.join(d, "meta/episodes/chunk-000/file-000.parquet"))
    tasks_rows = [{"task_index": i, "task": t} for i, t in enumerate(texts)]
    pq.write_table(pa.Table.from_pylist(tasks_rows, schema=tasks_schema), os.path.join(d, "meta/tasks.parquet"))

    for cam, chunk, fidx in sorted(video_files):
        rel = f"videos/{cam}/chunk-{chunk:03d}/file-{fidx:03d}.mp4"
        dst = os.path.join(d, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(src, rel), dst)

    info2 = dict(info)
    info2["total_episodes"] = len(items)
    info2["total_frames"] = gstart
    info2["total_tasks"] = len(texts)
    info2["splits"] = {"train": f"0:{len(items)}"}
    with open(os.path.join(d, "meta/info.json"), "w", encoding="utf-8") as fh:
        json.dump(info2, fh, indent=2)

    stats2 = dict(stats)
    for key in ("action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index"):
        if key not in stats:
            continue
        arr = np.array(data_out.column(key).to_pylist(), dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        stats2[key] = feature_stats(arr, stats[key])
    with open(os.path.join(d, "meta/stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats2, fh, indent=2)

    zip_path = os.path.join(out_dir, name + ".zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for root, _, files in os.walk(d):
            for fn in sorted(files):
                full = os.path.join(root, fn)
                comp = zipfile.ZIP_STORED if fn.endswith(".mp4") else zipfile.ZIP_DEFLATED
                zf.write(full, os.path.relpath(full, d), compress_type=comp)
    return gstart, texts


def verify(d, total):
    dt = read_glob(d + "/data/chunk-*/file-*.parquet")
    mt = read_glob(d + "/meta/episodes/chunk-*/file-*.parquet")
    assert dt.num_rows == total, (dt.num_rows, total)
    ep = dt.column("episode_index").to_numpy()
    ts = dt.column("timestamp").to_numpy()
    lines = []
    for r in mt.to_pylist():
        k = int(r["episode_index"])
        n = int(np.sum(ep == k))
        assert n == int(r["length"]), (k, n, r["length"])
        assert abs(float(ts[ep == k].min())) < 1e-6
        for cam in CAMS:
            span = float(r[f"videos/{cam}/to_timestamp"]) - float(r[f"videos/{cam}/from_timestamp"])
            assert abs(span - n / FPS) < 1e-3, (k, cam, span, n / FPS)
        lines.append(f"ep{k} frames={n} secs={n / FPS:.2f} task={r['tasks'][0]!r}")
    return lines


def main():
    src, analysis_path, out_dir, csv_out = sys.argv[1:5]
    data = read_glob(src + "/data/chunk-*/file-*.parquet")
    meta = read_glob(src + "/meta/episodes/chunk-*/file-*.parquet")
    tasks_schema = pq.read_table(src + "/meta/tasks.parquet").schema
    with open(src + "/meta/info.json", encoding="utf-8") as fh:
        info = json.load(fh)
    with open(src + "/meta/stats.json", encoding="utf-8") as fh:
        stats = json.load(fh)
    with open(analysis_path, encoding="utf-8") as fh:
        analysis = {int(r["episode"]): r for r in json.load(fh)}

    ep_col = data.column("episode_index").to_numpy()
    fi_col = data.column("frame_index").to_numpy()
    states = np.array(data.column("observation.state").to_pylist(), dtype=float)
    per_ep_rows = {}
    for ep in np.unique(ep_col):
        idx = np.where(ep_col == ep)[0]
        per_ep_rows[int(ep)] = idx[np.argsort(fi_col[idx], kind="stable")]
    clean = [ep for ep, idx in per_ep_rows.items() if np.abs(states[idx[0], :5] - LIVE_HOME).max() < 10]
    home = np.median(np.stack([states[per_ep_rows[ep][0], :5] for ep in clean]), axis=0)
    print("data home", np.round(home, 2).tolist(), "clean starts", clean)

    decisions = []
    for ep in sorted(per_ep_rows):
        rec = analysis[ep]
        arm = states[per_ep_rows[ep], :5]
        start_d = float(np.abs(arm[0] - home).max())
        end_d = float(np.abs(arm[-1] - home).max())
        n_grasps = int(rec["table_n_grasps"]) if rec["table_n_grasps"] else 0
        dest = rec.get("destination_by_pan")
        reasons = []
        if n_grasps == 0:
            reasons.append("no brick grasped")
        if n_grasps >= 2:
            reasons.append("double grasp")
        if dest is None:
            reasons.append("never released on a plate")
        elif dest == "AMBIG":
            reasons.append("release not on a plate (mat put-down)")
        if start_d > REST_TOL:
            reasons.append(f"not from rest: starts {start_d:.0f} units from rest, brick already held")
        if end_d > REST_TOL:
            reasons.append(f"incomplete: ends {end_d:.1f} units from rest")
        keep = not reasons and dest in TEXT
        decisions.append({
            "episode": ep, "keep": keep, "dest": dest, "start_d": start_d, "end_d": end_d,
            "n_grasps": n_grasps, "reasons": reasons,
            "cut_start": rec["cut_start"], "cut_end": rec["cut_end"], "frames": rec["frames"],
            "pan": rec.get("pan_at_release"), "closed": rec["table_closed_val"],
            "label": rec["table_release_label"], "type": rec["table_type"],
        })

    kept = [d for d in decisions if d["keep"]]
    report = []
    for name in ("PP-BLUE", "PP-GREEN", "PP-BOTH"):
        items = [d for d in kept if name == "PP-BOTH" or d["dest"] == DATASET_DEST[name]]
        if not items:
            report.append(f"{name}: no episodes")
            continue
        total, texts = build_dataset(name, items, src, out_dir, data, fi_col, meta, tasks_schema, info, stats,
                                     per_ep_rows)
        report.append(f"{name}: episodes={len(items)} frames={total} tasks={texts}")
        report += ["  " + line for line in verify(os.path.join(out_dir, name), total)]

    rows = []
    for d in decisions:
        ds = ""
        if d["keep"]:
            ds = "PP-BOTH|" + ("PP-BLUE" if d["dest"] == "BLUE" else "PP-GREEN")
        rows.append({
            "episode": d["episode"],
            "decision": "KEEP" if d["keep"] else "DROP",
            "datasets": ds,
            "destination": d["dest"] or "",
            "text": TEXT.get(d["dest"], "") if d["keep"] else "",
            "reasons": "; ".join(d["reasons"]) if d["reasons"] else "passes all rules",
            "frames_total": d["frames"],
            "cut_start": d["cut_start"] if d["keep"] else "",
            "cut_end": d["cut_end"] if d["keep"] else "",
            "frames_kept": (d["cut_end"] - d["cut_start"]) if d["keep"] else 0,
            "start_dist_rest": round(d["start_d"], 2),
            "end_dist_rest": round(d["end_d"], 2),
            "pan_at_release": d["pan"] if d["pan"] is not None else "",
            "n_grasps": d["n_grasps"],
            "closed_val": d["closed"],
            "table_release_label": d["label"],
            "table_type": d["type"],
            "notes": NOTES.get(d["episode"], ""),
        })
    with open(csv_out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\n".join(report))
    print("wrote", csv_out, len(rows), "rows")
    for r in rows:
        print(r["episode"], r["decision"], r["destination"], r["start_dist_rest"], r["end_dist_rest"], r["reasons"])


if __name__ == "__main__":
    main()
