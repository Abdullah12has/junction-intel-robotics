#!/usr/bin/env python3
"""Check the 7 curated LeRobot sets in R/data/curated against the Studio episodes. Read-only.

For every curated set: info.json (fps, totals, joint names, features), tasks, meta episode rows, data rows.
For every curated episode: the Studio entry (dataset:episode) whose action trajectory contains it. Candidates are
found by exact lookup of rounded action rows (anchor rows of the curated episode) and ranked by the mean absolute
error over the overlap; ties list every Studio entry with the same error and overlap.
Writes out/curated_check.json and prints a summary to stdout.
"""
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq

R = "/home/intel/intel-junction-2026"
CD = os.path.join(R, "data", "curated")
SD = os.path.join(R, "data", "studio-datasets")
OUTD = os.path.join(R, "work", "P3-LABEL-STUDIO", "E1", "out")
SETS = ["MT-ALL-v2", "PP-BLUE", "PP-BLUE-v2", "PP-BOTH", "PP-GREEN", "PP-GREEN-v2", "S1_v0"]
MAXCAND = 200     # skip anchor rows that occur in more Studio frames than this (static poses)
MAXK = 60         # curated rows tried as anchors


def entry_id(ds, ep):
    return "%s:%d" % (ds[:8], ep)


def load_studio_actions():
    eps = [json.loads(line) for line in open(os.path.join(OUTD, "episodes_census.jsonl"))]
    need = {}
    for e in eps:
        need.setdefault(e["dataset"], set()).add(e["episode_index"])
    traj = {}
    for ds, epset in need.items():
        by_ep = {}
        for f in sorted(glob.glob(os.path.join(SD, ds, "data", "chunk-*", "*.parquet"))):
            t = pq.read_table(f, columns=["episode_index", "frame_index", "action"])
            ep_col = t.column("episode_index").to_pylist()
            fi_col = t.column("frame_index").to_pylist()
            act = t.column("action").to_pylist()
            for i in range(len(ep_col)):
                by_ep.setdefault(ep_col[i], []).append((fi_col[i], act[i]))
        for ep in epset:
            rows = sorted(by_ep.get(ep, []), key=lambda r: r[0])
            traj[(ds, ep)] = np.array([r[1] for r in rows], dtype=np.float64)
    return traj


def index_rows(traj):
    idx = {}
    for key, A in traj.items():
        for f in range(len(A)):
            idx.setdefault(tuple(np.round(A[f], 3)), []).append((key, f))
    return idx


def match_episode(A, idx, traj):
    evals = []
    for k in range(min(len(A), MAXK)):
        cands = idx.get(tuple(np.round(A[k], 3)), [])
        if not cands or len(cands) > MAXCAND:
            continue
        for ekey, f in cands:
            B = traj[ekey]
            shift = f - k                      # curated frame j <-> Studio frame j + shift
            a0, a1 = max(0, -shift), min(len(A), len(B) - shift)
            if a1 - a0 < 10:
                continue
            mae = float(np.abs(A[a0:a1] - B[a0 + shift:a1 + shift]).mean())
            evals.append((round(mae, 6), -(a1 - a0), ekey, shift, a0, a1))
        if evals and min(evals)[0] < 1e-6:
            break
    if not evals:
        return None
    mae, negov, ekey, shift, a0, a1 = min(evals)
    tied = sorted({entry_id(*e[2]) for e in evals if e[0] == mae and e[1] == negov})
    return {"entry": entry_id(*ekey), "tied_entries": tied, "shift": int(shift),
            "overlap_frames": int(-negov), "mae": mae,
            "covers_full_curated": bool(a0 == 0 and a1 == len(A))}


def check_set(name, traj, idx):
    base = os.path.join(CD, name)
    info = json.load(open(os.path.join(base, "meta", "info.json")))
    feats = info.get("features", {})
    tasks = sorted({r["task"] for r in pq.read_table(os.path.join(base, "meta", "tasks.parquet")).to_pylist()})
    meta = []
    for f in sorted(glob.glob(os.path.join(base, "meta", "episodes", "chunk-*", "*.parquet"))):
        meta.extend(pq.read_table(f).to_pylist())
    meta_by_ep = {int(m["episode_index"]): m for m in meta}
    data = {}
    total_rows = 0
    for f in sorted(glob.glob(os.path.join(base, "data", "chunk-*", "*.parquet"))):
        t = pq.read_table(f, columns=["episode_index", "frame_index", "action"])
        ep_col = t.column("episode_index").to_pylist()
        fi_col = t.column("frame_index").to_pylist()
        act = t.column("action").to_pylist()
        total_rows += len(ep_col)
        for i in range(len(ep_col)):
            data.setdefault(ep_col[i], []).append((fi_col[i], act[i]))
    eps_out = []
    for ep in sorted(data):
        rows = sorted(data[ep], key=lambda r: r[0])
        A = np.array([r[1] for r in rows], dtype=np.float64)
        m = meta_by_ep.get(ep, {})
        task_list = m.get("tasks") or []
        rec = {"episode_index": ep, "rows": len(A), "meta_length": m.get("length"),
               "task": task_list[0] if task_list else None,
               "frame_index_contiguous": [r[0] for r in rows] == list(range(len(A)))}
        rec["match"] = match_episode(A, idx, traj)
        eps_out.append(rec)
    summary = {
        "set": name, "fps": info.get("fps"), "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"), "total_episodes_info": info.get("total_episodes"),
        "total_frames_info": info.get("total_frames"), "total_tasks_info": info.get("total_tasks"),
        "meta_episode_rows": len(meta), "data_episodes": len(data), "data_rows": total_rows,
        "tasks": tasks, "action_names": feats.get("action", {}).get("names"),
        "state_names": feats.get("observation.state", {}).get("names"),
        "video_keys": sorted(k for k, v in feats.items() if v.get("dtype") == "video"),
        "episodes_matched": sum(1 for e in eps_out if e["match"]),
        "episodes_unmatched": [e["episode_index"] for e in eps_out if not e["match"]],
        "issues": [], "episodes": eps_out}
    if info.get("total_episodes") != len(meta) or info.get("total_episodes") != len(data):
        summary["issues"].append("episode count info=%s meta=%d data=%d" % (
            info.get("total_episodes"), len(meta), len(data)))
    if info.get("total_frames") != total_rows:
        summary["issues"].append("frames info=%s data=%d" % (info.get("total_frames"), total_rows))
    return summary


def main():
    traj = load_studio_actions()
    idx = index_rows(traj)
    out = {"sets": []}
    for name in SETS:
        s = check_set(name, traj, idx)
        out["sets"].append(s)
        print("SET %s fps=%s info_eps=%s meta=%d data_eps=%d rows=%d matched=%d unmatched=%s issues=%s" % (
            name, s["fps"], s["total_episodes_info"], s["meta_episode_rows"], s["data_episodes"],
            s["data_rows"], s["episodes_matched"], s["episodes_unmatched"], s["issues"]), flush=True)
        print("  tasks=%s" % json.dumps(s["tasks"]), flush=True)
        for e in s["episodes"]:
            mt = e["match"]
            print("  ep %3d rows %4s len %4s | %s | %s shift=%s ov=%s mae=%s full=%s tied=%s" % (
                e["episode_index"], e["rows"], e["meta_length"], e["task"],
                mt["entry"] if mt else "NO-MATCH", mt["shift"] if mt else "-",
                mt["overlap_frames"] if mt else "-", mt["mae"] if mt else "-",
                mt["covers_full_curated"] if mt else "-", mt["tied_entries"] if mt else "-"), flush=True)
    with open(os.path.join(OUTD, "curated_check.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("WROTE", os.path.join(OUTD, "curated_check.json"), flush=True)


if __name__ == "__main__":
    main()
