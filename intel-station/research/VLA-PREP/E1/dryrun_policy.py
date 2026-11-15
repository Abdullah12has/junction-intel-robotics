#!/usr/bin/env python3
"""E1 (VLA-PREP) offline dry run: Studio ACT export 389da0fd through physicalai InferenceModel on OpenVINO.

Offline only. Touches no robot, serial port, camera device, Studio API or Studio data folder.
Reads: the export copy (manifest.json, OpenVINO IR), the recorded monitor MKVs top and close (absolute epoch pts)
and one sort episode (joints.csv). Writes one result file (--out). Progress lines go to stdout.

Run once, from the laptop repo root:
  nice -n 19 env OMP_NUM_THREADS=4 timeout 1500 envs/anomalib-env/bin/python3 -B -I work/VLA-PREP/E1/dryrun_policy.py
"""
import argparse
import csv
import gc
import hashlib
import json
import math
import subprocess
import sys
import time

import numpy as np

R = "/home/intel/intel-junction-2026"
sys.path.insert(0, f"{R}/repos/physicalai/src")
from physicalai.inference import InferenceModel  # noqa: E402
import openvino as ov  # noqa: E402

MODEL_ID = "389da0fd-e840-4224-b0a0-09aa5be7a8fc"
EXPORT = f"{R}/work/P2-B-OVRUN/E1/models/389da0fd/openvino"
TASK = "move the brick to the inspection spot"  # task_index 0 of dataset a4ba6ede, the only task of that set
EPISODE = f"{R}/logs/sort/any_1740/ep_0015"
MKV = {"top": f"{R}/logs/monitor/top_20261114T170351Z.mkv",
       "close": f"{R}/logs/monitor/close_20261114T170351Z.mkv"}
# Studio camera key -> monitor view. Role map from app/anomaly/spot_samples.py (close=243722064316, top=243622061421).
VIEW_FOR_KEY = {"images.realsense 243722064316": "close", "images.realsense 243622061421": "top"}
H, W = 480, 640
FPS = 30.0
N_QUERIES = 30
HORIZONS = [0, 10, 25, 50, 99]
GOAL_TOL_S = 0.1
DEVICES = ["GPU", "NPU", "CPU"]
LO = np.array([-105.0] * 5 + [-5.0])
HI = np.array([105.0] * 5 + [105.0])


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean(x):
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    return x


def write(res, out):
    with open(out, "w") as fh:
        json.dump(clean(res), fh, indent=1)


def load_joints(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    t = np.array([float(r["t"]) for r in rows])
    goal = np.array([[float(r[f"goal{j}"]) for j in range(6)] for r in rows])
    pos = np.array([[float(r[f"pos{j}"]) for j in range(6)] for r in rows])
    return t, goal, pos


def probe_start(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=start_time", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def grab(path, start, t_abs):
    rel = max(0.0, t_abs - start)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{rel:.4f}", "-i", path, "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    if len(raw) != H * W * 3:
        raise RuntimeError(f"{path} at rel {rel:.3f} s: {len(raw)} bytes, expected {H * W * 3}")
    return np.frombuffer(raw, np.uint8).reshape(H, W, 3).copy(), round(rel, 4)


def goal_index(t, target):
    j = int(np.searchsorted(t, target))
    cands = [k for k in (j - 1, j) if 0 <= k < len(t)]
    if not cands:
        return None
    k = min(cands, key=lambda c: abs(t[c] - target))
    return k if abs(t[k] - target) <= GOAL_TOL_S else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{R}/work/VLA-PREP/E1/dryrun.json")
    args = ap.parse_args()
    core = ov.Core()
    res = {"script": __file__, "script_sha256": sha256_file(__file__), "started_utc": utc(),
           "python": sys.version.split()[0], "openvino": ov.__version__, "available_devices": core.available_devices,
           "model_id": MODEL_ID, "export": EXPORT, "manifest_sha256": sha256_file(f"{EXPORT}/manifest.json"),
           "task_text_used": TASK,
           "task_note": "dataset a4ba6ede has one task (index 0); the export is the S1 ACT curated v0 model trained on that set",
           "notes": [
               "sane = all values finite, joints 0-4 within [-105, 105], gripper within [-5, 105], chunk shape (100, 6), "
               "and the mean policy error at horizon 0 is below the mean hold-state error at horizon 0",
               "hold-state baseline predicts the present pose (pos0..5) for every horizon step",
               "horizon k compares chunk[k] with the recorded goal (goal0..5) at t + k/30 s, nearest row within 0.1 s",
               "frames: ffmpeg input seek to t minus MKV start_time, one frame per view per query; accurate to about one frame (33 ms), not checked per frame",
               "state is the recorded present pose (pos0..5); the recorded goals (goal0..5) are the sort loop's commanded positions; units are normalized",
           ]}

    manifest = json.load(open(f"{EXPORT}/manifest.json"))
    feats = manifest["model"]["input_features"]
    visual = sorted(f["init_args"]["name"] for f in feats if f["init_args"]["ftype"] == "VISUAL")
    states = [f["init_args"]["name"] for f in feats if f["init_args"]["ftype"] == "STATE"]
    if visual != sorted(VIEW_FOR_KEY) or states != ["state"]:
        raise SystemExit(f"manifest inputs differ from the camera map: {visual} {states}")
    res["image_keys"] = visual

    t, goal, pos = load_joints(f"{EPISODE}/joints.csv")
    gaps = np.diff(t)
    res["episode"] = {"dir": EPISODE, "rows": int(len(t)), "t_first": float(t[0]), "t_last": float(t[-1]),
                      "span_s": round(float(t[-1] - t[0]), 3), "median_gap_s": round(float(np.median(gaps)), 4),
                      "max_gap_s": round(float(gaps.max()), 4)}
    qrows = sorted(set(np.linspace(0, len(t) - 1, N_QUERIES).round().astype(int).tolist()))
    res["queries"] = {"n": len(qrows), "rows": qrows, "t_rel_s": [round(float(t[i] - t[0]), 3) for i in qrows]}

    frames, cams = {}, {}
    for view, path in MKV.items():
        start = probe_start(path)
        frames[view], offs = [], []
        for i in qrows:
            img, rel = grab(path, start, float(t[i]))
            frames[view].append(img)
            offs.append(rel)
        cams[view] = {"mkv": path, "start_time": start, "seek_rel_s": offs,
                      "unique_frames": len({hashlib.sha256(f.tobytes()).hexdigest() for f in frames[view]})}
    res["cameras"] = cams

    key_of = {view: key for key, view in VIEW_FOR_KEY.items()}
    obs_list = []
    for n, i in enumerate(qrows):
        obs = {"state": np.asarray([pos[i]], dtype=np.float32)}
        for view in ("top", "close"):
            obs[key_of[view]] = frames[view][n][np.newaxis]
        obs["task"] = [TASK]
        obs_list.append(obs)
    write(res, args.out)

    res["devices"], stacks = {}, {}
    for dev in DEVICES:
        r = {"ok": False, "error": None}
        res["devices"][dev] = r
        model = None
        try:
            t0 = time.perf_counter()
            model = InferenceModel(EXPORT, backend="openvino", device=dev)
            r["load_s"] = round(time.perf_counter() - t0, 3)
            lat, chunks = [], []
            for obs in obs_list:
                t1 = time.perf_counter()
                out = model.predict_action_chunk(obs)
                lat.append(time.perf_counter() - t1)
                arr = np.asarray(out, dtype=np.float64)
                chunks.append(arr[0] if arr.ndim == 3 else arr)
            shapes = sorted({c.shape for c in chunks})
            r["chunk_shapes"] = [list(s) for s in shapes]
            if len(shapes) != 1:
                raise RuntimeError(f"chunk shapes differ: {shapes}")
            stack = np.stack(chunks)
            ms = np.asarray(lat) * 1000.0
            r.update({"first_infer_ms": round(float(ms[0]), 2),
                      "p50_ms": round(float(np.percentile(ms[1:], 50)), 2),
                      "p95_ms": round(float(np.percentile(ms[1:], 95)), 2),
                      "max_ms": round(float(ms[1:].max()), 2),
                      "latency_ms_per_query": [round(float(x), 2) for x in ms]})
            r["all_finite"] = bool(np.isfinite(stack).all())
            r["in_range"] = bool(((stack >= LO) & (stack <= HI)).all())
            pol = {k: [] for k in HORIZONS}
            hold = {k: [] for k in HORIZONS}
            for n, i in enumerate(qrows):
                for k in HORIZONS:
                    j = goal_index(t, t[i] + k / FPS)
                    if j is None:
                        continue
                    pol[k].append(np.abs(stack[n, k] - goal[j]))
                    hold[k].append(np.abs(pos[i] - goal[j]))
            used = [k for k in HORIZONS if pol[k]]
            r["pairs_by_k"] = {str(k): len(pol[k]) for k in used}
            r["policy_mae_per_joint_by_k"] = {str(k): np.mean(pol[k], axis=0).tolist() for k in used}
            r["hold_mae_per_joint_by_k"] = {str(k): np.mean(hold[k], axis=0).tolist() for k in used}
            r["policy_mae_mean_by_k"] = {str(k): float(np.mean(pol[k])) for k in used}
            r["hold_mae_mean_by_k"] = {str(k): float(np.mean(hold[k])) for k in used}
            r["beats_hold_by_k"] = {str(k): bool(np.mean(pol[k]) < np.mean(hold[k])) for k in used}
            r["beats_hold_k0"] = bool(r["beats_hold_by_k"].get("0", False))
            r["sane"] = bool(r["all_finite"] and r["in_range"] and r["beats_hold_k0"] and stack.shape[1:] == (100, 6))
            stacks[dev] = stack
            r["ok"] = True
        except Exception as exc:  # keep the error and go on with the next device
            r["error"] = f"{type(exc).__name__}: {str(exc)[:600]}"
        finally:
            if model is not None:
                model.close()
            del model
            gc.collect()
        print(utc(), dev, "ok" if r["ok"] else "FAILED", r["error"] or "", flush=True)
        write(res, args.out)

    ref = next((d for d in ("CPU", "GPU", "NPU") if d in stacks), None)
    res["parity_reference"] = ref
    res["parity"] = {}
    for d, st in stacks.items():
        if ref is None or d == ref:
            continue
        diff = np.abs(st - stacks[ref])
        res["parity"][f"{d}_vs_{ref}"] = {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
                                          "first_step_mean_abs": float(diff[:, 0, :].mean())}
    res["finished_utc"] = utc()
    write(res, args.out)
    print(utc(), "WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
