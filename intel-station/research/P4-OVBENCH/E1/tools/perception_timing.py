#!/usr/bin/env python3
"""E1 P4-OVBENCH: offline per-call cost of the pick-anywhere loop's perception and IK steps on recorded frames, plus
per-phase cycle timings from the real cycles.jsonl logs. Nothing moves: every grasp_auto motion, state, frame, camera
and serial entry point is replaced by a function that raises. usage: perception_timing.py OUT_DIR (envs/data-env -I -B)"""
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

R = "/home/intel/intel-junction-2026"
sys.path.insert(0, os.path.join(R, "app", "tools"))
sys.path.insert(0, os.path.join(R, "app", "anomaly", "det2_f2"))
import grasp_auto as ga  # noqa: E402


def _blocked(*_a, **_k):
    raise RuntimeError("E1 timing harness: motion, state, frame, camera or serial entry point blocked")


for _n in ("cmd", "move", "state", "fresh", "snap", "depth", "wrist_brick", "note", "jaw_tip_low", "brick_blob",
           "jaws_clear_of_brick"):
    if hasattr(ga, _n):
        setattr(ga, _n, _blocked)
import kin  # noqa: E402
import pick_any as pa  # noqa: E402
import handdet  # noqa: E402
import sort_any as SA  # noqa: E402

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
NF, REPS = 12, 3
RUNS = ("any_1740", "any_1731")
rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "steps": {}, "setup": {}, "cycles_real": {}}
with open("/proc/loadavg") as f:
    rec["loadavg_before"] = f.read().split()[:3]


def frames(pattern):
    files = []
    for run in RUNS:
        files += sorted(glob.glob(f"{R}/logs/sort/{run}/ep_*/{pattern}"))
    out = []
    for f in files[:NF]:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is not None:
            out.append((os.path.relpath(f, R), img))
    return out


def stats(xs):
    a = np.asarray(xs, float)
    return {"n": int(a.size), "p50": round(float(np.percentile(a, 50)), 4), "p90": round(float(np.percentile(a, 90)), 4),
            "mean": round(float(a.mean()), 4), "min": round(float(a.min()), 4), "max": round(float(a.max()), 4)}


def timed(fn):
    fn()
    out = []
    for _ in range(REPS):
        t0 = time.perf_counter_ns()
        fn()
        out.append((time.perf_counter_ns() - t0) / 1e6)
    return out


def step(label, note, make_timings):
    try:
        rec["steps"][label] = dict(stats(make_timings()), note=note)
    except Exception as e:  # noqa: BLE001
        rec["steps"][label] = {"error": f"{type(e).__name__}: {str(e)[:300]}", "note": note}
    print(label, rec["steps"][label], flush=True)


def per_frame(items, factory):
    xs = []
    for _, img in items:
        xs += timed(factory(img))
    return xs


wrist, top, close = frames("*_wrist.jpg"), frames("*_top.jpg"), frames("*_close.jpg")
rec["frames"] = {"wrist": [f for f, _ in wrist], "top": [f for f, _ in top], "close": [f for f, _ in close]}
rec["frame_shapes"] = {k: (list(v[0][1].shape) if v else None) for k, v in (("wrist", wrist), ("top", top), ("close", close))}
c = SA.scan_cal()
q_scan = [float(v) for v in c["q_scan"]]
rec["setup"]["q_scan"] = q_scan
try:
    SA.fit_scan_corr()
    rec["setup"]["fit_scan_corr"] = "ok"
except Exception as e:  # noqa: BLE001
    rec["setup"]["fit_scan_corr"] = f"{type(e).__name__}: {str(e)[:200]}"
tgt = np.asarray(pa.target(), float)


def f_blobs(img):
    return lambda: pa.blobs(img, min_area=pa.SERVO_MIN_AREA)


def f_find(img):
    bl = pa.blobs(img, min_area=pa.SERVO_MIN_AREA)
    return lambda: pa.find(bl, tgt, 140, None)


def f_servo(img):
    def fn():
        b = pa.find(pa.blobs(img, min_area=pa.SERVO_MIN_AREA), tgt, 140, None)
        if b is None:
            return None
        e = tgt - np.asarray(b["centre"], float)
        d = 0.65 * np.linalg.solve(pa.m_base(q_scan), e)
        return kin.ik_rel(q_scan, [float(d[0]), float(d[1]), 0.0])
    return fn


def f_cand_cold(img):
    def fn():
        SA.REACH_CACHE.clear()
        return SA.candidates(img, q_scan, c)
    return fn


def f_cand_warm(img):
    SA.candidates(img, q_scan, c)
    return lambda: SA.candidates(img, q_scan, c)


step("pa.blobs wrist (min_area SERVO_MIN_AREA)", "pick_any.blobs", lambda: per_frame(wrist, f_blobs))
step("pa.find wrist (blobs precomputed)", "pick_any.find, no colour set", lambda: per_frame(wrist, f_find))
step("servo compute per iteration (blobs+find+solve+ik_rel)", "mirrors pick_any.servo body; no move_rel",
     lambda: per_frame(wrist, f_servo))
step("kin.ik_rel one approach step (-SCAN_UP)", "one IK solve",
     lambda: timed(lambda: kin.ik_rel(q_scan, [0.0, 0.0, -pa.SCAN_UP])))
step("kin.fk at scan pose", "forward kinematics", lambda: timed(lambda: kin.fk(q_scan)))
step("candidates wrist, cold REACH_CACHE", "blobs + grasp_reachable per blob", lambda: per_frame(wrist, f_cand_cold))
step("candidates wrist, warm REACH_CACHE", "blobs + cached grasp_reachable", lambda: per_frame(wrist, f_cand_warm))
step("close_pre_verdict (want_hue None)", "DET2 close tape rule on recorded close view",
     lambda: per_frame(close, lambda img: (lambda: SA.close_pre_verdict(img, None))))
step("close_pre_verdict (want_hue 20)", "adds the hue check",
     lambda: per_frame(close, lambda img: (lambda: SA.close_pre_verdict(img, 20))))
step("handdet.is_hand top", "view top", lambda: per_frame(top, lambda img: (lambda: handdet.is_hand(img, "top", False))))
step("handdet.is_hand close", "view close",
     lambda: per_frame(close, lambda img: (lambda: handdet.is_hand(img, "close", False))))
step("handdet.is_hand wrist", "view wrist",
     lambda: per_frame(wrist, lambda img: (lambda: handdet.is_hand(img, "wrist", False))))
try:
    ds = []
    for _, img in wrist:
        SA.REACH_CACHE.clear()
        for b in SA.candidates(img, q_scan, c)[:3]:
            ds.append(np.asarray(b["d_mm"], float))
    ds = ds[:12]
    if not ds:
        raise RuntimeError("no candidate offsets in the recorded wrist frames")
    cold = []
    for d in ds:
        for _ in range(REPS + 1):
            SA.REACH_CACHE.clear()
            t0 = time.perf_counter_ns()
            SA.grasp_reachable(q_scan, d)
            cold.append((time.perf_counter_ns() - t0) / 1e6)
    rec["steps"]["grasp_reachable cold"] = dict(stats(cold), note=f"{len(ds)} offsets from recorded frames, cache cleared per call")
    SA.REACH_CACHE.clear()
    for d in ds:
        SA.grasp_reachable(q_scan, d)
    warm = []
    for d in ds:
        for _ in range(REPS + 1):
            t0 = time.perf_counter_ns()
            SA.grasp_reachable(q_scan, d)
            warm.append((time.perf_counter_ns() - t0) / 1e6)
    rec["steps"]["grasp_reachable cached"] = dict(stats(warm), note="cache hit on the same offsets")
except Exception as e:  # noqa: BLE001
    rec["steps"]["grasp_reachable"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
print("grasp_reachable done", flush=True)
cyc = {}
for run in RUNS:
    p = f"{R}/logs/sort/{run}/cycles.jsonl"
    if not os.path.exists(p):
        continue
    for line in open(p):
        r = json.loads(line)
        if r.get("timings"):
            for k, v in r["timings"].items():
                cyc.setdefault(k, []).append(float(v))
            cyc.setdefault("success", []).append(1.0 if r.get("success") else 0.0)
for k, v in cyc.items():
    if k != "success":
        rec["cycles_real"][k] = dict(stats(v), unit="s")
rec["cycles_real"]["n_cycles_with_timings"] = len(cyc.get("cycle_s", []))
rec["cycles_real"]["success_rate"] = round(float(np.mean(cyc["success"])), 3) if cyc.get("success") else None
with open(os.path.join(OUT, "perception_timing.tsv"), "w") as f:
    f.write("step\tn\tp50_ms\tp90_ms\tmean_ms\tmin_ms\tmax_ms\tnote\n")
    for k, v in rec["steps"].items():
        if "p50" in v:
            f.write(f"{k}\t{v['n']}\t{v['p50']}\t{v['p90']}\t{v['mean']}\t{v['min']}\t{v['max']}\t{v.get('note', '')}\n")
        else:
            f.write(f"{k}\t0\t\t\t\t\t\t{v.get('error', v)}\n")
with open(os.path.join(OUT, "cycle_phases.tsv"), "w") as f:
    f.write("phase\tn\tp50_s\tp90_s\tmean_s\tmin_s\tmax_s\n")
    for k, v in rec["cycles_real"].items():
        if isinstance(v, dict) and "p50" in v:
            f.write(f"{k}\t{v['n']}\t{v['p50']}\t{v['p90']}\t{v['mean']}\t{v['min']}\t{v['max']}\n")
with open("/proc/loadavg") as f:
    rec["loadavg_after"] = f.read().split()[:3]
with open(os.path.join(OUT, "perception_timing.json"), "w") as f:
    json.dump(rec, f, indent=1, default=str)
print("written", OUT, flush=True)
