#!/usr/bin/env python3
"""E1 P4-OVBENCH: batch-1 OpenVINO measurement of one FP16 IR on one device (a measured run; one process per run).

usage: bench_ov.py TAG KIND XML DEVICE OUT_DIR SYNC_S ASYNC_S MAX_ITERS [GPU_HINT]
KIND anomaly: one image input [1,3,H,W], resized and ImageNet-normalised. KIND act: image inputs letterboxed to the IR
size and scaled to [0,1]; other inputs (state) are zeros. Inputs are recorded phase snapshots from logs/sort/any_1740
and logs/sort/any_1731. Writes OUT_DIR/TAG__DEVICE.json (also on error) and OUT_DIR/TAG__DEVICE__out.npz.
"""
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import openvino as ov
from openvino import AsyncInferQueue

ROOT = "/home/intel/intel-junction-2026"
TAG, KIND, XML, DEVICE, OUT_DIR = sys.argv[1:6]
SYNC_S, ASYNC_S, MAX_IT = float(sys.argv[6]), float(sys.argv[7]), int(sys.argv[8])
GPU_HINT = sys.argv[9] if len(sys.argv) > 9 else ""
NPARITY, NIREQ, WARMUP = 8, 4, 20
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
OUT_JSON = os.path.join(OUT_DIR, f"{TAG}__{DEVICE}.json")
OUT_NPZ = os.path.join(OUT_DIR, f"{TAG}__{DEVICE}__out.npz")


def loadavg():
    with open("/proc/loadavg") as f:
        return f.read().split()[:3]


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


os.makedirs(OUT_DIR, exist_ok=True)
rec = {"tag": TAG, "kind": KIND, "xml": XML, "device": DEVICE, "gpu_hint": GPU_HINT or None,
       "ov_version": ov.__version__, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
       "utc_start": utc(), "load_before": loadavg(), "error": None, "status": None}


def read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception as e:  # noqa: BLE001
        return f"unreadable: {type(e).__name__}"


def counters():
    t = os.times()
    return {"t": time.perf_counter(), "cpu": t[0] + t[1],
            "npu_busy_us": read_int("/sys/class/accel/accel0/device/npu_busy_time_us"),
            "gt0_idle_ms": read_int("/sys/class/drm/card1/device/tile0/gt0/gtidle/idle_residency_ms")}


def describe(ports):
    return [{"name": p.get_any_name(), "shape": str(p.get_partial_shape()), "type": str(p.get_element_type())}
            for p in ports]


def xml_types(path):
    text = open(path).read()
    return {"f16": text.count('element_type="f16"'), "f32": text.count('element_type="f32"')}


def load_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"unreadable image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def letterbox(rgb, H, W):
    h, w = rgb.shape[:2]
    s = min(H / h, W / w)
    nh, nw = int(round(h * s)), int(round(w * s))
    im = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((H, W, 3), np.uint8)
    y0, x0 = (H - nh) // 2, (W - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = im
    return out


def snapshot_sets():
    closes, pairs = [], []
    for run in ("any_1740", "any_1731"):
        for d in sorted(glob.glob(f"{ROOT}/logs/sort/{run}/ep_*")):
            c = sorted(glob.glob(f"{d}/*_close.jpg"))
            t = sorted(glob.glob(f"{d}/*_top.jpg"))
            if c:
                closes.append(c[0])
            if c and t:
                pairs.append((t[0], c[0]))
    return closes, pairs


def make_samples(inputs):
    closes, pairs = snapshot_sets()
    samples = []
    if KIND == "anomaly":
        for p in closes[:NPARITY]:
            img = load_rgb(p)
            arrs = []
            for inp in inputs:
                _, _, H, W = [int(d) for d in inp.get_shape()]
                x = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                arrs.append(np.ascontiguousarray(((x - MEAN) / STD).transpose(2, 0, 1)[None], dtype=np.float32))
            samples.append(arrs)
    else:
        img_idx = [i for i, inp in enumerate(inputs) if len(inp.get_shape()) == 4]
        for tp, cp in pairs[:NPARITY]:
            views = {0: load_rgb(tp), 1: load_rgb(cp)}
            arrs = []
            for i, inp in enumerate(inputs):
                shp = [int(d) for d in inp.get_shape()]
                if i in img_idx:
                    _, _, H, W = shp
                    x = letterbox(views[min(img_idx.index(i), 1)], H, W).astype(np.float32) / 255.0
                    arrs.append(np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32))
                else:
                    arrs.append(np.zeros(shp, np.float32))
            samples.append(arrs)
    rec["input_source"] = "recorded snapshots" if samples else "random (no recorded snapshots found)"
    if not samples:
        rng = np.random.default_rng(0)
        samples = [[rng.random([int(d) for d in inp.get_shape()], dtype=np.float32) for inp in inputs]
                   for _ in range(NPARITY)]
    rec["n_samples"] = len(samples)
    return samples


def fix_batch(model):
    shapes = {}
    for inp in model.inputs:
        ps = inp.get_partial_shape()
        if ps.rank.is_static and ps.rank.get_length() == 4 and (ps[0].is_dynamic or ps[0].get_length() != 1):
            shapes[inp.get_any_name()] = [1] + [int(ps[i].get_length()) for i in range(1, 4)]
    if shapes:
        model.reshape(shapes)
        rec["reshaped_to_batch1"] = shapes
    return model


def base_config():
    if DEVICE == "CPU":
        return {"INFERENCE_NUM_THREADS": 4}
    if DEVICE == "GPU" and GPU_HINT:
        return {"INFERENCE_PRECISION_HINT": GPU_HINT}
    return {}


def compile_timed(core, model, cfg):
    t0 = time.perf_counter()
    cm = core.compile_model(model, DEVICE, cfg)
    return cm, round(time.perf_counter() - t0, 3)


def main():
    core = ov.Core()
    rec["available_devices"] = core.available_devices
    rec["device_name"] = core.get_property(DEVICE, "FULL_DEVICE_NAME")
    rec["xml_element_types"] = xml_types(XML)
    model = fix_batch(core.read_model(XML))
    rec["inputs"], rec["outputs"] = describe(model.inputs), describe(model.outputs)
    cfg = base_config()
    rec["config"] = cfg
    cm, rec["compile_cold_s"] = compile_timed(core, model, cfg)
    req = cm.create_infer_request()
    samples = make_samples(cm.inputs)
    t0 = time.perf_counter()
    req.infer(samples[0])
    rec["first_infer_s"] = round(time.perf_counter() - t0, 4)
    rec["compile_plus_first_infer_s"] = round(rec["compile_cold_s"] + rec["first_infer_s"], 3)
    for i in range(WARMUP):
        req.infer(samples[i % len(samples)])
    c0 = counters()
    lat, i, t_end = [], 0, time.perf_counter() + SYNC_S
    while len(lat) < MAX_IT and time.perf_counter() < t_end:
        x = samples[i % len(samples)]
        i += 1
        t1 = time.perf_counter()
        req.infer(x)
        lat.append((time.perf_counter() - t1) * 1000.0)
    lat = np.asarray(lat)
    rec["latency_ms"] = {"n": int(lat.size), "p50": round(float(np.percentile(lat, 50)), 3),
                         "p90": round(float(np.percentile(lat, 90)), 3), "p99": round(float(np.percentile(lat, 99)), 3),
                         "mean": round(float(lat.mean()), 3), "min": round(float(lat.min()), 3),
                         "max": round(float(lat.max()), 3)}
    done = [0]
    queue = AsyncInferQueue(cm, NIREQ)

    def cb(request, userdata):
        done[0] += 1

    queue.set_callback(cb)
    ta, n = time.perf_counter(), 0
    while time.perf_counter() - ta < ASYNC_S:
        queue.start_async(samples[n % len(samples)], userdata=n)
        n += 1
    queue.wait_all()
    tb = time.perf_counter()
    c1 = counters()
    rec["async"] = {"nireq": NIREQ, "started": n, "done": done[0], "window_s": round(tb - ta, 3),
                    "fps": round(done[0] / (tb - ta), 2)}
    wall = c1["t"] - c0["t"]
    busy = {"window_s": round(wall, 3), "cpu_cores_avg": round((c1["cpu"] - c0["cpu"]) / wall, 2)}
    if isinstance(c0["npu_busy_us"], int) and isinstance(c1["npu_busy_us"], int):
        busy["npu_busy_pct"] = round((c1["npu_busy_us"] - c0["npu_busy_us"]) / (wall * 1e6) * 100, 2)
    if isinstance(c0["gt0_idle_ms"], int) and isinstance(c1["gt0_idle_ms"], int):
        busy["gpu_gt0_busy_pct"] = round((1 - (c1["gt0_idle_ms"] - c0["gt0_idle_ms"]) / (wall * 1000)) * 100, 2)
    rec["busy"] = busy
    outs, scores = {}, {}
    for si, s in enumerate(samples):
        req.infer(s)
        for j, o in enumerate(cm.outputs):
            arr = req.get_output_tensor(j).data.copy().astype(np.float32)
            outs[f"s{si}_o{j}"] = arr
            name = o.get_any_name()
            if "score" in name.lower():
                scores.setdefault(name, []).append(float(arr.ravel()[0]))
    np.savez(OUT_NPZ, **outs)
    rec["scores"] = scores
    rec["output_names"] = [o.get_any_name() for o in cm.outputs]
    if rec["compile_cold_s"] < 120:
        cache_dir = os.path.join(OUT_DIR, "ovcache", f"{TAG}__{DEVICE}")
        os.makedirs(cache_dir, exist_ok=True)
        ccfg = dict(cfg, CACHE_DIR=cache_dir)
        _, rec["compile_cache_write_s"] = compile_timed(core, model, ccfg)
        _, rec["compile_cache_read_s"] = compile_timed(core, model, ccfg)
        rec["cache_mb"] = round(sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(cache_dir)
                                    for f in fs) / 1e6, 1)
    else:
        rec["compile_cache_note"] = "skipped: cold compile >= 120 s"
    rec["load_after"] = loadavg()
    rec["status"] = "ok"


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {str(e)[:500]}"
        rec["status"] = "error"
    finally:
        rec["status"] = rec["status"] or "error"
        rec["utc_end"] = utc()
        with open(OUT_JSON, "w") as f:
            json.dump(rec, f, indent=1, default=str)
        print(json.dumps({k: rec.get(k) for k in ("tag", "device", "status", "compile_cold_s", "first_infer_s",
                                                   "latency_ms", "async", "busy", "error")}, default=str), flush=True)
