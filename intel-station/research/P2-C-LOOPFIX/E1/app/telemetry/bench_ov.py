import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import openvino as ov

R = "/home/intel/intel-junction-2026"
W = os.path.join(R, "work", "P2-B-OPT", "E1")
OUT = os.path.join(W, "bench.json")
sys.path.insert(0, os.path.join(R, "app", "telemetry"))
import telemetry as tm  # noqa: E402

WARM, TIMED, COOL_S, SUSTAIN_S = 20, 200, 10, 30
DEVICES = ["CPU", "GPU", "NPU"]
ACT_XML = os.path.join(W, "models", "389da0fd", "act.xml")
ACT_FP16_XML = os.path.join(W, "models", "389da0fd_fp16", "act.xml")
PATCH_XML = os.path.join(R, "models", "anomaly", "patchcore", "weights", "openvino", "model.xml")
PADIM_B1_XML = os.path.join(R, "models", "anomaly", "padim", "weights", "openvino_b1", "model.xml")
MODELS = [
    {"key": "act_389da0fd_fp32", "label": "ACT 389da0fd as exported (IR weights FP32)", "xml": ACT_XML, "batch1": False},
    {"key": "act_389da0fd_fp16", "label": "ACT 389da0fd compressed to FP16 by save_model(compress_to_fp16=True)", "xml": ACT_FP16_XML, "batch1": False},
    {"key": "patchcore_detector_native_b2", "label": "PatchCore detector IR as exported, native batch 2 (weights FP16)", "xml": PATCH_XML, "batch1": False},
    {"key": "padim_b1_detector", "label": "PaDiM detector IR, native batch 1 (weights as exported)", "xml": PADIM_B1_XML, "batch1": False},
]


def now_iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def pct(a, p):
    return round(float(np.percentile(a, p)), 4)


def save(doc):
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, OUT)


def sampler_pid():
    with open(os.path.join(R, "logs", "telemetry", "telemetry.pid")) as f:
        return int(f.read().strip())


def proc_cpu_seconds(pid):
    with open("/proc/%d/stat" % pid) as f:
        parts = f.read().rsplit(")", 1)[1].split()
    return (int(parts[11]) + int(parts[12])) / os.sysconf("SC_CLK_TCK")


def make_feed(model, rng):
    feed = {}
    for inp in model.inputs:
        name = inp.get_any_name()
        arr = rng.random(list(inp.shape), dtype=np.float32)
        if "state" in name:
            arr = arr * 2.0 - 1.0
        feed[name] = arr
    return feed


def run_one(core, spec, device, full_name):
    rec = {"model": spec["key"], "label": spec["label"], "device": device, "device_full_name": full_name, "stage": "read"}
    try:
        model = core.read_model(spec["xml"])
        rec["input_shapes"] = {i.get_any_name(): list(i.shape) for i in model.inputs}
        rec["batch"] = int(list(model.inputs[0].shape)[0])
        rec["output_shapes"] = {o.get_any_name(): list(o.shape) for o in model.outputs}
        rec["stage"] = "compile"
        t0 = time.perf_counter()
        cm = core.compile_model(model, device, {"PERFORMANCE_HINT": "LATENCY"})
        rec["compile_s"] = round(time.perf_counter() - t0, 3)
        try:
            rec["inference_precision_hint"] = str(cm.get_property("INFERENCE_PRECISION_HINT"))
        except Exception as exc:
            rec["inference_precision_hint"] = "unreadable: " + type(exc).__name__
        rec["stage"] = "infer"
        req = cm.create_infer_request()
        feed = make_feed(model, np.random.default_rng(1234))
        for _ in range(WARM):
            req.infer(feed)
        before = tm.rolling(5)
        load_before = open("/proc/loadavg").read().split()[0]
        p0 = proc_cpu_seconds(os.getpid())
        lat = []
        w0 = time.time()
        for _ in range(TIMED):
            a = time.perf_counter_ns()
            req.infer(feed)
            lat.append((time.perf_counter_ns() - a) / 1e6)
        w1 = time.time()
        p1 = proc_cpu_seconds(os.getpid())
        arr = np.array(lat)
        rec["bench_process_cpu_cores"] = round((p1 - p0) / (w1 - w0), 3)
        rec["loadavg_1min_before_timed"] = float(load_before)
        rec["stage"] = "done"
        rec["status"] = "ok"
        rec["latency_ms"] = {
            "p50": pct(arr, 50), "p90": pct(arr, 90), "p95": pct(arr, 95),
            "mean": round(float(arr.mean()), 4), "std": round(float(arr.std()), 4),
            "min": round(float(arr.min()), 4), "max": round(float(arr.max()), 4),
        }
        rec["throughput_ips"] = round(TIMED / (float(arr.sum()) / 1000.0), 3)
        rec["images_per_s"] = round(rec["batch"] * rec["throughput_ips"], 3)
        rec["window_utc"] = [now_iso(w0), now_iso(w1)]
        tw = tm.window(w0, w1)
        rec["telemetry_during_timed"] = tw
        rec["telemetry_5s_before_timed"] = before
        s0 = time.time()
        n_sus = 0
        while time.time() - s0 < SUSTAIN_S:
            req.infer(feed)
            n_sus += 1
        s1 = time.time()
        rec["sustained"] = {
            "seconds": round(s1 - s0, 2), "inferences": n_sus,
            "ips": round(n_sus / (s1 - s0), 3), "images_per_s": round(rec["batch"] * n_sus / (s1 - s0), 3),
            "window_utc": [now_iso(s0), now_iso(s1)], "telemetry": tm.window(s0, s1),
        }
        pk = tw.get("pkg_w_mean")
        rec["energy_per_inference_j"] = round(pk * tw["span_s"] / TIMED, 5) if pk is not None else None
    except Exception as exc:
        rec["status"] = "failed"
        rec["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:600])
        rec["traceback_tail"] = traceback.format_exc()[-700:]
    return rec


def main():
    core = ov.Core()
    avail = core.available_devices
    full = {d: str(core.get_property(d, "FULL_DEVICE_NAME")) for d in DEVICES if d in avail}
    if not os.path.exists(ACT_FP16_XML):
        os.makedirs(os.path.dirname(ACT_FP16_XML), exist_ok=True)
        ov.save_model(core.read_model(ACT_XML), ACT_FP16_XML, compress_to_fp16=True)
    pid = sampler_pid()
    t_start = time.time()
    cpu0 = proc_cpu_seconds(pid)
    doc = {
        "status": "running",
        "started_utc": now_iso(t_start),
        "openvino": ov.__version__,
        "available_devices": avail,
        "full_device_name": full,
        "warmup": WARM,
        "timed": TIMED,
        "batch": 1,
        "cool_down_s": COOL_S,
        "power": {
            "pkg_w": "not measured: RAPL energy_uj is root-only on this laptop",
            "battery": "AC online; BAT0 power_now is not system power while on AC",
            "energy_per_inference_j": None,
        },
        "baseline_idle_30s": tm.rolling(30),
        "runs": [],
    }
    save(doc)
    for spec in MODELS:
        for dev in DEVICES:
            if dev not in avail:
                rec = {"model": spec["key"], "device": dev, "status": "not_available"}
            else:
                rec = run_one(core, spec, dev, full.get(dev))
            doc["runs"].append(rec)
            save(doc)
            time.sleep(COOL_S)
    t_end = time.time()
    doc["sampler_overhead_pct_during_bench"] = round(100.0 * (proc_cpu_seconds(pid) - cpu0) / (t_end - t_start), 3)
    doc["finished_utc"] = now_iso(t_end)
    doc["status"] = "done"
    save(doc)


if __name__ == "__main__":
    main()
