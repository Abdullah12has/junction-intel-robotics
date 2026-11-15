import hashlib
import json
import os

W = "/home/intel/intel-junction-2026/work/P2-B-OPT/E1"
SRC = os.path.join(W, "bench_final.json")
OUT = os.path.join(W, "bench_compact.json")


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


raw = open(SRC, "rb").read()
d = json.load(open(SRC))
RUNS = d["runs"]
c_doc = json.load(open(os.path.join(W, "bench.json")))


def lat(r, k):
    return (r.get("latency_ms") or {}).get(k)


def sus(r, k):
    return (r.get("sustained") or {}).get(k)


def sus_tel(r, k):
    return (((r.get("sustained") or {}).get("telemetry")) or {}).get(k)


def row_b(r):
    return {
        "model": r.get("model"), "device": r.get("device"), "batch": r.get("batch"),
        "status": r.get("status"),
        "p50_ms": lat(r, "p50"), "p90_ms": lat(r, "p90"), "p95_ms": lat(r, "p95"), "mean_ms": lat(r, "mean"),
        "calls_per_s": r.get("calls_per_s"), "images_per_s": r.get("images_per_s"),
        "compile_s": r.get("compile_s"), "precision_hint": r.get("inference_precision_hint"),
        "loadavg_1min_before_timed": r.get("loadavg_1min_before_timed"),
        "bench_cpu_cores": r.get("bench_process_cpu_cores"),
    }


def row_c(r):
    x = row_b(r)
    x.update({
        "sustained_seconds": sus(r, "seconds"), "sustained_inferences": sus(r, "inferences"),
        "sustained_images_per_s": sus(r, "images_per_s"),
        "sustained_pkg_temp_mean_c": sus_tel(r, "pkg_temp_c_mean"),
        "sustained_pkg_temp_max_c": sus_tel(r, "pkg_temp_c_max"),
        "sustained_cpu_util_mean": sus_tel(r, "cpu_util_mean"),
        "sustained_gpu_freq_ratio_mean": sus_tel(r, "gpu_freq_ratio_mean"),
        "sustained_npu_busy_mean": sus_tel(r, "npu_busy_mean"),
        "sustained_samples": sus_tel(r, "samples"),
    })
    return x


def row_a(r):
    e = r.get("error")
    return {
        "model": r.get("model"), "device": r.get("device"), "status": r.get("status"),
        "stage": r.get("stage"), "p50_ms": lat(r, "p50"), "p95_ms": lat(r, "p95"),
        "calls_per_s": r.get("calls_per_s"),
        "loadavg_1min_before_timed": r.get("loadavg_1min_before_timed"),
        "error": e[:160] if e else None,
    }


def row_recipe(r):
    keys = ["model", "device", "rc", "wall_s", "median_ms", "throughput_fps",
            "execution_devices", "npu_busy_time_us_delta", "npu_busy_fraction_of_wall"]
    return {k: r.get(k) for k in keys}


def pick(rows, model, device):
    for x in rows:
        if x["model"] == model and x["device"] == device:
            return x
    raise KeyError((model, device))


def budget_for(rows, act_dev, pc_dev):
    a = pick(rows, "act_389da0fd_fp32", act_dev)
    p = pick(rows, "patchcore_detector_native_b2", pc_dev)
    return {
        "act": [act_dev, a["p50_ms"], a["p95_ms"]],
        "patchcore_b2": [pc_dev, p["p50_ms"], p["p95_ms"]],
        "sum_p50_ms": round(a["p50_ms"] + p["p50_ms"], 4),
        "sum_p95_ms": round(a["p95_ms"] + p["p95_ms"], 4),
    }


def one(x):
    return json.dumps(x, separators=(",", ":"), ensure_ascii=False)


def block(name, head, rows):
    body = ",\n".join("  " + one(x) for x in rows)
    return '"' + name + '": {"head": ' + one(head) + ',\n"rows": [\n' + body + "\n]}"


def head_of(blk, revision):
    return {
        "bench_ov_revision": revision,
        "status": blk.get("status"), "started_utc": blk.get("started_utc"),
        "finished_utc": blk.get("finished_utc"),
        "sampler_overhead_pct": blk.get("sampler_overhead_pct_during_bench"),
        "baseline_idle_30s": blk.get("baseline_idle_30s"),
        "power": blk.get("power"),
    }


B_rows = [row_b(r) for r in RUNS["B_quiet_timed_only"]["runs"]]
C_rows = [row_c(r) for r in RUNS["C_sustained_telemetry"]["runs"]]
A_rows = [row_a(r) for r in RUNS["A_contended_timed_only"]["runs"]]
padim_b = pick(B_rows, "padim_b1_detector", "GPU")
budget = {
    "definition": "model compute only per call: one ACT call (batch 1) plus one PatchCore inspection (batch 2, two crops); excludes capture, preprocessing, postprocessing and robot motion; derived sums, not an end-to-end cycle time",
    "quiet_B_act_gpu_patchcore_gpu": budget_for(B_rows, "GPU", "GPU"),
    "quiet_B_act_npu_patchcore_gpu": budget_for(B_rows, "NPU", "GPU"),
    "contended_C_act_gpu_patchcore_gpu": budget_for(C_rows, "GPU", "GPU"),
    "contended_C_act_npu_patchcore_gpu": budget_for(C_rows, "NPU", "GPU"),
    "padim_b1_gpu_quiet_p50_p95_ms": [padim_b["p50_ms"], padim_b["p95_ms"]],
}
meta = {
    "source_file": SRC, "source_sha256": sha_bytes(raw), "source_bytes": len(raw),
    "openvino": c_doc.get("openvino"), "available_devices": c_doc.get("available_devices"),
    "full_device_name": c_doc.get("full_device_name"),
    "raw_sha256": {n: sha_bytes(open(os.path.join(W, n), "rb").read())
                   for n in ["bench_runA_contended.json", "bench_runB_quiet.json", "bench.json",
                             "bench_final.json", "benchmark_app/summary.txt"]},
}
method = {
    "warmup": 20, "timed": 200, "batch": "as exported (ACT 1, PatchCore 2, PaDiM 1)",
    "timer": "time.perf_counter_ns around each req.infer call",
    "percentiles": "numpy.percentile (linear)", "hint": "PERFORMANCE_HINT=LATENCY",
    "explicit_devices": ["CPU", "GPU", "NPU"], "random_input_seed": 1234,
    "throughput": "calls_per_s = 200 / sum(latency); images_per_s = batch * calls_per_s",
    "sustained": "30 s back-to-back infer loop after the timed block (run C only)",
    "cool_down_s": {"A": 15, "B": 15, "C": 10},
    "recipe": "benchmark_app -m <IR> -d <CPU|GPU|NPU> -hint latency -t 10; npu_busy_time_us delta; source RAS object-detection.yaml L38-47 at commit 4544ddb214c25443924a8d908492c60d05e1f667",
}
host = {
    "hostname": "intel-HP-OmniBook-Ultra-Laptop-14t-kd000", "product": "HP OmniBook Ultra 14",
    "cpu": "Intel Core Ultra X9 388H", "os": "Ubuntu 26.04.1 LTS", "kernel": "7.0.0-38-generic",
    "python": "3.12.16", "numpy": "2.4.6",
}
parts = [
    '"meta": ' + one(meta),
    '"host": ' + one(host),
    '"method": ' + one(method),
    '"budget_model_compute": ' + one(budget),
    block("runs_B_quiet", head_of(RUNS["B_quiet_timed_only"], "rev2 (cool-down 15 s, no sustained phase)"), B_rows),
    block("runs_C_sustained", head_of(RUNS["C_sustained_telemetry"], "rev3 final (cool-down 10 s, 30 s sustained phase)"), C_rows),
    block("runs_A_contended", head_of(RUNS["A_contended_timed_only"], "rev1 (cool-down 15 s, PatchCore batch-1 reshape, no sustained phase)"), A_rows),
    '"recipe_benchmark_app": [\n' + ",\n".join("  " + one(row_recipe(r)) for r in d["recipe_benchmark_app"]) + "\n]",
]
text = "{\n" + ",\n".join(parts) + "\n}\n"
json.loads(text)
with open(OUT, "w") as f:
    f.write(text)
print("written", OUT, len(text.encode()), sha_bytes(text.encode()))
