import json
import sys

d = json.load(open(sys.argv[1]))
cols = ["model", "device", "status", "stage", "compile_s", "p50_ms", "p90_ms", "p95_ms", "mean_ms", "ips", "pkg_temp_mean_C", "pkg_temp_max_C", "cpu_util_mean", "gpu_freq_mhz_mean", "npu_busy_mean", "bench_cpu_cores", "loadavg1_before", "precision_hint", "pkg_w_mean", "energy_J_per_inf"]
print("| " + " | ".join(cols) + " |")
print("|" + "---|" * len(cols))
for r in d["runs"]:
    lat = r.get("latency_ms", {}) or {}
    tw = r.get("telemetry_during_timed", {}) or {}
    vals = [
        r.get("model"), r.get("device"), r.get("status"), r.get("stage"), r.get("compile_s"),
        lat.get("p50"), lat.get("p90"), lat.get("p95"), lat.get("mean"), r.get("throughput_ips"),
        tw.get("pkg_temp_c_mean"), tw.get("pkg_temp_c_max"), tw.get("cpu_util_mean"), tw.get("gpu_freq_mhz_mean"),
        tw.get("npu_busy_mean"), r.get("bench_process_cpu_cores"), r.get("loadavg_1min_before_timed"),
        r.get("inference_precision_hint"), tw.get("pkg_w_mean"), r.get("energy_per_inference_j"),
    ]
    print("| " + " | ".join("-" if v is None else str(v) for v in vals) + " |")
print()
for r in d["runs"]:
    if r.get("status") != "ok":
        print("FAILED", r.get("model"), r.get("device"), "stage=" + str(r.get("stage")), str(r.get("error")))
