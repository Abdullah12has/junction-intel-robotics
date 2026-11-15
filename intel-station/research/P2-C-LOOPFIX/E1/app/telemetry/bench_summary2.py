import json
import sys

d = json.load(open(sys.argv[1]))
cols = ["model", "device", "batch", "status", "compile_s", "p50_ms", "p90_ms", "p95_ms", "mean_ms", "calls_per_s", "images_per_s",
        "sus_seconds", "sus_images_per_s", "sus_pkg_temp_mean_C", "sus_pkg_temp_max_C", "sus_cpu_util_mean", "sus_gpu_freq_ratio_mean",
        "sus_npu_busy_mean", "sus_samples", "precision_hint", "bench_cpu_cores", "loadavg1_before"]
print("| " + " | ".join(cols) + " |")
print("|" + "---|" * len(cols))
for r in d["runs"]:
    lat = r.get("latency_ms") or {}
    sus = r.get("sustained") or {}
    st = sus.get("telemetry") or {}
    vals = [
        r.get("model"), r.get("device"), r.get("batch"), r.get("status"), r.get("compile_s"),
        lat.get("p50"), lat.get("p90"), lat.get("p95"), lat.get("mean"), r.get("throughput_ips"), r.get("images_per_s"),
        sus.get("seconds"), sus.get("images_per_s"), st.get("pkg_temp_c_mean"), st.get("pkg_temp_c_max"), st.get("cpu_util_mean"),
        st.get("gpu_freq_ratio_mean"), st.get("npu_busy_mean"), st.get("samples"),
        r.get("inference_precision_hint"), r.get("bench_process_cpu_cores"), r.get("loadavg_1min_before_timed"),
    ]
    print("| " + " | ".join("-" if v is None else str(v) for v in vals) + " |")
print()
for r in d["runs"]:
    if r.get("status") != "ok":
        print("FAILED", r.get("model"), r.get("device"), "stage=" + str(r.get("stage")), str(r.get("error")))
print()
print("baseline_idle_30s pkg_temp_mean=%s pkg_temp_max=%s cpu_util_mean=%s samples=%s" % (
    d["baseline_idle_30s"].get("pkg_temp_c_mean"), d["baseline_idle_30s"].get("pkg_temp_c_max"),
    d["baseline_idle_30s"].get("cpu_util_mean"), d["baseline_idle_30s"].get("samples")))
print("sampler_overhead_pct_during_bench=%s status=%s started=%s finished=%s" % (
    d.get("sampler_overhead_pct_during_bench"), d.get("status"), d.get("started_utc"), d.get("finished_utc")))
