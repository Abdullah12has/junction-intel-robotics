import json
import os
import re
import sys

W = "/home/intel/intel-junction-2026/work/P2-B-OPT/E1"
RUNS = {
    "A_contended_timed_only": os.path.join(W, "bench_runA_contended.json"),
    "B_quiet_timed_only": os.path.join(W, "bench_runB_quiet.json"),
    "C_sustained_telemetry": os.path.join(W, "bench.json"),
}


def compact(r):
    lat = r.get("latency_ms") or {}
    tw = r.get("telemetry_during_timed") or {}
    sus = r.get("sustained") or {}
    st = sus.get("telemetry") or {}
    out = {
        "model": r.get("model"), "device": r.get("device"), "status": r.get("status"),
        "batch": r.get("batch"), "input_shapes": r.get("input_shapes"), "output_shapes": r.get("output_shapes"),
        "compile_s": r.get("compile_s"), "inference_precision_hint": r.get("inference_precision_hint"),
        "device_full_name": r.get("device_full_name"),
        "latency_ms": lat, "calls_per_s": r.get("throughput_ips"), "images_per_s": r.get("images_per_s"),
        "loadavg_1min_before_timed": r.get("loadavg_1min_before_timed"),
        "bench_process_cpu_cores": r.get("bench_process_cpu_cores"),
        "timed_window_utc": r.get("window_utc"),
        "timed_telemetry": {k: tw.get(k) for k in ["span_s", "samples", "pkg_temp_c_mean", "pkg_temp_c_max", "cpu_util_mean", "gpu_freq_ratio_mean", "npu_busy_mean"]},
        "pkg_w_mean": tw.get("pkg_w_mean"), "energy_per_inference_j": r.get("energy_per_inference_j"),
    }
    if r.get("error"):
        out["error"] = r.get("error")
        out["stage"] = r.get("stage")
    if sus:
        out["sustained"] = {
            "seconds": sus.get("seconds"), "inferences": sus.get("inferences"), "images_per_s": sus.get("images_per_s"),
            "window_utc": sus.get("window_utc"),
            "telemetry": {k: st.get(k) for k in ["span_s", "samples", "pkg_temp_c_mean", "pkg_temp_c_max", "core_max_c_mean", "cpu_util_mean", "gpu_freq_ratio_mean", "gpu_freq_mhz_mean", "npu_busy_mean", "npu_freq_mhz_mean", "ac_online_values"]},
        }
    return out


def parse_recipe(path):
    rows = []
    cur = None
    for line in open(path):
        line = line.rstrip("\n")
        m = re.match(r"^== (\S+) (CPU|GPU|NPU) rc=(\d+) wall_s=([\d.]+) npu_busy_time_us_delta=(-?\d+)", line)
        if m:
            cur = {"model": m.group(1), "device": m.group(2), "rc": int(m.group(3)), "wall_s": float(m.group(4)), "npu_busy_time_us_delta": int(m.group(5))}
            rows.append(cur)
            continue
        if cur is None:
            continue
        m = re.search(r"Median:\s*([\d.]+)\s*ms", line)
        if m:
            cur["median_ms"] = float(m.group(1))
        m = re.search(r"Throughput:\s*([\d.]+)\s*FPS", line)
        if m:
            cur["throughput_fps"] = float(m.group(1))
        m = re.search(r"EXECUTION_DEVICES\S*\s*[:=]?\s*(.*)$", line)
        if m and "execution_devices" not in cur:
            cur["execution_devices"] = m.group(1).strip()[:80]
    for r in rows:
        if r["wall_s"] and r.get("median_ms") is not None:
            r["npu_busy_fraction_of_wall"] = round((r["npu_busy_time_us_delta"] / 1e6) / r["wall_s"], 4)
    return rows


doc = {"runs": {}}
for key, path in RUNS.items():
    d = json.load(open(path))
    doc["runs"][key] = {
        "status": d.get("status"), "started_utc": d.get("started_utc"), "finished_utc": d.get("finished_utc"),
        "baseline_idle_30s": {k: (d.get("baseline_idle_30s") or {}).get(k) for k in ["pkg_temp_c_mean", "pkg_temp_c_max", "cpu_util_mean", "samples"]},
        "sampler_overhead_pct_during_bench": d.get("sampler_overhead_pct_during_bench"),
        "power": d.get("power"),
        "runs": [compact(r) for r in d["runs"]],
    }
doc["recipe_benchmark_app"] = parse_recipe(os.path.join(W, "benchmark_app", "summary.txt"))
doc["recipe_source"] = {
    "file": "/home/claude/kb/repos/robotics-ai-suite/src/components/object-detection/workshop/.workshop/object-detection.yaml",
    "lines": "38-47", "commit": "4544ddb214c25443924a8d908492c60d05e1f667",
    "command": "benchmark_app -m <IR> -d <CPU|GPU|NPU> -hint latency -t 10 (grep EXECUTION_DEVICES|Median:|Throughput:), npu_busy_time_us delta",
}
out = os.path.join(W, "bench_final.json")
with open(out, "w") as f:
    json.dump(doc, f, indent=1)
print("wrote", out, os.path.getsize(out))
print("| model | device | harness B p50 ms (quiet) | harness B images/s | harness C p50 ms (contended) | C sustained images/s | C sustained pkg temp mean/max C | benchmark_app median ms | benchmark_app FPS | NPU busy delta us | NPU busy fraction of wall |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
b = {(r["model"], r["device"]): r for r in doc["runs"]["B_quiet_timed_only"]["runs"]}
c = {(r["model"], r["device"]): r for r in doc["runs"]["C_sustained_telemetry"]["runs"]}
rec = {(r["model"], r["device"]): r for r in doc["recipe_benchmark_app"]}
keys = []
for r in doc["runs"]["B_quiet_timed_only"]["runs"]:
    keys.append((r["model"], r["device"]))
for k in keys:
    rb = b.get(k, {}); rc = c.get(k, {}); rr = rec.get((k[0].replace("patchcore_detector_native_b2", "patchcore_b2").replace("act_389da0fd_fp32", "act_fp32").replace("act_389da0fd_fp16", "act_fp16").replace("padim_b1_detector", "padim_b1"), k[1]), {})
    sus = rc.get("sustained") or {}
    st = sus.get("telemetry") or {}
    print("| %s | %s | %s | %s | %s | %s | %s/%s | %s | %s | %s | %s |" % (
        k[0], k[1],
        (rb.get("latency_ms") or {}).get("p50"), rb.get("images_per_s"),
        (rc.get("latency_ms") or {}).get("p50"), sus.get("images_per_s"),
        st.get("pkg_temp_c_mean"), st.get("pkg_temp_c_max"),
        rr.get("median_ms"), rr.get("throughput_fps"), rr.get("npu_busy_time_us_delta"), rr.get("npu_busy_fraction_of_wall"),
    ))
