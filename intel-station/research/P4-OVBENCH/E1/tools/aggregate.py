#!/usr/bin/env python3
"""E1 P4-OVBENCH aggregate: bench.tsv (one row per model x device run) and metrics.json (stage ovbench-v1).
usage: aggregate.py RAW_DIR OUT_DIR"""
import glob
import json
import os
import sys
import time

import numpy as np

RAW, OUT = sys.argv[1], sys.argv[2]
recs = {}
for p in sorted(glob.glob(os.path.join(RAW, "*.json"))):
    r = json.load(open(p))
    recs[(r["tag"], r["device"])] = r


def parity(tag, dev):
    a, b = os.path.join(RAW, f"{tag}__CPU__out.npz"), os.path.join(RAW, f"{tag}__{dev}__out.npz")
    if not (os.path.exists(a) and os.path.exists(b)):
        return None, None, None
    if dev == "CPU":
        return 0.0, 0.0, 0.0
    A, B = np.load(a), np.load(b)
    mad, ref = 0.0, 0.0
    for k in A.files:
        x, y = A[k].astype(np.float64), B[k].astype(np.float64)
        mad = max(mad, float(np.max(np.abs(x - y))))
        ref = max(ref, float(np.max(np.abs(x))))
    base = recs.get((tag, "CPU"), {}).get("scores", {})
    cur = recs.get((tag, dev), {}).get("scores", {})
    smad = None
    for name, vals in base.items():
        if name in cur and len(cur[name]) == len(vals):
            d = max(abs(p - q) for p, q in zip(vals, cur[name]))
            smad = d if smad is None else max(smad, d)
    return mad, (mad / ref if ref else None), smad


def g(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


COLS = ["tag", "device", "status", "gpu_hint", "ir_f16", "ir_f32", "compile_cold_s", "first_infer_s",
        "cache_write_s", "cache_read_s", "p50_ms", "p90_ms", "p99_ms", "mean_ms", "n_sync", "async_fps",
        "cpu_cores_avg", "npu_busy_pct", "gpu_gt0_busy_pct", "parity_max_abs_vs_cpu", "parity_max_rel",
        "score_max_abs_vs_cpu", "device_name", "error"]
lines = ["\t".join(COLS)]
for (tag, dev), r in sorted(recs.items()):
    mad, rel, smad = parity(tag, dev)
    vals = [tag, dev, r.get("status"), r.get("gpu_hint") or "", g(r, "xml_element_types", "f16"),
            g(r, "xml_element_types", "f32"), r.get("compile_cold_s"), r.get("first_infer_s"),
            r.get("compile_cache_write_s"), r.get("compile_cache_read_s"), g(r, "latency_ms", "p50"),
            g(r, "latency_ms", "p90"), g(r, "latency_ms", "p99"), g(r, "latency_ms", "mean"), g(r, "latency_ms", "n"),
            g(r, "async", "fps"), g(r, "busy", "cpu_cores_avg"), g(r, "busy", "npu_busy_pct"),
            g(r, "busy", "gpu_gt0_busy_pct"), mad, rel, smad, r.get("device_name"), r.get("error")]
    lines.append("\t".join("" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v)) for v in vals))
open(os.path.join(OUT, "bench.tsv"), "w").write("\n".join(lines) + "\n")

per_tag, best = {}, {}
for (tag, dev), r in sorted(recs.items()):
    entry = {"status": r.get("status")}
    if r.get("status") == "ok":
        entry.update({"p50_ms": r["latency_ms"]["p50"], "p90_ms": r["latency_ms"]["p90"],
                      "compile_cold_s": r.get("compile_cold_s"), "first_infer_s": r.get("first_infer_s"),
                      "async_fps": r["async"]["fps"]})
    per_tag.setdefault(tag, {})[dev] = entry
for tag, devs in per_tag.items():
    ok = [(v["p50_ms"], d) for d, v in devs.items() if "p50_ms" in v]
    if ok:
        best[tag] = {"device": min(ok)[1], "p50_ms": min(ok)[0]}
metrics = {"runs": len(recs), "runs_ok": sum(1 for r in recs.values() if r.get("status") == "ok"),
           "per_tag": per_tag, "best_device_by_p50": best}
json.dump({"stage": "ovbench-v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "metrics": metrics,
           "previous": None, "delta": None}, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)
print(open(os.path.join(OUT, "bench.tsv")).read())
print(json.dumps(best, indent=1))
