"""OpenVINO latency of an exported anomalib IR on explicit devices (CPU, GPU, NPU), one 256x256 crop per call.

usage: python3 -I bench_ov.py <model.xml> <out.json> [runs]
Compile time, warm-up 10 calls, then <runs> timed calls (default 100). Device errors are recorded, not hidden.
"""
import json
import sys
import time

import numpy as np
import openvino as ov

xml, out = sys.argv[1], sys.argv[2]
runs = int(sys.argv[3]) if len(sys.argv) > 3 else 100
core = ov.Core()
model = core.read_model(xml)
ps = model.inputs[0].get_partial_shape()  # the exported IR is static (batch 2 for this export): use its own shape
shape = [int(d.get_length()) if d.is_static else (1 if i == 0 else int(d.get_max_length())) for i, d in enumerate(ps)]
x = np.random.default_rng(0).random(shape, dtype=np.float32)
res = {"model": xml, "input_shape": shape, "devices_available": core.available_devices, "devices": {}}
for dev in ("CPU", "GPU", "NPU"):
    entry = {}
    try:
        try:
            entry["full_name"] = core.get_property(dev, "FULL_DEVICE_NAME")
        except Exception as e:  # noqa: BLE001
            entry["full_name"] = f"unavailable: {type(e).__name__}"
        t0 = time.time()
        cm = core.compile_model(model, dev)
        entry["compile_s"] = round(time.time() - t0, 2)
        req = cm.create_infer_request()
        for _ in range(10):
            req.infer({0: x})
        ts = []
        for _ in range(runs):
            t1 = time.perf_counter()
            req.infer({0: x})
            ts.append((time.perf_counter() - t1) * 1000.0)
        ts = np.array(ts)
        entry.update({"runs": runs, "batch": shape[0], "median_ms_per_crop": round(float(np.median(ts)) / shape[0], 2),
                      "median_ms": round(float(np.median(ts)), 2),
                      "p90_ms": round(float(np.percentile(ts, 90)), 2), "mean_ms": round(float(ts.mean()), 2),
                      "fps_from_mean": round(1000.0 / float(ts.mean()), 1)})
    except Exception as e:  # noqa: BLE001
        entry["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    res["devices"][dev] = entry
    print(dev, entry)
json.dump(res, open(out, "w"), indent=1)
