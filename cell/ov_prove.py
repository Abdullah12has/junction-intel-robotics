#!/usr/bin/env python3
"""ov_prove.py MODEL.xml DEVICE [f16|f32]  -> devices, compile time, EXECUTION_DEVICES, precision, p50/p95."""
import json, sys, time
import numpy as np
import openvino as ov

core = ov.Core()
print("openvino", ov.get_version())
for d in core.available_devices:
    print(f"  {d:6s} {core.get_property(d, 'FULL_DEVICE_NAME')}")
xml, dev = sys.argv[1], sys.argv[2]
model = core.read_model(xml)
# NPU (and fastest GPU path) wants static shapes: pin dynamic batch dims to 1.
if any(i.get_partial_shape().is_dynamic for i in model.inputs):
    model.reshape({i.get_any_name(): [1 if d.is_dynamic else d.get_length() for d in i.get_partial_shape()]
                   for i in model.inputs})  # fails loudly if H/W are dynamic: re-export with input_size
cfg = {"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": "./ov_cache"}
if len(sys.argv) > 3: cfg["INFERENCE_PRECISION_HINT"] = sys.argv[3]
t = time.perf_counter(); cm = core.compile_model(model, dev, cfg); compile_s = time.perf_counter() - t
props = {}
for k in ("EXECUTION_DEVICES", "PERFORMANCE_HINT", "INFERENCE_PRECISION_HINT", "OPTIMAL_NUMBER_OF_INFER_REQUESTS"):
    try: props[k] = str(cm.get_property(k))
    except RuntimeError: props[k] = "n/a"
req = cm.create_infer_request()
feed = {i.get_any_name(): np.random.rand(*list(i.shape)).astype(i.get_element_type().to_dtype()) for i in cm.inputs}
t = time.perf_counter(); req.infer(feed); first_ms = (time.perf_counter() - t) * 1e3
lat = []
for _ in range(300):
    t = time.perf_counter(); req.infer(feed); lat.append((time.perf_counter() - t) * 1e3)
p50, p95 = np.percentile(lat, [50, 95])
print(json.dumps({"model": xml, "device": dev, "compile_s": round(compile_s, 2), "first_ms": round(first_ms, 1),
                  "p50_ms": round(p50, 2), "p95_ms": round(p95, 2), "fps": round(1e3 / np.mean(lat), 1), **props}))
