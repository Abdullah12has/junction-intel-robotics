"""OpenVINO latency per explicit device for the PaDiM pilot export (E3). The exported IR has a dynamic input, so it is
reshaped to a static 1x3x256x256 before compiling. Each device is tried separately; failures are recorded as text.
Run: envs/anomalib-env/bin/python -I -B work/P2-B-DET2/E3/tools/ov_latency_e3.py"""
import json
import os
import time

import numpy as np
import openvino as ov

WORK = "/home/intel/intel-junction-2026/work/P2-B-DET2/E3"
XML = os.path.join(WORK, "anom", "openvino_cpu", "weights", "openvino", "model.xml")
core = ov.Core()
res = {"available_devices": list(core.available_devices), "xml": XML}
for dev in ("CPU", "GPU", "NPU"):
    try:
        model = core.read_model(XML)
        model.reshape([1, 3, 256, 256])
        t0 = time.perf_counter()
        cm = core.compile_model(model, dev)
        compile_s = round(time.perf_counter() - t0, 2)
        x = np.random.rand(1, 3, 256, 256).astype(np.float32)
        req = cm.create_infer_request()
        for _ in range(5):
            req.infer({0: x})
        tms = []
        for _ in range(50):
            t1 = time.perf_counter()
            req.infer({0: x})
            tms.append((time.perf_counter() - t1) * 1000.0)
        res[dev] = {"compile_s": compile_s, "median_ms": round(float(np.median(tms)), 2),
                    "p90_ms": round(float(np.percentile(tms, 90)), 2), "input": [1, 3, 256, 256]}
    except Exception as exc:
        res[dev] = {"error": str(exc).strip().splitlines()[-1][:240]}
with open(os.path.join(WORK, "out", "ov_latency_e3.json"), "w") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps(res, indent=1))
