import glob
import os
import sys
import time

import cv2
import numpy as np
import openvino as ov

APP = "/home/intel/intel-junction-2026/app/anomaly/anom3"
sys.path.insert(0, APP)
import infer  # noqa: E402

SCR = "/home/intel/intel-junction-2026/work/P2-B-ANOM3/E1/scratch"
D = "/home/intel/intel-junction-2026/work/P2-B-DET2/E2/anomalib/bricks_close"
core = ov.Core()
print("devices", core.available_devices, flush=True)
fp32_xml = sorted(glob.glob(os.path.join(SCR, "export", "smoke1", "pre_close", "**", "*.xml"), recursive=True))[0]
fp16_xml = os.path.join(SCR, "smoke_models", "smoke1", "pre_close", "model.xml")
paths = sorted(glob.glob(os.path.join(D, "test", "good", "*.png")))[:2] + sorted(glob.glob(os.path.join(D, "test", "defect", "*.png")))[:2]
bgrs = [cv2.imread(p, cv2.IMREAD_COLOR) for p in paths]
for dev in ("CPU", "GPU"):
    for tag, xml in (("fp32", fp32_xml), ("fp16", fp16_xml)):
        try:
            cfg = {"INFERENCE_PRECISION_HINT": "f32"} if dev == "GPU" else {}
            cm = core.compile_model(core.read_model(xml), dev, cfg)
        except Exception as exc:
            print(dev, tag, "compile error", str(exc)[:200], flush=True)
            continue
        req = cm.create_infer_request()
        for start in (0, 2):
            x = np.stack([infer.preprocess(b) for b in bgrs[start:start + 2]]).astype(np.float32)
            t0 = time.perf_counter()
            res = req.infer({0: x})
            ms = (time.perf_counter() - t0) * 1000
            v = np.asarray(res[cm.output("pred_score")]).reshape(-1)
            amap = np.asarray(res[cm.output("anomaly_map")])
            print(dev, tag, "pair", start, "pred_score", v.round(3).tolist(), "nan", int(np.isnan(v).sum()),
                  "map_nan", int(np.isnan(amap).sum()), "ms %.1f" % ms, flush=True)
