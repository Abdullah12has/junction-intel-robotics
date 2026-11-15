import glob
import os
import sys

import cv2
import numpy as np
import openvino as ov

APP = "/home/intel/intel-junction-2026/app/anomaly/anom3"
sys.path.insert(0, APP)
import infer  # noqa: E402

SCR = "/home/intel/intel-junction-2026/work/P2-B-ANOM3/E1/scratch"
D = "/home/intel/intel-junction-2026/work/P2-B-DET2/E2/anomalib/bricks_close"
core = ov.Core()
fp32_xml = sorted(glob.glob(os.path.join(SCR, "export", "smoke1", "pre_close", "**", "*.xml"), recursive=True))
fp16_xml = os.path.join(SCR, "smoke_models", "smoke1", "pre_close", "model.xml")
paths = sorted(glob.glob(os.path.join(D, "test", "good", "*.png")))[:2] + sorted(glob.glob(os.path.join(D, "test", "defect", "*.png")))[:2]
bgrs = [cv2.imread(p, cv2.IMREAD_COLOR) for p in paths]
print("images", [(os.path.basename(p), None if b is None else b.shape) for p, b in zip(paths, bgrs)])
for tag, xml in (("fp32", fp32_xml[0]), ("fp16", fp16_xml)):
    m = core.read_model(xml)
    print(tag, "outputs", [(p.get_any_name(), str(p.get_partial_shape()), str(p.get_element_type())) for p in m.outputs])
    cm = core.compile_model(m, "CPU")
    req = cm.create_infer_request()
    for start in (0, 2):
        x = np.stack([infer.preprocess(b) for b in bgrs[start:start + 2]]).astype(np.float32)
        res = req.infer({0: x})
        for port in cm.outputs:
            v = np.asarray(res[port])
            print(tag, "pair", start, port.get_any_name(), "nan", int(np.isnan(v).sum()), "inf", int(np.isinf(v).sum()),
                  "min %.4g max %.4g" % (float(np.nanmin(v)), float(np.nanmax(v))), v.reshape(-1)[:4].tolist()
                  if port.get_any_name() == "pred_score" else "")
    print(tag, "x range", float(x.min()), float(x.max()), x.shape)
