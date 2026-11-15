"""Time-boxed PaDiM pilot on the close-view crops (E3, D-068 Opus labels) in our own env (envs/anomalib-env, READY 14:43:03Z).
Train = Opus GOOD crops of episodes e03-e07. Test = Opus GOOD crops of e09-e14 (normal) and Opus DEFECT_TAPE crops (abnormal).
Crops touching the frame edge are skipped. Perturbed copies of the test set: JPEG q80 re-encode, 8 px shift.
Far too few images to support an accuracy claim; this checks the pipeline on XPU and records latency.
Run: envs/anomalib-env/bin/python -I -B work/P2-B-DET2/E3/tools/anom_padim_e3.py xpu|cpu"""
import json
import os
import shutil
import sys
import time

WORK = "/home/intel/intel-junction-2026/work/P2-B-DET2/E3"
APP = "/home/intel/intel-junction-2026/app/anomaly/det2_e3"
sys.path.insert(0, APP)
sys.path.insert(0, os.path.join(WORK, "tools"))

import cv2  # noqa: E402
import torch  # noqa: E402
import inspector  # noqa: E402
import eval_close_opus_e3 as ev  # noqa: E402
from anomalib.data import Folder  # noqa: E402
from anomalib.engine import Engine  # noqa: E402
from anomalib.models import Padim  # noqa: E402

ACC = sys.argv[1] if len(sys.argv) > 1 else "xpu"
TRAIN_EPS = {"e03", "e04", "e05", "e06", "e07"}
BASE = os.path.join(WORK, "anom")
RES = {"accelerator": ACC, "torch": torch.__version__,
       "xpu_available": bool(hasattr(torch, "xpu") and torch.xpu.is_available())}


def crops_for(fid):
    img = cv2.imread(f"{WORK}/frames/6d6bdce3_{fid}_243722064316.png")
    out = {}
    for r in inspector.scan(close_rgb=img):
        if r.get("bbox") is None:
            continue
        x, y, w, h = r["bbox"]
        p = 6
        out[r["id"]] = (img[max(0, y - p):y + h + p, max(0, x - p):x + w + p], r)
    return out


def build(root):
    if os.path.isdir(root):
        shutil.rmtree(root)
    for d in ("train/good", "test/good", "test/defect"):
        os.makedirs(os.path.join(root, d))
    cache, n = {}, {"train_good": 0, "test_good": 0, "test_defect": 0}
    for _, label, fid, k, _ in ev.MAP:
        if k is None or not (label == "GOOD" or label.startswith("DEFECT_TAPE")):
            continue
        if fid not in cache:
            cache[fid] = crops_for(fid)
        hit = cache[fid].get(f"scan_close_{k}")
        if hit is None or "touches_edge" in hit[1]["reasons"]:
            continue
        crop, _ = hit
        ep = fid.split("_")[0]
        if label == "GOOD":
            sub = "train/good" if ep in TRAIN_EPS else "test/good"
            n["train_good" if ep in TRAIN_EPS else "test_good"] += 1
        else:
            sub, _ = "test/defect", None
            n["test_defect"] += 1
        cv2.imwrite(os.path.join(root, sub, f"{fid}_k{k}.png"), crop)
    return n


def datamodule(root, eval_bs=4):
    return Folder(name="e3_close", root=root, normal_dir="train/good", abnormal_dir="test/defect",
                  normal_test_dir="test/good", train_batch_size=4, eval_batch_size=eval_bs, num_workers=0)


def perturbed(src, dst, mode):
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(os.path.join(src, "train"), os.path.join(dst, "train"))
    for sub in ("test/good", "test/defect"):
        os.makedirs(os.path.join(dst, sub))
        for f in sorted(os.listdir(os.path.join(src, sub))):
            im = cv2.imread(os.path.join(src, sub, f))
            if mode == "jpeg80":
                ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 80])
                im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            elif mode == "shift8":
                m = [[1, 0, 8], [0, 1, 8]]
                im = cv2.warpAffine(im, __import__("numpy").float32(m), (im.shape[1], im.shape[0]),
                                    borderMode=cv2.BORDER_REPLICATE)
            cv2.imwrite(os.path.join(dst, sub, f), im)
    return dst


def main():
    root = os.path.join(BASE, "close")
    RES["counts"] = build(root)
    print("dataset", RES["counts"], flush=True)
    model = Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"])
    engine = Engine(accelerator=ACC, devices=[0] if ACC == "xpu" else 1, default_root_dir=os.path.join(BASE, "runs"),
                    logger=False)
    t0 = time.perf_counter()
    engine.fit(model=model, datamodule=datamodule(root))
    RES["fit_s"] = round(time.perf_counter() - t0, 1)
    print("fit done", RES["fit_s"], "s", flush=True)
    RES["test_clean"] = engine.test(model=model, datamodule=datamodule(root))
    for mode in ("jpeg80", "shift8"):
        dst = perturbed(root, os.path.join(BASE, f"close_{mode}"), mode)
        RES[f"test_{mode}"] = engine.test(model=model, datamodule=datamodule(dst))
    print("tests done", flush=True)
    dm1 = datamodule(root, eval_bs=1)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        engine.predict(model=model, datamodule=dm1)
        times.append((time.perf_counter() - t0) / max(1, RES["counts"]["test_good"] + RES["counts"]["test_defect"]))
    RES[f"predict_s_per_image_{ACC}"] = round(min(times) * 1000.0, 2)
    print("predict ms/img", RES[f"predict_s_per_image_{ACC}"], flush=True)
    try:
        import glob
        import numpy as np
        import openvino as ov
        from anomalib.deploy import ExportType
        exp_root = os.path.join(BASE, f"openvino_{ACC}")
        engine.export(model=model, export_type=ExportType.OPENVINO, export_root=exp_root,
                      onnx_kwargs={"opset_version": 14})
        xml = sorted(glob.glob(os.path.join(exp_root, "**", "*.xml"), recursive=True))[0]
        RES["openvino_xml"] = xml
        core = ov.Core()
        RES["openvino_devices"] = list(core.available_devices)
        for dev in ("CPU", "GPU", "NPU"):
            try:
                cm = core.compile_model(xml, dev)
                shape = [int(s) for s in cm.input(0).shape]
                x = np.random.rand(*shape).astype(np.float32)
                req = cm.create_infer_request()
                for _ in range(3):
                    req.infer({0: x})
                tms = []
                for _ in range(30):
                    t0 = time.perf_counter()
                    req.infer({0: x})
                    tms.append((time.perf_counter() - t0) * 1000.0)
                RES[f"openvino_{dev}_ms_median"] = round(float(np.median(tms)), 2)
                RES[f"openvino_{dev}_input"] = shape
            except Exception as exc:  # record the device error, keep going
                RES[f"openvino_{dev}_error"] = str(exc)[:300]
    except Exception as exc:
        RES["openvino_export_error"] = str(exc)[:300]
    print("openvino block done", flush=True)
    with open(os.path.join(WORK, f"out/anom_padim_e3_{ACC}.json"), "w") as fh:
        json.dump(RES, fh, indent=1, default=str)
    print(json.dumps({k: RES[k] for k in RES if k.startswith("test_") or k.startswith("predict") or k in ("fit_s", "counts")}, default=str))


if __name__ == "__main__":
    main()
