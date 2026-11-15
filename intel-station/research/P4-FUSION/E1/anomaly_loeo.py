"""P4-FUSION E1 step 2: PatchCore and PaDiM (Anomalib 2.7.0, CPU fit) on GOOD-only crops, leave-one-episode-out.

Per model and per held-out group h (GOOD or DEFECT): fit on the GOOD crops of the other groups, export OpenVINO FP16 IR,
score the crops of h through OpenVINO on CPU. Threshold of that fold = max score of its own training GOOD crops (chosen on
the training folds only; conservative). Then one all-GOOD model per architecture is fitted and scored through OpenVINO on
CPU, GPU and NPU (device agreement). Scores in E/eval/scores.tsv, device agreement in E/eval/device_check.tsv.
"""
import csv
import glob
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np
import torch

R = "/home/intel/intel-junction-2026"
E = os.path.join(R, "work", "P4-FUSION", "E1")
sys.path.insert(0, os.path.join(R, "app", "anomaly", "anom3"))
import infer  # noqa: E402  (anom3 preprocessing: BGR -> RGB, resize, ImageNet normalisation)
import openvino as ov  # noqa: E402
from anomalib.data import Folder  # noqa: E402
from anomalib.deploy import ExportType  # noqa: E402
from anomalib.engine import Engine  # noqa: E402
from anomalib.models import Padim, Patchcore  # noqa: E402
from anomalib.post_processing.post_processor import PostProcessor  # noqa: E402

SIZE = 256
SEED = 0
MODELS = ("patchcore", "padim")
WORK = os.path.join(E, "scratch")
MODELS_DIR = os.path.join(E, "models")
EVAL = os.path.join(E, "eval")
DEVICES = ("CPU", "GPU", "NPU")


def safe(name):
    return name.replace("/", "__")


def load_items():
    out = []
    with open(os.path.join(E, "index.tsv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["status"] == "ok" and r["crop"]:
                out.append(r)
    return out


def train_dir(root, goods):
    shutil.rmtree(root, ignore_errors=True)
    d = os.path.join(root, "train", "good")
    os.makedirs(d)
    for i, r in enumerate(goods):
        img = cv2.imread(r["crop"], cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(d, "%04d.png" % i), img)
    return len(goods)


def fit_export(model_name, root, n, ir_dir, tag):
    torch.manual_seed(SEED)
    dm = Folder(name=tag, root=root, normal_dir="train/good", train_batch_size=4, eval_batch_size=4,
                num_workers=0, test_split_mode="none", val_split_mode="from_train", val_split_ratio=0.2,
                seed=SEED)
    if model_name == "patchcore":
        pre = Patchcore.configure_pre_processor(image_size=(SIZE, SIZE))
        ratio = float(min(0.1, max(0.001, 2000.0 / float(n * (SIZE // 8) ** 2))))
        model = Patchcore(backbone="wide_resnet50_2", layers=("layer2", "layer3"), pre_trained=True,
                          coreset_sampling_ratio=ratio, num_neighbors=9, pre_processor=pre,
                          post_processor=PostProcessor(enable_normalization=False))
    else:
        pre = Padim.configure_pre_processor(image_size=(SIZE, SIZE))
        ratio = None
        model = Padim(backbone="resnet18", layers=("layer1", "layer2", "layer3"), pre_trained=True,
                      pre_processor=pre, post_processor=PostProcessor(enable_normalization=False))
    engine = Engine(accelerator="cpu", devices=1, max_epochs=1, logger=False,
                    default_root_dir=os.path.join(WORK, tag, "runs"))
    t0 = time.perf_counter()
    engine.fit(model=model, datamodule=dm)
    fit_s = round(time.perf_counter() - t0, 2)
    exp_root = os.path.join(WORK, tag, "export")
    shutil.rmtree(exp_root, ignore_errors=True)
    t1 = time.perf_counter()
    engine.export(model=model, export_type=ExportType.OPENVINO, export_root=exp_root, input_size=(SIZE, SIZE),
                  onnx_kwargs={"opset_version": 14})
    export_s = round(time.perf_counter() - t1, 2)
    found = sorted(glob.glob(os.path.join(exp_root, "**", "*.xml"), recursive=True))
    if not found:
        raise RuntimeError("export wrote no IR under " + exp_root)
    shutil.rmtree(ir_dir, ignore_errors=True)
    os.makedirs(ir_dir)
    ov.save_model(ov.Core().read_model(found[0]), os.path.join(ir_dir, "model.xml"), compress_to_fp16=True)
    return fit_s, export_s, ratio


class IR:
    """OpenVINO IR scorer. GPU runs FP32 precision (as anom3 infer.py: FP16 GPU overflowed on squared distances)."""

    def __init__(self, xml, device):
        core = ov.Core()
        cfg = {"INFERENCE_PRECISION_HINT": "f32"} if device == "GPU" else {}
        self.compiled = core.compile_model(xml, device, cfg)
        self.device = device
        self.req = self.compiled.create_infer_request()
        self.port, self.is_map = None, False
        for p in self.compiled.outputs:
            try:
                names = p.get_names()
            except RuntimeError:
                names = set()
            if any("pred_score" in n for n in names):
                self.port, self.is_map = p, False
                break
        if self.port is None:
            for p in self.compiled.outputs:
                try:
                    names = p.get_names()
                except RuntimeError:
                    names = set()
                if any("anomaly_map" in n for n in names):
                    self.port, self.is_map = p, True
                    break
        if self.port is None:
            raise RuntimeError("IR has no pred_score or anomaly_map output")

    def score(self, bgr_list):
        out = []
        pshape = self.compiled.input(0).partial_shape
        b = pshape[0].get_length() if pshape[0].is_static else 1
        for start in range(0, len(bgr_list), b):
            chunk = list(bgr_list[start:start + b])
            real = len(chunk)
            xs = [infer.preprocess(c, SIZE) for c in chunk]
            xs += [xs[-1]] * (b - real)
            x = np.stack(xs).astype(np.float32)
            res = np.asarray(self.req.infer({0: x})[self.port], dtype=np.float32).reshape(b, -1)
            vals = res.max(axis=1) if self.is_map else res[:, 0]
            out.extend(float(v) for v in vals[:real])
        return out


def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    os.makedirs(EVAL, exist_ok=True)
    items = load_items()
    groups = sorted({r["group"] for r in items})
    print("items", len(items), "groups", len(groups), flush=True)
    rows, fits = [], []
    for model_name in MODELS:
        for h in groups:
            goods = [r for r in items if r["truth"] == "GOOD" and r["group"] != h]
            test = [r for r in items if r["group"] == h]
            if not goods or not test:
                print("skip", model_name, h, flush=True)
                continue
            tag = "%s_fold_%s" % (model_name, safe(h))
            root = os.path.join(WORK, tag, "build")
            n = train_dir(root, goods)
            ir_dir = os.path.join(MODELS_DIR, model_name, "fold_" + safe(h))
            fit_s, export_s, ratio = fit_export(model_name, root, n, ir_dir, tag)
            m = IR(os.path.join(ir_dir, "model.xml"), "CPU")
            train_scores = m.score([cv2.imread(r["crop"], cv2.IMREAD_COLOR) for r in goods])
            thr = max(train_scores)
            test_scores = m.score([cv2.imread(r["crop"], cv2.IMREAD_COLOR) for r in test])
            with open(os.path.join(ir_dir, "threshold.json"), "w", encoding="utf-8") as fh:
                json.dump({"threshold": thr, "rule": "max score of the fold's training GOOD crops (in-sample)",
                           "n_train": n}, fh, indent=1)
            fit = {"model": model_name, "heldout": h, "n_train_crops": n, "fit_s": fit_s, "export_s": export_s,
                   "coreset_ratio": ratio, "threshold": round(thr, 4)}
            fits.append(fit)
            for r, s in zip(test, test_scores):
                rows.append({"model": model_name, "fold": h, "rec_id": r["rec_id"], "group": r["group"],
                             "truth": r["truth"], "truth_status": r["truth_status"], "family": r["family"],
                             "score": "%.5f" % s, "threshold": "%.5f" % thr, "device": "CPU", "n_train": n})
            print("FOLD", json.dumps(fit), "held", h, "truth", test[0]["truth"],
                  "mean_test", round(float(np.mean(test_scores)), 3), flush=True)
    cols = ["model", "fold", "rec_id", "group", "truth", "truth_status", "family", "score", "threshold", "device",
            "n_train"]
    with open(os.path.join(EVAL, "scores.tsv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(EVAL, "fits.json"), "w", encoding="utf-8") as fh:
        json.dump(fits, fh, indent=1)
    # all-GOOD model per architecture and device agreement
    goods = [r for r in items if r["truth"] == "GOOD"]
    dev_rows = []
    for model_name in MODELS:
        tag = "%s_all_good" % model_name
        root = os.path.join(WORK, tag, "build")
        n = train_dir(root, goods)
        ir_dir = os.path.join(MODELS_DIR, model_name, "all_good")
        fit_s, export_s, ratio = fit_export(model_name, root, n, ir_dir, tag)
        imgs = [cv2.imread(r["crop"], cv2.IMREAD_COLOR) for r in items]
        ref = None
        cpu_thr = None
        for dev in DEVICES:
            rec = {"model": model_name, "device": dev}
            try:
                m = IR(os.path.join(ir_dir, "model.xml"), dev)
                t0 = time.perf_counter()
                sc = np.asarray(m.score(imgs), dtype=np.float64)
                rec["latency_ms_per_crop"] = round((time.perf_counter() - t0) * 1000.0 / len(imgs), 2)
                rec["status"] = "ok"
                if dev == "CPU":
                    ref = sc
                    cpu_thr = max(sc[[i for i, r in enumerate(items) if r["truth"] == "GOOD"]])
                    with open(os.path.join(ir_dir, "threshold.json"), "w", encoding="utf-8") as fh:
                        json.dump({"threshold": float(cpu_thr), "rule": "max score of all training GOOD crops"
                                   " (in-sample)", "n_train": n}, fh, indent=1)
                if ref is not None:
                    rec["max_abs_diff_vs_cpu"] = float(np.max(np.abs(sc - ref)))
                    rec["max_rel_diff_vs_cpu"] = float(np.max(np.abs(sc - ref) / np.maximum(np.abs(ref), 1e-9)))
                rec["device_used"] = m.device
                rec["n_crops"] = len(imgs)
                rec["first_scores"] = [round(float(v), 4) for v in sc[:3]]
            except Exception as exc:  # noqa: BLE001
                rec["status"] = "error"
                rec["error"] = repr(exc)[:300]
            dev_rows.append(rec)
            print("DEVICE", json.dumps(rec), flush=True)
        dev_rows.append({"model": model_name, "device": "meta", "fit_s": fit_s, "export_s": export_s,
                         "n_train_crops": n, "threshold_all_good": cpu_thr})
    with open(os.path.join(EVAL, "device_check.json"), "w", encoding="utf-8") as fh:
        json.dump(dev_rows, fh, indent=1)
    print("DONE rows", len(rows), "fits", len(fits), flush=True)


if __name__ == "__main__":
    main()
