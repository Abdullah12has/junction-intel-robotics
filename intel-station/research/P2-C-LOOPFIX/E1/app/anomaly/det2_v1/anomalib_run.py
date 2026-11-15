"""Train and evaluate an anomalib model (PatchCore or PaDiM) on E2 close-view brick crops; export OpenVINO IR.

Uses the laptop env envs/anomalib-env (ours; READY 14:43Z). Backbone weights come from the local torch cache
(TORCH_HOME inside the repo). Good-only training; test = episode-disjoint good (test/good) vs taped (test/defect).

usage: python3 -I anomalib_run.py --model patchcore|padim --mode plain|aug --accel xpu|cpu --out DIR
  mode aug adds hue-shifted and desaturated copies of the training goods (colour coverage of normal bricks).
"""
import argparse
import glob
import json
import os
import shutil
import time

os.environ.setdefault("TORCH_HOME", "/home/intel/intel-junction-2026/.cache/torch")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

WORK = "/home/intel/intel-junction-2026/work/P2-B-DET2/E2"
SRC = os.path.join(WORK, "anomalib", "bricks_close")


def build_aug(root_src, root_dst):
    if os.path.exists(root_dst):
        shutil.rmtree(root_dst)
    os.makedirs(os.path.join(root_dst, "train/good"))
    shutil.copytree(os.path.join(root_src, "test"), os.path.join(root_dst, "test"))
    n = 0
    for p in sorted(glob.glob(os.path.join(root_src, "train/good/*.png"))):
        img = cv2.imread(p)
        stem = os.path.splitext(os.path.basename(p))[0]
        cv2.imwrite(os.path.join(root_dst, "train/good", stem + ".png"), img)
        n += 1
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        for shift in (25, 50, 75, 100, 125, 150):
            h2 = hsv.copy()
            h2[..., 0] = (h2[..., 0] + shift) % 180
            cv2.imwrite(os.path.join(root_dst, "train/good", f"{stem}_h{shift}.png"),
                        cv2.cvtColor(h2.astype(np.uint8), cv2.COLOR_HSV2BGR))
            n += 1
        h3 = hsv.copy()
        h3[..., 1] = (h3[..., 1] * 0.2).astype(np.int16)  # desaturated, white-ish variant
        h3[..., 2] = np.clip(h3[..., 2] * 1.1, 0, 255).astype(np.int16)
        cv2.imwrite(os.path.join(root_dst, "train/good", f"{stem}_desat.png"),
                    cv2.cvtColor(h3.astype(np.uint8), cv2.COLOR_HSV2BGR))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="patchcore")
    ap.add_argument("--mode", default="plain")
    ap.add_argument("--accel", default="xpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    a = ap.parse_args()
    from anomalib.data import Folder
    from anomalib.deploy import ExportType
    from anomalib.engine import Engine
    from anomalib.models import Padim, Patchcore

    os.makedirs(a.out, exist_ok=True)
    root = SRC
    if a.mode == "aug":
        root = os.path.join(WORK, "anomalib", "bricks_close_aug")
        n_aug = build_aug(SRC, root)
    else:
        n_aug = len(glob.glob(os.path.join(SRC, "train/good/*.png")))
    dm = Folder(name="bricks_close", root=root, normal_dir="train/good", abnormal_dir="test/defect",
                normal_test_dir="test/good", val_split_mode="same_as_test", train_batch_size=4, eval_batch_size=4, num_workers=0)
    if a.model == "patchcore":
        model = Patchcore(backbone="wide_resnet50_2", layers=("layer2", "layer3"), pre_trained=True,
                          coreset_sampling_ratio=0.25, num_neighbors=9)
    else:
        model = Padim(backbone="resnet18", layers=("layer1", "layer2", "layer3"), pre_trained=True)
    accel, devs = a.accel, 1
    if a.accel == "xpu":
        from anomalib.engine.accelerator.xpu import XPUAccelerator
        accel, devs = XPUAccelerator(), [0]
    eng = Engine(accelerator=accel, devices=devs, default_root_dir=a.out)
    t0 = time.time()
    eng.fit(datamodule=dm, model=model)
    train_s = time.time() - t0
    test = eng.test(datamodule=dm, model=model)
    metrics = {k: float(v) for k, v in (test[0] if test else {}).items() if isinstance(v, (int, float, np.floating))}
    preds = eng.predict(datamodule=dm, model=model)
    rows = []
    for batch in preds:
        scores = batch.pred_score.detach().cpu().numpy().reshape(-1)
        paths = batch.image_path
        for p, s in zip(paths, scores):
            rows.append({"image": os.path.basename(p), "score": float(s)})
    exp_dir = os.path.join(a.out, "export_ov")
    exp_ok, exp_err = True, None
    try:
        eng.export(model=model, export_type=ExportType.OPENVINO, export_root=exp_dir, input_size=(256, 256))
    except Exception as e:  # keep the run honest: record the failure
        exp_ok, exp_err = False, f"{type(e).__name__}: {str(e)[:300]}"
    res = {"model": a.model, "mode": a.mode, "accel": a.accel, "train_images": n_aug, "train_seconds": round(train_s, 1),
           "metrics": metrics, "per_image": rows, "export_ok": exp_ok, "export_error": exp_err}
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_image"})[:1500])
    print("per_image", len(rows))


if __name__ == "__main__":
    main()
