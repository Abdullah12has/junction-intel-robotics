"""V1 (P2-B-ANOM-LAPTOP): torch-side scores for the 20 synthetic test images on CPU, anomalib 2.7.0 predict path.

Loads the E1 training checkpoint and runs anomalib's own predict path (pre-processing, model, post-processing) on the
Folder test set. val_split_mode is "none" so that the test set is all 20 images: with the Folder default (from_test,
ratio 0.5, seed None) anomalib splits the test folder at random and predicts on about 10 images only.
Writes one row per image: file, gt (0 good, 1 defect), pred_score, pred_label, and the image AUROC.
Run as: <project env python> -I torch_predict.py --model M --ckpt CKPT --data DATA --out OUT
"""

import argparse
import json
import traceback
from pathlib import Path


def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def field(obj, name):
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["padim", "patchcore"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {"model": args.model, "ckpt": args.ckpt, "status": "started"}
    try:
        import anomalib
        import torch
        from anomalib.data import Folder
        from anomalib.engine import Engine
        from anomalib.models import Padim, Patchcore

        result["anomalib_version"] = anomalib.__version__
        result["torch_version"] = torch.__version__
        model_cls = Padim if args.model == "padim" else Patchcore
        model = model_cls.load_from_checkpoint(args.ckpt, map_location="cpu")
        datamodule = Folder(
            name="synthetic",
            root=args.data,
            normal_dir="train/good",
            normal_test_dir="test/good",
            abnormal_dir="test/defect",
            train_batch_size=1,
            eval_batch_size=1,
            val_split_mode="none",
        )
        engine = Engine(accelerator="cpu", devices=1, default_root_dir=str(out.parent / "engine"))
        batches = engine.predict(model=model, datamodule=datamodule)
        rows = []
        for batch in batches:
            paths = [str(p) for p in field(batch, "image_path")]
            scores = field(batch, "pred_score").reshape(-1).tolist()
            labels = field(batch, "pred_label")
            labels = labels.reshape(-1).tolist() if labels is not None else [None] * len(paths)
            for path, score, label in zip(paths, scores, labels):
                rows.append({
                    "file": Path(path).name,
                    "gt": 1 if Path(path).parent.name == "defect" else 0,
                    "pred_score": float(score),
                    "pred_label": None if label is None else bool(label),
                })
        rows.sort(key=lambda r: (r["gt"], r["file"]))
        result.update({
            "status": "ok",
            "n_images": len(rows),
            "auroc": auroc([r["pred_score"] for r in rows], [r["gt"] for r in rows]),
            "items": rows,
        })
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback_tail"] = traceback.format_exc().splitlines()[-8:]
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "items"}, indent=2))


if __name__ == "__main__":
    main()
