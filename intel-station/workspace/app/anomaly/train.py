"""Train PaDiM or PatchCore (anomalib 2.7.0) on the Folder dataset, on XPU or CPU (P2-B-ANOM-LAPTOP, E1).

Checkpoints go under --out (Engine default_root_dir). <out>/train_summary.json records the accelerator used,
exact versions, timings, the engine.test metrics and the checkpoint paths. Run with the project env python.
"""

import argparse
import json
import time
from pathlib import Path


def plain(obj):
    import torch

    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def count_pngs(folder):
    return len(list(Path(folder).glob("*.png")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Folder dataset root with train/good, test/good, test/defect")
    ap.add_argument("--name", default="synthetic")
    ap.add_argument("--model", choices=["padim", "patchcore"], required=True)
    ap.add_argument("--accelerator", choices=["xpu", "cpu"], required=True)
    ap.add_argument("--out", required=True, help="run directory for checkpoints and train_summary.json")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--fallback_reason", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import anomalib
    import torch
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import Padim, Patchcore

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from lightning.pytorch import seed_everything

    seed_everything(args.seed, workers=True)
    if args.accelerator == "xpu" and not torch.xpu.is_available():
        raise SystemExit("torch.xpu.is_available() is False; use --accelerator cpu")

    datamodule = Folder(
        name=args.name,
        root=str(data),
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir="test/defect",
        train_batch_size=args.batch,
        eval_batch_size=args.batch,
    )
    model = Padim() if args.model == "padim" else Patchcore()

    if args.accelerator == "xpu":
        from anomalib.engine import SingleXPUStrategy, XPUAccelerator

        engine = Engine(strategy=SingleXPUStrategy(), accelerator=XPUAccelerator(), default_root_dir=out)
    else:
        engine = Engine(accelerator="cpu", devices=1, default_root_dir=out)

    t0 = time.perf_counter()
    engine.train(model=model, datamodule=datamodule)
    train_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    test_results = engine.test(model=model, datamodule=datamodule)
    test_s = time.perf_counter() - t1

    ckpts = sorted(out.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    summary = {
        "model": args.model,
        "seed": args.seed,
        "accelerator_used": args.accelerator,
        "fallback_reason": args.fallback_reason,
        "anomalib_version": anomalib.__version__,
        "torch_version": torch.__version__,
        "xpu_available": bool(torch.xpu.is_available()),
        "xpu_device_name": torch.xpu.get_device_name(0) if args.accelerator == "xpu" else None,
        "dataset": {
            "root": str(data),
            "train_good": count_pngs(data / "train" / "good"),
            "test_good": count_pngs(data / "test" / "good"),
            "test_defect": count_pngs(data / "test" / "defect"),
        },
        "train_seconds": round(train_s, 2),
        "test_seconds": round(test_s, 2),
        "test_metrics": plain(test_results),
        "checkpoints": [str(p) for p in ckpts],
        "best_checkpoint": str(ckpts[-1]) if ckpts else None,
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
