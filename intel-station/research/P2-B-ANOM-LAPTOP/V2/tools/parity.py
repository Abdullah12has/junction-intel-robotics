"""V2 parity check for P2-B-ANOM-LAPTOP: anomalib PyTorch forward (CPU) vs the exported FP16 IR on CPU, GPU and NPU.

Five test images (3 good, 2 defect) chosen by sorted index. The PyTorch forward applies anomalib's own pre-processor and
post-processor. The IR gets raw RGB/255 NCHW float32, the same input as E1 infer.py. Run with python -I.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_chw(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["padim", "patchcore"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ir", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--devices", default="CPU,GPU,NPU")
    args = ap.parse_args()

    import openvino as ov
    from anomalib.models import Padim, Patchcore

    test = Path(args.data) / "test"
    goods = sorted((test / "good").glob("*.png"))
    defects = sorted((test / "defect").glob("*.png"))
    pairs = [(goods[i], 0) for i in (0, 4, 9)] + [(defects[i], 1) for i in (0, 5)]

    cls = Padim if args.model == "padim" else Patchcore
    torch_model = cls.load_from_checkpoint(args.ckpt, map_location="cpu")
    torch_model.eval()
    ref = []
    with torch.no_grad():
        for path, gt in pairs:
            out = torch_model(torch.from_numpy(load_chw(path))[None])
            ref.append({
                "file": path.name,
                "gt": gt,
                "score": float(out.pred_score.reshape(-1)[0]),
                "label": bool(out.pred_label.reshape(-1)[0]),
                "amap": out.anomaly_map.detach().cpu().numpy().reshape(-1),
            })

    core = ov.Core()
    ir_model = core.read_model(args.ir)
    devices = {}
    for dev in args.devices.split(","):
        compiled = core.compile_model(ir_model, dev)
        ports = list(compiled.outputs)
        by_name = {o.get_any_name(): o for o in ports}
        label_port = by_name["pred_label"]
        amap_port = by_name["anomaly_map"]
        if "pred_score" in by_name:
            score_port = by_name["pred_score"]
        else:
            others = [o for o in ports if o.get_any_name() != label_port.get_any_name() and int(np.prod(o.get_shape())) == 1]
            if len(others) != 1:
                raise RuntimeError(f"cannot pick score output on {dev}")
            score_port = others[0]
        in_name = compiled.input(0).get_any_name()
        rows = []
        for (path, gt), r in zip(pairs, ref):
            batch = np.repeat(load_chw(path)[None], args.batch, axis=0)
            res = compiled({in_name: batch})
            ir_score = float(np.asarray(res[score_port]).reshape(-1)[0])
            ir_label = bool(np.asarray(res[label_port]).reshape(-1)[0])
            ir_amap = np.asarray(res[amap_port]).reshape(args.batch, -1)[0]
            rows.append({
                "file": path.name,
                "gt": gt,
                "torch_score": r["score"],
                "ir_score": ir_score,
                "abs_diff": abs(r["score"] - ir_score),
                "torch_label": r["label"],
                "ir_label": ir_label,
                "amap_max_abs_diff": float(np.max(np.abs(r["amap"] - ir_amap))),
            })
        devices[dev] = {
            "score_port": score_port.get_any_name(),
            "label_port": label_port.get_any_name(),
            "max_abs_score_diff": max(x["abs_diff"] for x in rows),
            "within_tolerance_1e-2": all(x["abs_diff"] <= 1e-2 for x in rows),
            "labels_match": sum(int(x["torch_label"] == x["ir_label"]) for x in rows),
            "max_amap_abs_diff": max(x["amap_max_abs_diff"] for x in rows),
            "rows": rows,
        }

    result = {
        "model": args.model,
        "ckpt": args.ckpt,
        "ir": args.ir,
        "batch": args.batch,
        "pytorch_device": "cpu",
        "files": [p.name for p, _ in pairs],
        "tolerance": 1e-2,
        "devices": devices,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    for dev, d in devices.items():
        print(dev, {k: v for k, v in d.items() if k != "rows"})


if __name__ == "__main__":
    main()
