"""Run an exported OpenVINO IR on every test image on ONE named device (P2-B-ANOM-LAPTOP, E1).

Raw OpenVINO inference: ov.Core().compile_model(IR, device). Pre-processing matches anomalib's OpenVINOInferencer
(RGB scaled to [0,1], NCHW float32, no resize). The image is repeated --batch times because the IR has a static batch
(1 for the static PaDiM copy, 2 for the PatchCore export). Sample 0 of pred_score and pred_label is the result.
Per image: {file, gt (0 good, 1 defect), pass, score, label, latency_ms}. Summary: image AUROC (pass 0), label
accuracy, latency median and p95 over all timed passes, compile_ms and first_infer_ms.
A device that fails writes status "error" and exits 0 so the runner continues.
"""

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ir", required=True, help="path to the OpenVINO model.xml")
    ap.add_argument("--batch", type=int, required=True, help="static batch size of the IR (1 or 2)")
    ap.add_argument("--device", required=True, choices=["CPU", "GPU", "NPU"])
    ap.add_argument("--data", required=True, help="Folder dataset root with test/good and test/defect")
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {"model": args.model, "device": args.device, "ir": args.ir, "batch": args.batch, "status": "started"}
    try:
        import anomalib
        import openvino as ov

        result["anomalib_version"] = anomalib.__version__
        result["openvino_version"] = ov.__version__

        test_root = Path(args.data) / "test"
        pairs = [(p, 0) for p in sorted((test_root / "good").glob("*.png"))]
        pairs += [(p, 1) for p in sorted((test_root / "defect").glob("*.png"))]
        images = []
        for p, gt in pairs:
            chw = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
            images.append((p.name, gt, np.repeat(chw[None], args.batch, axis=0)))

        core = ov.Core()
        model = core.read_model(args.ir)
        result["ir_input_shape"] = str(model.input(0).get_partial_shape())
        t0 = time.perf_counter()
        compiled = core.compile_model(model, args.device)
        result["compile_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        ports = list(compiled.outputs)
        result["output_names"] = [o.get_any_name() for o in ports]
        by_name = {o.get_any_name(): o for o in ports}
        if "pred_score" in by_name and "pred_label" in by_name:
            score_port, label_port = by_name["pred_score"], by_name["pred_label"]
        else:
            # The CPU plugin renames the score output on the batch-1 IR (clamp_1 in attempt 2). Take the label by
            # name and the score as the only other one-element output, so the choice does not depend on port order.
            label_port = by_name["pred_label"]
            cands = [o for o in ports if o.get_any_name() != label_port.get_any_name() and int(np.prod(o.get_shape())) == 1]
            if len(cands) != 1:
                raise RuntimeError(f"cannot pick score output among {[o.get_any_name() for o in cands]}")
            score_port = cands[0]
        result["score_port_name"] = score_port.get_any_name()
        result["label_port_name"] = label_port.get_any_name()
        in_name = compiled.input(0).get_any_name()

        def run(batch):
            res = compiled({in_name: batch})
            score = float(np.asarray(res[score_port]).reshape(-1)[0])
            label = bool(np.asarray(res[label_port]).reshape(-1)[0])
            return score, label

        t0 = time.perf_counter()
        run(images[0][2])
        result["first_infer_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        for _, _, batch in images[: args.warmup]:
            run(batch)

        items = []
        for rep in range(args.repeats):
            for name, gt, batch in images:
                t = time.perf_counter()
                score, label = run(batch)
                lat = (time.perf_counter() - t) * 1000
                items.append({
                    "file": name,
                    "gt": gt,
                    "pass": rep,
                    "score": score,
                    "label": "defective" if label else "good",
                    "latency_ms": round(lat, 3),
                })

        first = [it for it in items if it["pass"] == 0]
        accuracy = sum(int((it["label"] == "defective") == bool(it["gt"])) for it in first) / len(first)
        lats = np.array([it["latency_ms"] for it in items])
        result.update({
            "status": "ok",
            "n_images": len(images),
            "n_good": sum(1 for _, gt, _ in images if gt == 0),
            "n_defect": sum(1 for _, gt, _ in images if gt == 1),
            "n_timed": len(items),
            "warmup": args.warmup,
            "auroc": auroc([it["score"] for it in first], [it["gt"] for it in first]),
            "label_accuracy": accuracy,
            "latency_ms": {
                "median": round(float(np.median(lats)), 3),
                "p95": round(float(np.percentile(lats, 95)), 3),
                "mean": round(float(lats.mean()), 3),
                "min": round(float(lats.min()), 3),
                "max": round(float(lats.max()), 3),
            },
            "items": items,
        })
    except Exception as exc:  # record the failure; the runner continues with the next device
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback_tail"] = traceback.format_exc().splitlines()[-8:]
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "items"}, indent=2))


if __name__ == "__main__":
    main()
