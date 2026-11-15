"""V1 (P2-B-ANOM-LAPTOP): summarise V1's infer.py runs and torch CPU scores into one JSON (stdlib only).

Per model and device: status, AUROC and label accuracy over the 20 test images (pass 0), latency median and p95 over
all timed calls (recomputed from items and checked against the reported summary), max and mean absolute score
difference vs CPU and vs torch. Torch-vs-OpenVINO parity is reported for the 5 parity images.
Run as: python3 -I compare.py --runs DIR --data DATA --out FILE
"""

import argparse
import json
import statistics
from pathlib import Path

DEVICES = ["CPU", "GPU", "NPU"]
MODELS = ["padim", "patchcore"]
PARITY_IDX = [0, 4, 9, 10, 15]


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


def percentile(values, q):
    ordered = sorted(values)
    k = (len(ordered) - 1) * q / 100.0
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs = Path(args.runs)
    test = Path(args.data) / "test"
    good = sorted(p.name for p in (test / "good").glob("*.png"))
    defect = sorted(p.name for p in (test / "defect").glob("*.png"))
    order = good + defect
    gt = {name: 0 for name in good}
    gt.update({name: 1 for name in defect})
    parity_files = [order[i] for i in PARITY_IDX]

    summary = {"role": "V1", "task": "P2-B-ANOM-LAPTOP", "n_images": len(order),
               "parity_files": parity_files, "models": {}}
    for model in MODELS:
        torch_doc = json.loads((runs / f"torch-{model}.json").read_text())
        titems = torch_doc.get("items", [])
        torch_scores = {it["file"]: it["pred_score"] for it in titems}
        torch_labels = {it["file"]: it["pred_label"] for it in titems}
        t_auroc = None
        t_acc = None
        if len(torch_scores) == len(order):
            t_auroc = auroc([torch_scores[f] for f in order], [gt[f] for f in order])
            if all(torch_labels.get(f) is not None for f in order):
                t_acc = sum((bool(torch_labels[f]) == bool(gt[f])) for f in order) / len(order)
        entry = {
            "torch": {
                "status": torch_doc.get("status"),
                "error": torch_doc.get("error"),
                "anomalib_version": torch_doc.get("anomalib_version"),
                "torch_version": torch_doc.get("torch_version"),
                "n_images": len(torch_scores),
                "auroc_20": t_auroc,
                "label_accuracy_20": t_acc,
                "scores": torch_scores,
            },
            "devices": {},
            "parity_5": {},
        }
        device_scores = {}
        for dev in DEVICES:
            doc = json.loads((runs / f"infer-{model}-{dev}.json").read_text())
            info = {k: doc.get(k) for k in (
                "status", "error", "anomalib_version", "openvino_version", "ir_input_shape", "compile_ms",
                "first_infer_ms", "output_names", "score_port_name", "label_port_name", "n_images", "n_timed",
                "auroc", "label_accuracy", "latency_ms")}
            if doc.get("status") == "ok":
                items = doc["items"]
                first = {it["file"]: it for it in items if it["pass"] == 0}
                lats = [it["latency_ms"] for it in items]
                scores = {f: first[f]["score"] for f in order}
                device_scores[dev] = scores
                info["scores"] = scores
                info["auroc_recomputed"] = auroc([scores[f] for f in order], [gt[f] for f in order])
                info["label_accuracy_recomputed"] = sum(
                    ((first[f]["label"] == "defective") == bool(gt[f])) for f in order) / len(order)
                info["latency_recomputed"] = {"median": round(statistics.median(lats), 3),
                                              "p95": round(percentile(lats, 95), 3), "n": len(lats)}
            entry["devices"][dev] = info

        for dev, scores in device_scores.items():
            info = entry["devices"][dev]
            if "CPU" in device_scores:
                diffs = [abs(scores[f] - device_scores["CPU"][f]) for f in order]
                info["max_abs_diff_vs_CPU"] = max(diffs)
                info["mean_abs_diff_vs_CPU"] = sum(diffs) / len(diffs)
            tdiffs = [abs(scores[f] - torch_scores[f]) for f in order if f in torch_scores]
            info["max_abs_diff_vs_torch_20"] = max(tdiffs) if tdiffs else None
            entry["parity_5"][dev] = [
                {"file": f, "torch": torch_scores.get(f), "ov": scores[f],
                 "abs_diff": abs(scores[f] - torch_scores[f]) if f in torch_scores else None}
                for f in parity_files
            ]
        summary["models"][model] = entry

    Path(args.out).write_text(json.dumps(summary, indent=2))
    for model in MODELS:
        e = summary["models"][model]
        t = e["torch"]
        print(f"{model} torch status={t['status']} auroc_20={t['auroc_20']} label_acc_20={t['label_accuracy_20']}")
        for dev in DEVICES:
            i = e["devices"][dev]
            if i.get("status") != "ok":
                print(f"{model} {dev} status={i.get('status')} error={i.get('error')}")
                continue
            print(f"{model} {dev} status=ok auroc={i['auroc_recomputed']} acc={i['label_accuracy_recomputed']} "
                  f"med={i['latency_recomputed']['median']} p95={i['latency_recomputed']['p95']} "
                  f"rep_med={i['latency_ms']['median']} rep_p95={i['latency_ms']['p95']} "
                  f"max_vs_CPU={i.get('max_abs_diff_vs_CPU')} max_vs_torch20={i.get('max_abs_diff_vs_torch_20')}")
        for dev in DEVICES:
            rows = e["parity_5"].get(dev, [])
            print(f"{model} parity5 {dev} " + " ".join(
                f"{r['file']}:torch={r['torch']}:ov={r['ov']}:diff={r['abs_diff']}" for r in rows))


if __name__ == "__main__":
    main()
