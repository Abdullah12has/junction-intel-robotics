"""V2 summary of E1 infer.py outputs for P2-B-ANOM-LAPTOP. Run with python -I.

Recomputes image AUROC and label accuracy from the per-image items (pass 0), latency median and p95 over all timed
passes, and the largest pass-0 score difference of GPU and NPU against CPU. Prints a compact JSON summary.
"""

import json
import sys
from pathlib import Path

import numpy as np


def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def main():
    run_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    summary = {}
    for model in ("padim", "patchcore"):
        devs, pass0 = {}, {}
        for dev in ("CPU", "GPU", "NPU"):
            d = json.loads((run_dir / f"infer-{model}-{dev}.json").read_text())
            if d.get("status") != "ok":
                devs[dev] = {"status": d.get("status"), "error": d.get("error")}
                continue
            items = d["items"]
            first = [it for it in items if it["pass"] == 0]
            lats = np.array([it["latency_ms"] for it in items])
            pass0[dev] = {it["file"]: (it["score"], it["label"]) for it in first}
            devs[dev] = {
                "status": "ok",
                "batch": d["batch"],
                "score_port": d["score_port_name"],
                "compile_ms": d["compile_ms"],
                "first_infer_ms": d["first_infer_ms"],
                "n_images": d["n_images"],
                "n_timed": len(items),
                "auroc_recomputed": round(auroc([it["score"] for it in first], [it["gt"] for it in first]), 4),
                "auroc_from_infer_json": d["auroc"],
                "label_accuracy": round(sum(int((it["label"] == "defective") == bool(it["gt"])) for it in first) / len(first), 4),
                "latency_median_ms": round(float(np.median(lats)), 3),
                "latency_p95_ms": round(float(np.percentile(lats, 95)), 3),
                "latency_min_ms": round(float(lats.min()), 3),
                "latency_max_ms": round(float(lats.max()), 3),
            }
        cross = {}
        cpu = pass0.get("CPU")
        for dev in ("GPU", "NPU"):
            if cpu and dev in pass0:
                diffs = {f: abs(pass0[dev][f][0] - cpu[f][0]) for f in cpu}
                worst = max(diffs, key=diffs.get)
                cross[f"{dev}_vs_CPU"] = {
                    "max_abs_score_diff": round(diffs[worst], 6),
                    "file": worst,
                    "label_flips": sum(int(pass0[dev][f][1] != cpu[f][1]) for f in cpu),
                }
        summary[model] = {"devices": devs, "cross_device": cross}
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
