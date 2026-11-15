"""V3 analysis (P2-B-ANOM-LAPTOP): recompute AUROC, label accuracy, latency median and p95 from the V3 infer runs (all 100 timed
calls), compare with E1 logs/results.json, and compute GPU/NPU vs CPU pass-0 score differences. Writes V3/analysis.json.
Run as: <project venv python> -I analyze_v3.py. Reads only the V3 run JSON and the E1 results copy in the project logs folder."""
import hashlib
import json
from pathlib import Path

import numpy as np

PROJ = Path("/home/intel/intel-junction-2026")
V3 = PROJ / "work/P2-B-ANOM-LAPTOP/V3"
RUNS = V3 / "runs"
E1_PATH = PROJ / "logs/results.json"
DEVICES = ["CPU", "GPU", "NPU"]
SETS = {
    "padim_static_b1": ("padim-static-b1", "padim"),
    "padim_dyn_b1": ("padim-dyn-b1", None),
    "patchcore_b2": ("patchcore-b2", "patchcore"),
}


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
    e1 = json.loads(E1_PATH.read_text())
    out = {"e1_results_sha256_laptop_copy": hashlib.sha256(E1_PATH.read_bytes()).hexdigest(), "sets": {}}
    for key, (run_name, e1_key) in SETS.items():
        out["sets"][key] = {}
        scores = {}
        for dev in DEVICES:
            p = RUNS / f"infer-{run_name}-{dev}.json"
            if not p.exists():
                out["sets"][key][dev] = {"status": "missing"}
                continue
            d = json.loads(p.read_text())
            if d.get("status") != "ok":
                out["sets"][key][dev] = {"status": d.get("status"), "error": (d.get("error") or "")[:300]}
                continue
            items = d["items"]
            first = [it for it in items if it["pass"] == 0]
            lats = np.array([it["latency_ms"] for it in items])
            acc = sum(int((it["label"] == "defective") == bool(it["gt"])) for it in first) / len(first)
            entry = {
                "status": "ok",
                "ir_input_shape": d["ir_input_shape"],
                "score_port": d["score_port_name"],
                "compile_ms": d["compile_ms"],
                "first_infer_ms": d["first_infer_ms"],
                "n_images": d["n_images"],
                "n_timed": len(items),
                "auroc_recomputed": auroc([it["score"] for it in first], [it["gt"] for it in first]),
                "auroc_infer_summary": d["auroc"],
                "label_accuracy": acc,
                "median_ms": round(float(np.median(lats)), 3),
                "p95_ms": round(float(np.percentile(lats, 95)), 3),
                "mean_ms": round(float(lats.mean()), 3),
                "min_ms": round(float(lats.min()), 3),
                "max_ms": round(float(lats.max()), 3),
                "defect_000_pass0_score": next((it["score"] for it in first if it["file"] == "defect_000.png"), None),
            }
            if e1_key:
                ed = e1["models"][e1_key]["devices"][dev]
                entry["e1_claimed"] = {
                    "auroc": ed["auroc"], "label_accuracy": ed["label_accuracy"],
                    "median_ms": ed["latency_ms"]["median"], "p95_ms": ed["latency_ms"]["p95"],
                }
            scores[dev] = {it["file"]: it["score"] for it in first}
            out["sets"][key][dev] = entry
        if "CPU" in scores:
            for dev in ("GPU", "NPU"):
                if dev in scores:
                    diffs = [abs(scores["CPU"][f] - scores[dev][f]) for f in scores["CPU"]]
                    out["sets"][key][dev]["max_abs_diff_vs_cpu"] = max(diffs)
                    if e1_key:
                        out["sets"][key][dev]["e1_claimed_max_abs_diff_vs_cpu"] = e1["models"][e1_key]["parity"][f"{dev}_vs_CPU_max_abs_score_diff"]
        if key == "padim_dyn_b1" and "CPU" in scores and "CPU" in out["sets"]["padim_static_b1"] and "scores" not in out["sets"]["padim_static_b1"]["CPU"]:
            pass
        if key == "padim_dyn_b1" and "CPU" in scores:
            st = json.loads((RUNS / "infer-padim-static-b1-CPU.json").read_text())
            st_scores = {it["file"]: it["score"] for it in st["items"] if it["pass"] == 0}
            out["sets"][key]["CPU"]["max_abs_diff_vs_static_b1_cpu"] = max(abs(st_scores[f] - scores["CPU"][f]) for f in scores["CPU"])
    print(json.dumps(out, indent=1))
    (V3 / "analysis.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
