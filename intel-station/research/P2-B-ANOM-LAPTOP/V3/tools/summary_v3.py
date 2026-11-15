"""V3 short summary (P2-B-ANOM-LAPTOP): largest GPU differences per model from perfile.json, and the repeat runs.
Run as: <project venv python> -I summary_v3.py from the V3 folder. Reads V3 JSON only, prints a compact summary."""
import json
from pathlib import Path

V3 = Path("/home/intel/intel-junction-2026/work/P2-B-ANOM-LAPTOP/V3")
per = json.loads((V3 / "perfile.json").read_text())
for k in per:
    top = sorted(per[k]["rows"], key=lambda r: -r["d_gpu"])[:2]
    for r in top:
        print("top_gpu", k, r["file"], "gt", r["gt"], "cpu", r["cpu"], "gpu", r["gpu"], "d_gpu", r["d_gpu"], "cpu_label", r["cpu_label"])
for n in ("rep-padim-static-b1-CPU", "rep-patchcore-b2-CPU", "rep-padim-static-b1-GPU", "rep-patchcore-b2-GPU", "rep-padim-static-b1-NPU", "rep-patchcore-b2-NPU"):
    j = json.loads((V3 / "runs" / f"infer-{n}.json").read_text())
    lat = j["latency_ms"]
    print("repeat", n, "status", j["status"], "auroc", j["auroc"], "acc", j["label_accuracy"], "compile_ms", j["compile_ms"],
          "first_ms", j["first_infer_ms"], "median", lat["median"], "p95", lat["p95"])
