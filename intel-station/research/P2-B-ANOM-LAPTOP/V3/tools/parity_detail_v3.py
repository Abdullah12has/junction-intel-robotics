"""V3 parity detail (P2-B-ANOM-LAPTOP): prints the per-image PyTorch vs OpenVINO rows from V3/parity.json and label mismatches
of GPU and NPU against CPU over all passes. Run as: <project venv python> -I parity_detail_v3.py. Reads V3 JSON only."""
import json
from pathlib import Path

V3 = Path("/home/intel/intel-junction-2026/work/P2-B-ANOM-LAPTOP/V3")
p = json.loads((V3 / "parity.json").read_text())
for c, e in p["cases"].items():
    print(c, "cpu_max", round(e["max_abs_diff_cpu_vs_torch"], 6), "gpu_max", e["GPU"].get("max_abs_diff_vs_torch"),
          "npu_max", e["NPU"].get("max_abs_diff_vs_torch"), "labels_agree_cpu", e["labels_agree_cpu"])
    for r in e["rows"]:
        print("  ", r["file"], "gt", r["gt"], "torch", round(r["torch_score"], 6), "cpu_ir", round(r["ir_cpu_score"], 6),
              "gpu", r.get("gpu_run_score"), "npu", r.get("npu_run_score"), "d_gpu", r.get("abs_diff_gpu_vs_torch"),
              "d_npu", r.get("abs_diff_npu_vs_torch"), "torch_label", r["torch_label"], "cpu_label", r["ir_cpu_label"],
              "amap_d_cpu", round(r["amap_max_abs_diff_cpu"], 8) if r["amap_max_abs_diff_cpu"] is not None else None)
for n in ("padim-static-b1", "patchcore-b2"):
    cpu = json.loads((V3 / "runs" / f"infer-{n}-CPU.json").read_text())["items"]
    for dev in ("GPU", "NPU"):
        o = json.loads((V3 / "runs" / f"infer-{n}-{dev}.json").read_text())["items"]
        mis = [(a["file"], a["pass"]) for a, b in zip(cpu, o) if a["label"] != b["label"]]
        print(n, dev, "label_mismatch_vs_cpu", len(mis), "of", len(cpu), mis[:5])
