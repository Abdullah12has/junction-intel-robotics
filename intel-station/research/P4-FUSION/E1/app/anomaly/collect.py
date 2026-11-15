"""Collect train, export and per-device inference results into logs/results.json (P2-B-ANOM-LAPTOP, E1).

Reads models/anomaly/<model>/run/train_summary.json, models/anomaly/<model>/export_summary.json and
logs/runs/infer-<model>-<DEVICE>.json. Per-image items stay in the runs files; the report keeps summaries
and the max absolute score difference of GPU and NPU against CPU (pass 0, same images).
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.project)
    runs = root / "logs" / "runs"
    report = {
        "task": "P2-B-ANOM-LAPTOP",
        "role": "E1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models": {},
    }
    for model in ("padim", "patchcore"):
        mdir = root / "models" / "anomaly" / model
        devices, scores = {}, {}
        for dev in ("GPU", "CPU", "NPU"):
            raw = load(runs / f"infer-{model}-{dev}.json")
            if raw is None:
                devices[dev] = {"status": "missing"}
                continue
            items = raw.pop("items", [])
            devices[dev] = raw
            scores[dev] = {it["file"]: it["score"] for it in items if it["pass"] == 0}
        parity = {}
        ref = scores.get("CPU", {})
        for dev in ("GPU", "NPU"):
            if ref and scores.get(dev):
                diffs = [abs(ref[f] - scores[dev][f]) for f in ref if f in scores[dev]]
                parity[f"{dev}_vs_CPU_max_abs_score_diff"] = max(diffs) if diffs else None
        report["models"][model] = {
            "train": load(mdir / "run" / "train_summary.json"),
            "export": load(mdir / "export_summary.json"),
            "static_batch1_check": load(mdir / "weights" / "openvino_b1" / "static_ir_check.json"),
            "devices": devices,
            "parity": parity,
        }
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
