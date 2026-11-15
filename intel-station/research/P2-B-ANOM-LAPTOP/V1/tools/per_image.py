"""V1 (P2-B-ANOM-LAPTOP): per-image difference table from v1-summary.json (stdlib only).

Run as: python3 -I per_image.py SUMMARY_JSON. For each model and device: signed mean difference against torch,
the number of images below torch, and the three images with the largest absolute difference.
"""

import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
for model in ("padim", "patchcore"):
    m = summary["models"][model]
    torch_scores = m["torch"]["scores"]
    for dev in ("CPU", "GPU", "NPU"):
        scores = m["devices"][dev]["scores"]
        diffs = {f: scores[f] - torch_scores[f] for f in torch_scores}
        below = sum(1 for d in diffs.values() if d < 0)
        mean = sum(diffs.values()) / len(diffs)
        print(f"{model} {dev} signed_mean={mean:+.5f} n_below_torch={below}/{len(diffs)}")
        for f in sorted(diffs, key=lambda k: -abs(diffs[k]))[:3]:
            print(f"  {f} torch={torch_scores[f]:.5f} ov={scores[f]:.5f} diff={diffs[f]:+.5f}")
