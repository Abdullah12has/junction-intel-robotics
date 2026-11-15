"""V3 per-file differences (P2-B-ANOM-LAPTOP): CPU vs GPU vs NPU pass-0 scores for all 20 test files, read from the V3 infer
run JSON. Writes V3/perfile.json and prints the largest differences. Run as: <project venv python> -I perfile_v3.py."""
import json
from pathlib import Path

V3 = Path("/home/intel/intel-junction-2026/work/P2-B-ANOM-LAPTOP/V3")
RUNS = V3 / "runs"
out = {}
for name in ("padim-static-b1", "patchcore-b2"):
    per = {}
    for dev in ("CPU", "GPU", "NPU"):
        d = json.loads((RUNS / f"infer-{name}-{dev}.json").read_text())
        per[dev] = {it["file"]: (it["score"], it["gt"], it["label"]) for it in d["items"] if it["pass"] == 0}
    rows = []
    for f in sorted(per["CPU"]):
        c, g, n = per["CPU"][f], per["GPU"][f], per["NPU"][f]
        rows.append({
            "file": f, "gt": c[1], "cpu": round(c[0], 6), "gpu": round(g[0], 6), "npu": round(n[0], 6),
            "cpu_label": c[2], "npu_label": n[2],
            "d_gpu": round(abs(c[0] - g[0]), 6), "d_npu": round(abs(c[0] - n[0]), 6),
        })
    top = sorted(rows, key=lambda r: -r["d_npu"])[:3]
    out[name] = {
        "n_files": len(rows),
        "max_d_gpu": max(r["d_gpu"] for r in rows),
        "max_d_npu": max(r["d_npu"] for r in rows),
        "top3_npu": top,
        "rows": rows,
    }
    print(name, "n", len(rows), "max_d_gpu", out[name]["max_d_gpu"], "max_d_npu", out[name]["max_d_npu"])
    for r in top:
        print("  ", r["file"], "gt", r["gt"], "cpu", r["cpu"], "gpu", r["gpu"], "npu", r["npu"], "d_npu", r["d_npu"], "cpu_label", r["cpu_label"], "npu_label", r["npu_label"])
(V3 / "perfile.json").write_text(json.dumps(out, indent=2))
