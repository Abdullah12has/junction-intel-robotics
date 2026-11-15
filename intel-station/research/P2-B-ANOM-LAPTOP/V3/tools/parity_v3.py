"""V3 parity check (P2-B-ANOM-LAPTOP): PyTorch anomalib 2.7.0 checkpoint on CPU vs the exported OpenVINO FP16 IR on CPU,
on 5 synthetic test images (2 good, 3 defect). GPU and NPU pass-0 scores for the same files are read from the V3 infer runs.
Run as: <project venv python> -I parity_v3.py. Writes V3/parity.json. Reads E1 checkpoints, IRs, images and V3 run JSON only."""
import json
from pathlib import Path

import anomalib
import numpy as np
import openvino as ov
import torch
from anomalib.models import Padim, Patchcore
from PIL import Image

PROJ = Path("/home/intel/intel-junction-2026")
DATA = PROJ / "data/anomaly/synthetic"
V3 = PROJ / "work/P2-B-ANOM-LAPTOP/V3"
RUNS = V3 / "runs"
TOL = 1e-2
PADIM_CKPT = PROJ / "models/anomaly/padim/run/Padim/synthetic/v0/weights/lightning/model.ckpt"
PATCH_CKPT = PROJ / "models/anomaly/patchcore/run/Patchcore/synthetic/v0/weights/lightning/model.ckpt"
PADIM_B1 = PROJ / "models/anomaly/padim/weights/openvino_b1/model.xml"
PADIM_DYN = PROJ / "models/anomaly/padim/weights/openvino/model.xml"
PATCH_B2 = PROJ / "models/anomaly/patchcore/weights/openvino/model.xml"
CKPT = {"padim": PADIM_CKPT, "patchcore": PATCH_CKPT}
CLS = {"padim": Padim, "patchcore": Patchcore}
CASES = [
    ("padim_static_b1", "padim", PADIM_B1, 1, "padim-static-b1"),
    ("padim_dyn_b1", "padim", PADIM_DYN, 1, "padim-dyn-b1"),
    ("patchcore_b2", "patchcore", PATCH_B2, 2, "patchcore-b2"),
]


def images():
    good = sorted((DATA / "test" / "good").glob("*.png"))[:2]
    defect = sorted((DATA / "test" / "defect").glob("*.png"))[:3]
    return [(p, 0) for p in good] + [(p, 1) for p in defect]


def to_chw(path):
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None])


def torch_side(mname, imgs):
    model = CLS[mname].load_from_checkpoint(str(CKPT[mname]), map_location="cpu")
    model.eval()
    rows = {}
    with torch.no_grad():
        for path, _ in imgs:
            out = model(torch.from_numpy(to_chw(path)))
            pl, am = out.pred_label, out.anomaly_map
            rows[path.name] = {
                "score": float(out.pred_score.reshape(-1)[0]),
                "label": bool(pl.reshape(-1)[0]) if pl is not None else None,
                "amap": am.reshape(-1).cpu().numpy() if am is not None else None,
            }
    return rows


def ir_side(core, xml, batch, imgs):
    compiled = core.compile_model(core.read_model(str(xml)), "CPU")
    outs = list(compiled.outputs)
    by = {o.get_any_name(): o for o in outs}
    label_p = by["pred_label"]
    if "pred_score" in by:
        score_p = by["pred_score"]
    else:
        cands = [o for o in outs if o.get_any_name() != "pred_label" and int(np.prod(o.get_shape())) == 1]
        if len(cands) != 1:
            raise RuntimeError(f"score output not unique: {[o.get_any_name() for o in cands]}")
        score_p = cands[0]
    amap_p = by.get("anomaly_map")
    in_name = compiled.input(0).get_any_name()
    rows = {}
    for path, _ in imgs:
        res = compiled({in_name: np.repeat(to_chw(path), batch, axis=0)})
        rows[path.name] = {
            "score": float(np.asarray(res[score_p]).reshape(-1)[0]),
            "label": bool(np.asarray(res[label_p]).reshape(-1)[0]),
            "amap": np.asarray(res[amap_p])[0].reshape(-1) if amap_p is not None else None,
        }
    return rows, score_p.get_any_name()


def run_side(run_name, dev, names):
    p = RUNS / f"infer-{run_name}-{dev}.json"
    if not p.exists():
        return {"status": "missing"}
    d = json.loads(p.read_text())
    if d.get("status") != "ok":
        return {"status": d.get("status"), "error": d.get("error", "")}
    return {"status": "ok", "scores": {it["file"]: it["score"] for it in d["items"] if it["pass"] == 0 and it["file"] in names}}


def main():
    imgs = images()
    names = [p.name for p, _ in imgs]
    core = ov.Core()
    torch_rows = {m: torch_side(m, imgs) for m in CKPT}
    report = {
        "tolerance_abs": TOL,
        "versions": {"anomalib": anomalib.__version__, "torch": torch.__version__, "openvino": ov.__version__},
        "files": names,
        "gt": [g for _, g in imgs],
        "cases": {},
    }
    ir_by_case = {}
    for case, mname, xml, batch, run_name in CASES:
        ir_rows, score_port = ir_side(core, xml, batch, imgs)
        ir_by_case[case] = ir_rows
        tr = torch_rows[mname]
        run_info = {dev: run_side(run_name, dev, set(names)) for dev in ("GPU", "NPU")}
        rows = []
        for path, gt in imgs:
            n = path.name
            t, c = tr[n], ir_rows[n]
            row = {
                "file": n,
                "gt": gt,
                "torch_score": t["score"],
                "ir_cpu_score": c["score"],
                "abs_diff_cpu": abs(t["score"] - c["score"]),
                "torch_label": t["label"],
                "ir_cpu_label": c["label"],
                "amap_max_abs_diff_cpu": float(np.max(np.abs(t["amap"] - c["amap"]))) if t["amap"] is not None else None,
            }
            for dev in ("GPU", "NPU"):
                ri = run_info[dev]
                if ri["status"] == "ok" and n in ri["scores"]:
                    row[f"{dev.lower()}_run_score"] = ri["scores"][n]
                    row[f"abs_diff_{dev.lower()}_vs_torch"] = abs(t["score"] - ri["scores"][n])
            rows.append(row)
        diffs = [r["abs_diff_cpu"] for r in rows]
        entry = {
            "ir": str(xml),
            "batch": batch,
            "score_output_on_cpu": score_port,
            "max_abs_diff_cpu_vs_torch": max(diffs),
            "all_within_tol_cpu": bool(max(diffs) <= TOL),
            "labels_agree_cpu": all(r["torch_label"] == r["ir_cpu_label"] for r in rows),
            "rows": rows,
        }
        for dev in ("GPU", "NPU"):
            ri = run_info[dev]
            if ri["status"] != "ok":
                entry[dev] = {"status": ri["status"], "error": ri.get("error", "")}
            else:
                dd = [r[f"abs_diff_{dev.lower()}_vs_torch"] for r in rows if f"abs_diff_{dev.lower()}_vs_torch" in r]
                entry[dev] = {"status": "ok", "max_abs_diff_vs_torch": max(dd) if dd else None}
        if case == "padim_dyn_b1":
            st = ir_by_case["padim_static_b1"]
            entry["max_abs_diff_dyn_vs_static_cpu"] = max(abs(ir_rows[n]["score"] - st[n]["score"]) for n in names)
        report["cases"][case] = entry
    (V3 / "parity.json").write_text(json.dumps(report, indent=2))
    for case, e in report["cases"].items():
        print(case, "cpu_max_diff", round(e["max_abs_diff_cpu_vs_torch"], 6), "within_tol", e["all_within_tol_cpu"],
              "labels_agree", e["labels_agree_cpu"], "GPU", e["GPU"], "NPU", e["NPU"])
    print("torch_scores", {m: {k: round(v["score"], 6) for k, v in torch_rows[m].items()} for m in torch_rows})


if __name__ == "__main__":
    main()
