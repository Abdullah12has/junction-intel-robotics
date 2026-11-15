"""Train one PatchCore per view on GOOD crops only (anomalib 2.7.0, CPU), export OpenVINO FP16 IR, calibrate the threshold.

Writes MODELS/VERSION/VIEW/: model.xml, model.bin, metadata.json (if anomalib wrote it), threshold.json, metrics.json,
train_list.txt, eval_list.tsv (held-out scores), disagreements.tsv (model vs label, for Opus to check by eye),
and MODELS/VERSION/summary.json. Held-out crops are never trained on. The threshold is fitted on the same held-out
crops that the metrics use, so the metrics are optimistic; eval.py compares versions on that same set.

CPU only. Run (see RUN.md): nice -n 19 $PY -I -B train.py [--dry-run] [--version VNNN] [--source episodes|folder]
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")  # leave CPU for the live robot loop

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cv2  # noqa: E402

import common  # noqa: E402
import infer  # noqa: E402

FIT_BUDGET_S = 180.0
SEED = 0
MIN_TRAIN_GOOD = 2  # a floor only: a view with fewer than 4 training GOOD crops is written with low_n true
CORESET_TARGET = 2000
SCRATCH = os.path.join(common.R, "work", "P2-B-ANOM3", "E1", "scratch")


def next_version(models_root):
    nums = []
    if os.path.isdir(models_root):
        for name in os.listdir(models_root):
            m = re.fullmatch(r"v(\d{3})", name)
            if m:
                nums.append(int(m.group(1)))
    return "v%03d" % (max(nums) + 1 if nums else 1)


def coreset_ratio(n_train):
    patches = n_train * (infer.SIZE // 8) ** 2  # layer2 stride 8: 32 x 32 patches per crop at 256 px
    return float(min(0.1, max(0.001, CORESET_TARGET / float(patches))))


def build_train_dir(root, items):
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(os.path.join(root, "train", "good"))
    kept, skipped = [], 0
    for i, it in enumerate(items):
        img = cv2.imread(it["path"], cv2.IMREAD_COLOR)
        if img is None:
            skipped += 1
            continue
        name = "%05d.png" % i
        img = cv2.resize(img, (infer.SIZE, infer.SIZE), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(root, "train", "good", name), img)
        kept.append(dict(it, file=name))
    return kept, skipped


def fit_and_export(root, version, scratch, out_dir, n_train, ratio, view):
    from anomalib.data import Folder
    from anomalib.deploy import ExportType
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
    from anomalib.post_processing import PostProcessor
    import openvino as ov

    import torch
    torch.manual_seed(SEED)  # seeds the coreset projection and the val split; identical runs still differ by ~0.2% in threshold (Lightning shuffle)
    dm = Folder(name="anom3_" + view, root=root, normal_dir="train/good",
                train_batch_size=4, eval_batch_size=4, num_workers=0,
                test_split_mode="none", val_split_mode="from_train",
                val_split_ratio=0.2 if n_train >= 10 else 0.1, seed=SEED)
    engine = Engine(accelerator="cpu", devices=1, max_epochs=1, logger=False,
                    default_root_dir=os.path.join(scratch, "runs", version, view))
    model = Patchcore(backbone="wide_resnet50_2", layers=("layer2", "layer3"), pre_trained=True,
                      coreset_sampling_ratio=ratio, num_neighbors=9,
                      post_processor=PostProcessor(enable_normalization=False))  # raw pred_score: min-max from a few val crops clamps held-out scores to 1.0
    t0 = time.perf_counter()
    engine.fit(model=model, datamodule=dm)
    fit_s = time.perf_counter() - t0
    exp_root = os.path.join(scratch, "export", version, view)
    shutil.rmtree(exp_root, ignore_errors=True)
    t1 = time.perf_counter()
    engine.export(model=model, export_type=ExportType.OPENVINO, export_root=exp_root,
                  input_size=(infer.SIZE, infer.SIZE), onnx_kwargs={"opset_version": 14})
    export_s = time.perf_counter() - t1
    found = sorted(glob.glob(os.path.join(exp_root, "**", "*.xml"), recursive=True))
    if not found:
        raise RuntimeError("export wrote no IR under " + exp_root)
    os.makedirs(out_dir, exist_ok=True)
    ov.save_model(ov.Core().read_model(found[0]), os.path.join(out_dir, "model.xml"), compress_to_fp16=True)
    for meta in glob.glob(os.path.join(os.path.dirname(found[0]), "metadata.json")):
        shutil.copy(meta, os.path.join(out_dir, "metadata.json"))
    return round(fit_s, 1), round(export_s, 1)


def counts_by(items, key):
    out = {}
    for it in items:
        out[it[key]] = out.get(it[key], 0) + 1
    return out


def write_lines(path, header, rows):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(str(x) for x in row) + "\n")


def train_view(view, items, version, models_root, scratch):
    out_dir = os.path.join(models_root, version, view)
    train_items = common.select(items, view, "train")
    eval_good = common.select(items, view, "heldout", "GOOD")
    eval_def = common.select(items, view, "heldout", "DEFECT")
    info = {"view": view, "train_good": len(train_items), "heldout_good": len(eval_good),
            "heldout_defect": len(eval_def), "low_n": len(train_items) < 4,
            "train_groups": sorted({it["group"] for it in train_items}),
            "heldout_good_groups": sorted({it["group"] for it in eval_good}),
            "heldout_defect_groups": sorted({it["group"] for it in eval_def}),
            "success_missing": sum(1 for it in train_items + eval_good + eval_def
                                   if it.get("success") == "missing")}
    if len(train_items) < MIN_TRAIN_GOOD:
        info["skipped"] = "only %d GOOD train crops (min %d)" % (len(train_items), MIN_TRAIN_GOOD)
        return info
    if not eval_good:
        info["skipped"] = "no held-out GOOD crops to calibrate the threshold"
        return info
    root = os.path.join(scratch, "build", version, view)
    kept, skipped = build_train_dir(root, train_items)
    ratio = coreset_ratio(len(kept))
    fit_s, export_s = fit_and_export(root, version, scratch, out_dir, len(kept), ratio, view)

    eval_all = eval_good + eval_def
    all_scores, dev = infer.score_files(out_dir, [it["path"] for it in eval_all], infer.DEVICES)
    pairs = [(it, s) for it, s in zip(eval_all, all_scores) if s is not None]
    g_pairs = [(it, s) for it, s in pairs if it["label"] == "GOOD"]
    d_pairs = [(it, s) for it, s in pairs if it["label"] == "DEFECT"]
    s_g = [s for _, s in g_pairs]
    s_d = [s for _, s in d_pairs]
    if s_d:
        thr, _ = common.pick_threshold(s_g, s_d)
        rule = "max balanced accuracy on held-out GOOD+DEFECT; ties go to the higher threshold"
    else:
        thr = float(max(s_g))
        rule = "no held-out DEFECT: max held-out GOOD score (uncalibrated)"
    m = common.binary_metrics(s_g, s_d, thr)
    m.update({"view": view, "version": version, "threshold_rule": rule, "fit_on_heldout": True,
              "n_train": len(kept), "low_n": len(kept) < 4, "train_groups": sorted({it["group"] for it in kept}),
              "n_train_unreadable": skipped, "train_sources": counts_by(kept, "source"),
              "heldout_sources": counts_by([it for it, _ in g_pairs + d_pairs], "source"),
              "backbone": "wide_resnet50_2", "layers": ["layer2", "layer3"], "image_size": infer.SIZE,
              "coreset_ratio": round(ratio, 4), "fit_s": fit_s, "export_s": export_s,
              "fit_budget_s": FIT_BUDGET_S, "fit_within_budget": fit_s <= FIT_BUDGET_S,
              "scoring_device": dev, "ir_precision": "FP16", "anomalib": "2.7.0",
              "files": ["model.xml", "model.bin", "metadata.json", "threshold.json", "metrics.json",
                        "train_list.txt", "eval_list.tsv", "disagreements.tsv"]})
    with open(os.path.join(out_dir, "threshold.json"), "w", encoding="utf-8") as fh:
        json.dump({"view": view, "version": version, "threshold": thr, "rule": rule,
                   "score": "IR pred_score (FP16), DEFECT when score > threshold",
                   "fit_on_heldout": True, "n_good": m["n_good"], "n_defect": m["n_defect"],
                   "balanced_accuracy": m["balanced_accuracy"], "auroc": m["auroc"], "device": dev},
                  fh, indent=1)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=1)
    write_lines(os.path.join(out_dir, "train_list.txt"), ["file", "path", "label", "source", "group"],
                [(it["file"], it["path"], it["label"], it["source"], it["group"]) for it in kept])
    rows = [(it["path"], it["label"], it["source"], it["group"], "%.6f" % s,
             "DEFECT" if s > thr else "GOOD") for it, s in g_pairs + d_pairs]
    header = ["path", "label", "source", "group", "score", "pred"]
    write_lines(os.path.join(out_dir, "eval_list.tsv"), header, rows)
    write_lines(os.path.join(out_dir, "disagreements.tsv"), header, [r for r in rows if r[1] != r[5]])
    info.update({"n_train": len(kept), "auroc": m["auroc"], "balanced_accuracy": m["balanced_accuracy"],
                 "threshold": thr, "fit_s": fit_s, "export_s": export_s, "device": dev,
                 "disagreements": sum(1 for r in rows if r[1] != r[5]), "coreset_ratio": round(ratio, 4)})
    return info


def main():
    ap = argparse.ArgumentParser(description="Train anom3 PatchCore per view (GOOD only), export FP16 IR.")
    ap.add_argument("--source", choices=("episodes", "folder"), default="episodes")
    ap.add_argument("--logs-root", default=common.LOGS_SORT)
    ap.add_argument("--folder-root", default=common.LEGACY_E2)
    ap.add_argument("--label-sources", default="", help="comma list such as opus,det2 (empty = all)")
    ap.add_argument("--views", default=",".join(common.VIEWS))
    ap.add_argument("--version", default="", help="e.g. v002 (empty = next free number)")
    ap.add_argument("--models-root", default=os.path.join(HERE, "models"))
    ap.add_argument("--scratch", default=SCRATCH)
    ap.add_argument("--dry-run", action="store_true", help="print the dataset summary and train nothing")
    args = ap.parse_args()
    label_sources = [s for s in args.label_sources.split(",") if s] or None
    views = [v for v in args.views.split(",") if v]
    items, excluded = common.load(args.source, args.logs_root, args.folder_root, label_sources)
    report = {"source": args.source, "excluded": excluded, "views": {}}
    for view in views:
        tr = common.select(items, view, "train")
        hg = common.select(items, view, "heldout", "GOOD")
        hd = common.select(items, view, "heldout", "DEFECT")
        report["views"][view] = {
            "train_good": len(tr), "heldout_good": len(hg), "heldout_defect": len(hd),
            "train_groups": sorted({it["group"] for it in tr}),
            "heldout_good_groups": sorted({it["group"] for it in hg}),
            "heldout_defect_groups": sorted({it["group"] for it in hd}),
            "success_missing": sum(1 for it in items if it["view"] == view and it.get("success") == "missing")}
    if args.dry_run:
        print(json.dumps(report, indent=1))
        return
    version = args.version or next_version(args.models_root)
    os.makedirs(os.path.join(args.models_root, version), exist_ok=True)
    summary = {"version": version, "source": args.source, "excluded": excluded, "views": []}
    for view in views:
        t0 = time.perf_counter()
        info = train_view(view, items, version, args.models_root, args.scratch)
        info["wall_s"] = round(time.perf_counter() - t0, 1)
        summary["views"].append(info)
        print(json.dumps(info), flush=True)
    with open(os.path.join(args.models_root, version, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({"version": version, "written": os.path.join(args.models_root, version)}))


if __name__ == "__main__":
    main()
