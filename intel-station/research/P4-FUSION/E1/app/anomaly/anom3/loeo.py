"""anom3 round 3: leave-one-episode-out PatchCore on the brick-centred multi crops (CPU only, FP16 IR).

Input: R/work/P2-B-ANOM3/E1/multi/manifest.tsv (written by extract_crops.py).
Clean crops = status ok, minus REVIEW_EXCLUDE (wrong brick seen on the contact sheets). ok_hue_mismatch crops are
never trained on; they are scored and reported as flagged. Clean episodes with no GOOD training crop left are not
folds.

For each view (wrist_multi, close_multi) and each clean episode E: train PatchCore on the clean GOOD crops of the
other clean episodes, export an OpenVINO FP16 IR, score every crop of E. A fit is shared when two held-out episodes
leave the same GOOD training set (both DEFECT episodes do).

Variants
  base      input 256 px, no augmentation (train.py settings: wide_resnet50_2, layer2 + layer3, num_neighbors 9)
  fewshot   input 448 px, 8 augmentations per training crop (4 rotations x horizontal flip), the settings of the
            VAND few-shot track (Santos et al. 2023, arXiv 2307.10792) that Opus asked for; otherwise the same.
Both variants resize the crops to the input size with cv2 INTER_LINEAR, both at training and at scoring time.

Models go to R/work/P2-B-ANOM3/E1/scratch/loeo/. Nothing under models/ is written.
Run: nice -n 19 env OMP_NUM_THREADS=4 $PY -I -B loeo.py --variant base|fewshot [--keep-review-flagged]
"""
import argparse
import csv
import glob
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import infer  # noqa: E402

R = "/home/intel/intel-junction-2026"
E1 = os.path.join(R, "work", "P2-B-ANOM3", "E1")
MANIFEST = os.path.join(E1, "multi", "manifest.tsv")
SCRATCH = os.path.join(E1, "scratch", "loeo")
OUT = os.path.join(E1, "loeo")
SEED = 0
CORESET_TARGET = 2000
FIT_BUDGET_S = 180.0
VIEWS = ("wrist_multi", "close_multi")
# Close crops seen on the contact sheets as the wrong brick: the yellow tape brick on the table is merged into the
# box together with the red GOOD brick (ep_0015 idx 93, 95, 97, 98; idx 97 is CONFIDENT_DEFECT because of that brick).
REVIEW_EXCLUDE = {("close_multi", "any_1740/ep_0015", "93"), ("close_multi", "any_1740/ep_0015", "95"),
                  ("close_multi", "any_1740/ep_0015", "97"), ("close_multi", "any_1740/ep_0015", "98")}
VARIANTS = {"base": {"size": 256, "aug": False}, "fewshot": {"size": 448, "aug": True}}


def load_items(keep_review_flagged):
    items = []
    with open(MANIFEST, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["view"] not in VIEWS or r["status"] not in ("ok", "ok_hue_mismatch") or not r["crop_path"]:
                continue
            flag = "clean"
            if r["status"] == "ok_hue_mismatch":
                flag = "hue_mismatch"
            elif not keep_review_flagged and (r["view"], r["key"], r["idx"]) in REVIEW_EXCLUDE:
                flag = "wrong_brick_review"
            items.append({"view": r["view"], "key": r["key"], "idx": r["idx"], "label": r["label"],
                          "status": r["status"], "flag": flag, "path": r["crop_path"]})
    return items


def build_train_dir(root, crops, size, aug):
    shutil.rmtree(root, ignore_errors=True)
    good = os.path.join(root, "train", "good")
    os.makedirs(good)
    n = 0
    for it in crops:
        img = cv2.imread(it["path"], cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("cannot read " + it["path"])
        if aug:
            variants = []
            for k in range(4):
                rot = np.ascontiguousarray(np.rot90(img, k))
                variants.append(rot)
                variants.append(cv2.flip(rot, 1))
        else:
            variants = [img]
        for v in variants:
            v = cv2.resize(v, (size, size), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(good, "%05d.png" % n), v)
            n += 1
    return n


def fit_export(name, root, size, n_images, ir_dir, tag):
    from anomalib.data import Folder
    from anomalib.deploy import ExportType
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
    from anomalib.post_processing.post_processor import PostProcessor
    import openvino as ov

    torch.manual_seed(SEED)
    ratio = float(min(0.1, max(0.001, CORESET_TARGET / float(n_images * (size // 8) ** 2))))
    # Validation split as in train.py (the Lightning loop needs val_data). Score normalisation is switched off below:
    # with a validation split anomalib normalises pred_score against the validation scores, and every crop saturated
    # at 1.0 in the first base run (wrist folds).
    dm = Folder(name=name, root=root, normal_dir="train/good", train_batch_size=4, eval_batch_size=4,
                num_workers=0, test_split_mode="none", val_split_mode="from_train",
                val_split_ratio=0.2 if n_images >= 10 else 0.1, seed=SEED)
    pre = Patchcore.configure_pre_processor(image_size=(size, size))
    model = Patchcore(backbone="wide_resnet50_2", layers=("layer2", "layer3"), pre_trained=True,
                      coreset_sampling_ratio=ratio, num_neighbors=9, pre_processor=pre,
                      post_processor=PostProcessor(enable_normalization=False))
    engine = Engine(accelerator="cpu", devices=1, max_epochs=1, logger=False,
                    default_root_dir=os.path.join(SCRATCH, tag, "runs", name))
    t0 = time.perf_counter()
    engine.fit(model=model, datamodule=dm)
    fit_s = time.perf_counter() - t0
    exp_root = os.path.join(SCRATCH, tag, "export", name)
    shutil.rmtree(exp_root, ignore_errors=True)
    t1 = time.perf_counter()
    engine.export(model=model, export_type=ExportType.OPENVINO, export_root=exp_root,
                  input_size=(size, size), onnx_kwargs={"opset_version": 14})
    export_s = time.perf_counter() - t1
    found = sorted(glob.glob(os.path.join(exp_root, "**", "*.xml"), recursive=True))
    if not found:
        raise RuntimeError("export wrote no IR under " + exp_root)
    shutil.rmtree(ir_dir, ignore_errors=True)
    os.makedirs(ir_dir)
    ov.save_model(ov.Core().read_model(found[0]), os.path.join(ir_dir, "model.xml"), compress_to_fp16=True)
    return ratio, round(fit_s, 1), round(export_s, 1)


def auroc(neg, pos):
    """P(score of a DEFECT item > score of a GOOD item); ties count 0.5 (Mann-Whitney). None if a side is empty."""
    if not neg or not pos:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return round(wins / (len(pos) * len(neg)), 4)


def mean_or_none(x):
    return round(float(np.mean(x)), 3) if x else None


def max_or_none(x):
    return round(float(np.max(x)), 3) if x else None


def main():
    ap = argparse.ArgumentParser(description="Leave-one-episode-out PatchCore on the anom3 multi crops (CPU).")
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--keep-review-flagged", action="store_true",
                    help="sensitivity run: keep the review-excluded wrong-brick crops as clean (train and score)")
    args = ap.parse_args()
    torch.set_num_threads(4)
    cfg = VARIANTS[args.variant]
    tag = args.variant + ("_keepflagged" if args.keep_review_flagged else "")
    items = load_items(args.keep_review_flagged)
    labels = {}
    for it in items:
        labels.setdefault(it["key"], it["label"])
    summary = {"tag": tag, "variant": args.variant, "size": cfg["size"], "augmentations": 8 if cfg["aug"] else 1,
               "coreset_target": CORESET_TARGET, "fit_budget_s": FIT_BUDGET_S, "manifest": MANIFEST,
               "review_exclude": sorted("%s %s idx %s" % x for x in REVIEW_EXCLUDE),
               "keep_review_flagged": args.keep_review_flagged, "views": {}, "fits": []}
    rows, cache = [], {}
    for view in [v for v in args.views.split(",") if v]:
        clean_keys = sorted({it["key"] for it in items if it["view"] == view and it["flag"] == "clean"})
        vs = {"episodes": {}, "folds": []}
        for k in clean_keys:
            vs["episodes"][k] = {
                "label": labels[k],
                "clean": sum(1 for it in items if it["view"] == view and it["key"] == k and it["flag"] == "clean"),
                "flagged": sum(1 for it in items if it["view"] == view and it["key"] == k and it["flag"] != "clean")}
        for h in clean_keys:
            fold = {"heldout": h, "label": labels[h]}
            train_eps = tuple(k for k in clean_keys if k != h and labels[k] == "GOOD")
            crops = [it for it in items if it["view"] == view and it["flag"] == "clean" and it["label"] == "GOOD"
                     and it["key"] in train_eps]
            if not crops:
                fold["skipped"] = "no clean GOOD training crop left"
                vs["folds"].append(fold)
                print(view, h, "skipped:", fold["skipped"], flush=True)
                continue
            if (view, train_eps) not in cache:
                fid = "%s__%s" % (view, "+".join(e.split("/")[-1] for e in train_eps))
                root = os.path.join(SCRATCH, tag, fid, "build")
                n_images = build_train_dir(root, crops, cfg["size"], cfg["aug"])
                ir_dir = os.path.join(SCRATCH, tag, fid, "ir")
                t0 = time.time()
                ratio, fit_s, export_s = fit_export(fid, root, cfg["size"], n_images, ir_dir, tag)
                cache[(view, train_eps)] = {"fit": fid, "ir": ir_dir, "train_episodes": list(train_eps),
                                    "n_train_crops": len(crops), "n_train_images": n_images,
                                    "coreset_ratio": round(ratio, 5), "fit_s": fit_s, "export_s": export_s,
                                    "fit_within_budget": fit_s <= FIT_BUDGET_S, "wall_s": round(time.time() - t0, 1)}
                summary["fits"].append(cache[(view, train_eps)])
                print("fit", fid, json.dumps(cache[(view, train_eps)]), flush=True)
            fit = cache[(view, train_eps)]
            ev = [it for it in items if it["view"] == view and it["key"] == h]
            scores, dev = infer.score_files(fit["ir"], [it["path"] for it in ev], infer.DEVICES)
            cs, fs = [], []
            for it, s in zip(ev, scores):
                rows.append({"tag": tag, "view": view, "heldout": h, "fit": fit["fit"], "key": it["key"],
                             "idx": it["idx"], "label": it["label"], "status": it["status"], "flag": it["flag"],
                             "score": "" if s is None else "%.4f" % s})
                if s is not None:
                    (cs if it["flag"] == "clean" else fs).append(float(s))
            fold.update({"fit": fit["fit"], "train_episodes": fit["train_episodes"],
                         "n_train_crops": fit["n_train_crops"], "n_train_images": fit["n_train_images"],
                         "device": dev, "clean_n": len(cs), "clean_mean": mean_or_none(cs),
                         "clean_max": max_or_none(cs), "flagged_n": len(fs), "flagged_mean": mean_or_none(fs),
                         "flagged_max": max_or_none(fs)})
            vs["folds"].append(fold)
            print(view, h, labels[h], json.dumps({k: fold[k] for k in ("clean_n", "clean_mean", "clean_max",
                                                                         "flagged_n", "flagged_mean")}), flush=True)
        g_ep = [f["clean_mean"] for f in vs["folds"] if f.get("label") == "GOOD" and f.get("clean_mean") is not None]
        d_ep = [f["clean_mean"] for f in vs["folds"] if f.get("label") == "DEFECT" and f.get("clean_mean") is not None]
        g_cr = [float(r["score"]) for r in rows if r["view"] == view and r["flag"] == "clean" and
                r["label"] == "GOOD" and r["score"] != ""]
        d_cr = [float(r["score"]) for r in rows if r["view"] == view and r["flag"] == "clean" and
                r["label"] == "DEFECT" and r["score"] != ""]
        vs.update({"auroc_episode_mean": auroc(g_ep, d_ep), "episode_pairs": len(g_ep) * len(d_ep),
                   "n_good_episodes": len(g_ep), "n_defect_episodes": len(d_ep),
                   "auroc_crop_pooled": auroc(g_cr, d_cr), "n_clean_good_crops": len(g_cr),
                   "n_clean_defect_crops": len(d_cr)})
        summary["views"][view] = vs
        print("VIEW", view, "auroc_episode_mean", vs["auroc_episode_mean"], "pairs", vs["episode_pairs"],
              "auroc_crop_pooled", vs["auroc_crop_pooled"], flush=True)
    out_dir = os.path.join(OUT, tag)
    os.makedirs(out_dir, exist_ok=True)
    cols = ["tag", "view", "heldout", "fit", "key", "idx", "label", "status", "flag", "score"]
    with open(os.path.join(out_dir, "scores.tsv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print("wrote", out_dir, "rows", len(rows), flush=True)


if __name__ == "__main__":
    main()
