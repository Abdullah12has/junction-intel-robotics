"""Compare a candidate anom3 version with a baseline on the same held-out set and print PROMOTE or KEEP per view.

python3 -I -B eval.py --candidate v002 [--baseline current|vNNN|none] [--source episodes|folder] [--views a,b]
                       [--models-root DIR]

The held-out set is rebuilt with the same fixed split rule as train.py (common.py), so both models are scored on
identical crops. Each model is scored with its own threshold.json (fitted on that held-out set, so optimistic).
Verdict per view:
- PROMOTE if the candidate has >= 5 held-out GOOD and >= 5 held-out DEFECT and, against a baseline,
  AUROC gain >= 0.01 and balanced accuracy not lower.
- With no baseline: PROMOTE only if AUROC >= 0.70 (same class minimum).
- Otherwise KEEP. The models/current symlink is never moved here (Opus does that).
Writes MODELS/CANDIDATE/eval_vs_BASELINE.json.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cv2  # noqa: E402,F401  (used by infer)

import common  # noqa: E402
import infer  # noqa: E402

MIN_PER_CLASS = 5
MIN_GAIN = 0.01
NO_BASE_AUROC = 0.70


def resolve_root(models_root, name):
    if name in ("", "none"):
        return None
    path = os.path.join(models_root, name)
    return path if os.path.isdir(path) else None


def score_model(root, view, items):
    """Scores of one model on the held-out items of a view. Returns (metrics dict, device)."""
    vdir = os.path.join(root, view)
    if not os.path.isfile(os.path.join(vdir, "model.xml")):
        return None, None
    scores, dev = infer.score_files(vdir, [it["path"] for it in items], infer.DEVICES)
    pairs = [(it, s) for it, s in zip(items, scores) if s is not None]
    g = [s for it, s in pairs if it["label"] == "GOOD"]
    d = [s for it, s in pairs if it["label"] == "DEFECT"]
    m = common.binary_metrics(g, d, infer.read_threshold(vdir))
    m["version"] = os.path.basename(os.path.realpath(root))
    m["device"] = dev
    return m, dev


def verdict(cand, base):
    if cand is None:
        return "KEEP", "no candidate model for this view"
    if cand["n_good"] < MIN_PER_CLASS or cand["n_defect"] < MIN_PER_CLASS:
        return "KEEP", "too few held-out items (GOOD %d, DEFECT %d, need %d of each)" % (
            cand["n_good"], cand["n_defect"], MIN_PER_CLASS)
    if cand["auroc"] is None or cand["balanced_accuracy"] is None:
        return "KEEP", "metrics undefined on the held-out set"
    if base is None or base["auroc"] is None:
        if cand["auroc"] >= NO_BASE_AUROC:
            return "PROMOTE", "no baseline; AUROC %.3f >= %.2f" % (cand["auroc"], NO_BASE_AUROC)
        return "KEEP", "no baseline; AUROC %.3f below %.2f" % (cand["auroc"], NO_BASE_AUROC)
    gain = cand["auroc"] - base["auroc"]
    if gain >= MIN_GAIN and cand["balanced_accuracy"] >= base["balanced_accuracy"]:
        return "PROMOTE", "AUROC gain %+.3f and balanced accuracy %.3f >= %.3f" % (
            gain, cand["balanced_accuracy"], base["balanced_accuracy"])
    return "KEEP", "AUROC gain %+.3f; balanced accuracy %.3f vs %.3f" % (
        gain, cand["balanced_accuracy"], base["balanced_accuracy"])


def fmt(x):
    return "   -  " if x is None else "%.3f" % x


def main():
    ap = argparse.ArgumentParser(description="Compare an anom3 candidate version with a baseline.")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline", default="current")
    ap.add_argument("--source", choices=("episodes", "folder"), default="episodes")
    ap.add_argument("--logs-root", default=common.LOGS_SORT)
    ap.add_argument("--folder-root", default=common.LEGACY_E2)
    ap.add_argument("--views", default=",".join(common.VIEWS))
    ap.add_argument("--models-root", default=os.path.join(HERE, "models"))
    ap.add_argument("--out", default="", help="output json (default MODELS/CANDIDATE/eval_vs_BASELINE.json)")
    args = ap.parse_args()
    cand_root = resolve_root(args.models_root, args.candidate)
    if cand_root is None:
        raise SystemExit("candidate folder not found: %s" % os.path.join(args.models_root, args.candidate))
    base_root = resolve_root(args.models_root, args.baseline)
    items, _ = common.load(args.source, args.logs_root, args.folder_root)
    report = {"candidate": args.candidate, "baseline": args.baseline if base_root else "none", "views": {}}
    print("view          nG  nD  cand_AUROC base_AUROC cand_BA base_BA  verdict  reason")
    for view in [v for v in args.views.split(",") if v]:
        held = common.select(items, view, "heldout")
        cand, _ = score_model(cand_root, view, held) if held else (None, None)
        base = None
        if base_root is not None and held:
            base, _ = score_model(base_root, view, held)
        v, reason = verdict(cand, base)
        report["views"][view] = {"candidate": cand, "baseline": base, "verdict": v, "reason": reason}
        ng = cand["n_good"] if cand else 0
        nd = cand["n_defect"] if cand else 0
        print("%-13s %3d %3d  %s   %s   %s  %s  %-7s  %s" % (
            view, ng, nd, fmt(cand and cand["auroc"]), fmt(base and base["auroc"]),
            fmt(cand and cand["balanced_accuracy"]), fmt(base and base["balanced_accuracy"]), v, reason))
    out = args.out or os.path.join(args.models_root, args.candidate, "eval_vs_%s.json" % report["baseline"])
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print("written " + out)


if __name__ == "__main__":
    main()
