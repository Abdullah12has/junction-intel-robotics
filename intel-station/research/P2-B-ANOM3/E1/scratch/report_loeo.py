"""Compact report of R/work/P2-B-ANOM3/E1/loeo/<tag>/summary.json (read-only; prints to stdout).

Usage: python3 -I -B report_loeo.py TAG [TAG ...]
"""
import json
import os
import sys

LOEO = "/home/intel/intel-junction-2026/work/P2-B-ANOM3/E1/loeo"
for tag in sys.argv[1:]:
    with open(os.path.join(LOEO, tag, "summary.json"), encoding="utf-8") as fh:
        s = json.load(fh)
    print("== %s: size %s, augmentations %s, keep_review_flagged %s"
          % (tag, s["size"], s["augmentations"], s["keep_review_flagged"]), flush=True)
    for view, vs in s["views"].items():
        print("  %s  AUROC over episodes (mean score) %s from %d GOOD x DEFECT pairs (GOOD eps %d, DEFECT eps %d); "
              "AUROC over clean crops (pooled) %s from %d GOOD and %d DEFECT crops"
              % (view, vs["auroc_episode_mean"], vs["episode_pairs"], vs["n_good_episodes"],
                 vs["n_defect_episodes"], vs["auroc_crop_pooled"], vs["n_clean_good_crops"],
                 vs["n_clean_defect_crops"]), flush=True)
        for f in vs["folds"]:
            if "skipped" in f:
                print("    held-out %-18s %-6s SKIPPED: %s" % (f["heldout"], f["label"], f["skipped"]), flush=True)
                continue
            print("    held-out %-18s %-6s train %-46s clean n %2d mean %8.1f max %8.1f | flagged n %d mean %s"
                  % (f["heldout"], f["label"], "+".join(e.split("/")[-1] for e in f["train_episodes"]),
                     f["clean_n"], f["clean_mean"], f["clean_max"], f["flagged_n"],
                     "-" if f["flagged_mean"] is None else "%.1f" % f["flagged_mean"]), flush=True)
    for fit in s["fits"]:
        print("  fit %-40s crops %2d images %3d coreset_ratio %s fit_s %s export_s %s within_180s %s"
              % (fit["fit"], fit["n_train_crops"], fit["n_train_images"], fit["coreset_ratio"], fit["fit_s"],
                 fit["export_s"], fit["fit_within_budget"]), flush=True)
