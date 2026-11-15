"""Evaluate the PatchCore-style patch-kNN tape score on E2 labelled close-view bricks (episode-disjoint split).

Bank = patches of train G bricks (episodes 3-6). Score = 95th percentile over a brick's patches of the distance
to the nearest bank patch. Also reports the rule features (off_frac, glare_frac). Writes eval/tape_eval.json.
usage: python3 -I tape_eval.py
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK  # noqa: E402

SEG = os.path.join(WORK, "seg")
rows = list(csv.DictReader(open(os.path.join(WORK, "labels", "close_labels_E2.tsv")), delimiter="\t"))
train = [r for r in rows if r["label"] == "G" and int(r["episode"]) in (3, 4, 5, 6)]
test_g = [r for r in rows if r["label"] == "G" and int(r["episode"]) not in (3, 4, 5, 6)]
test_d = [r for r in rows if r["label"] in ("T", "NT")]


def desc(r):
    return np.load(os.path.join(SEG, r["brick_id"] + "_patches.npy"))


def score(d, bank):
    dist = np.sqrt(((d[:, None, :] - bank[None, :, :]) ** 2).sum(-1)).min(axis=1)
    return float(np.percentile(dist, 95))


def auroc(pos, neg):
    """P(score_pos > score_neg) with ties counted half (Mann-Whitney)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


bank_all = np.concatenate([desc(r) for r in train])
# leave-one-brick-out scores on train goods (bank without the brick itself)
loo = []
for i, r in enumerate(train):
    bank = np.concatenate([desc(q) for j, q in enumerate(train) if j != i])
    loo.append(score(desc(r), bank))
thr = float(np.max(loo))
s_g = {r["brick_id"]: score(desc(r), bank_all) for r in test_g}
s_d = {r["brick_id"]: score(desc(r), bank_all) for r in test_d}
au = auroc(list(s_d.values()), list(s_g.values()))
acc_at_thr = (sum(v > thr for v in s_d.values()) + sum(v <= thr for v in s_g.values())) / (len(s_d) + len(s_g))
off_g = [float(r["off_frac"]) for r in test_g]
off_d = [float(r["off_frac"]) for r in test_d]
gl_g = [float(r["glare_frac"]) for r in test_g]
gl_d = [float(r["glare_frac"]) for r in test_d]
out = {"bank_bricks": len(train), "bank_patches": int(bank_all.shape[0]), "train_good_ids": [r["brick_id"] for r in train],
       "loo_train_scores": [round(x, 4) for x in loo], "threshold_max_loo": round(thr, 4),
       "test_good_scores": {k: round(v, 4) for k, v in s_g.items()},
       "test_defect_scores": {k: round(v, 4) for k, v in s_d.items()},
       "auroc_patch_knn_defect_vs_good": round(au, 4), "acc_at_threshold": round(float(acc_at_thr), 4),
       "n_test_good": len(test_g), "n_test_defect": len(test_d),
       "auroc_rule_off_frac": round(auroc(off_d, off_g), 4), "auroc_rule_glare_frac": round(auroc(gl_d, gl_g), 4)}
os.makedirs(os.path.join(WORK, "eval"), exist_ok=True)
json.dump(out, open(os.path.join(WORK, "eval", "tape_eval.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("train_good_ids",)}, indent=0)[:1800])
