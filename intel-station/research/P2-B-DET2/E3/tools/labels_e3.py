"""E3 strict visual labels by tile index (verify_idx_close.jpg / verify_idx_top.jpg, labelled by E3 on the
current bricks_build.json order) and evaluation: top-view shape agreement, close-view tape AUROC per feature.
Labels are E3 visual judgements, not Opus-verified. Run: envs/data-env/bin/python -I -B tools/labels_e3.py"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/intel/intel-junction-2026/app/anomaly/det2_e3")
import det2  # noqa: E402

WORK = "/home/intel/intel-junction-2026/work/P2-B-DET2/E3"
CLOSE_TAPE = {27: "YES", 38: "YES", 32: "LIKELY", 34: "LIKELY",
              3: "UNSURE", 28: "UNSURE", 30: "UNSURE", 41: "UNSURE", 46: "UNSURE", 54: "UNSURE",
              7: "NOT_BRICK", 20: "NOT_BRICK", 36: "NOT_BRICK", 43: "NOT_BRICK"}
TOP_SHAPE = {0: "not_2x2", 3: "not_2x2", 5: "not_2x2", 30: "not_2x2", 45: "not_2x2",
             22: "unknown", 34: "unknown", 35: "unknown", 37: "unknown"}
TOP_TAPE = {28: "LIKELY"}


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return round(float((gt + 0.5 * eq) / (len(pos) * len(neg))), 3)


def main():
    recs = json.load(open(f"{WORK}/out/bricks_build.json"))
    cal = det2.load_cal()
    top = [r for r in recs if r["view"] == "top"]
    close = [r for r in recs if r["view"] == "close"]
    with open(f"{WORK}/out/e3_visual_labels.tsv", "w") as fh:
        fh.write("view\tindex\tid\tshape_visual\ttape_visual\talgo_colour\talgo_shape\n")
        for i, r in enumerate(top):
            fh.write(f"top\t{i}\t{r['id']}\t{TOP_SHAPE.get(i, '2x2')}\t{TOP_TAPE.get(i, 'NO')}\t{r['colour']}\t{r.get('shape')}\n")
        for i, r in enumerate(close):
            fh.write(f"close\t{i}\t{r['id']}\tNA\t{CLOSE_TAPE.get(i, 'NO')}\t{r['colour']}\t-\n")
    pairs = [(TOP_SHAPE.get(i, "2x2"), r.get("shape"), i, r["id"]) for i, r in enumerate(top)]
    acc = sum(1 for v, a, _, _ in pairs if v == a) / max(len(pairs), 1)
    mism = [(i, rid, v, a) for v, a, i, rid in pairs if v != a]
    print("SHAPE top n", len(pairs), "accuracy", round(acc, 3), "mismatches", mism)
    cats = ["2x2", "not_2x2", "unknown"]
    conf = {}
    for v, a, _, _ in pairs:
        conf[f"{v}->{a}"] = conf.get(f"{v}->{a}", 0) + 1
    print("SHAPE confusion", conf)
    pos = [r for i, r in enumerate(close) if CLOSE_TAPE.get(i) in ("YES", "LIKELY")]
    yes = [r for i, r in enumerate(close) if CLOSE_TAPE.get(i) == "YES"]
    neg = [r for i, r in enumerate(close) if CLOSE_TAPE.get(i, "NO") == "NO"]
    print("TAPE close pos", len(pos), "yes", len(yes), "neg", len(neg))
    res = {}
    for f in det2.TAPE_FEATS:
        p95 = cal["views"]["close"]["tape_stats"][f]["p95"]
        sp = [r["tape_feats"][f] / max(p95, 1e-6) for r in pos]
        sn = [r["tape_feats"][f] / max(p95, 1e-6) for r in neg]
        sy = [r["tape_feats"][f] / max(p95, 1e-6) for r in yes]
        res[f] = {"p95_neg_calib": p95, "auroc_pos_vs_neg": auroc(sp, sn), "auroc_yes_vs_neg": auroc(sy, sn),
                  "pos_ratio": [round(x, 2) for x in sp], "neg_frac_ratio_over_1": round(float(np.mean(np.array(sn) > 1.0)), 3)}
        print("TAPE", f, res[f])
    def mx(r):
        return max(r["tape_feats"][f] / max(cal["views"]["close"]["tape_stats"][f]["p95"], 1e-6) for f in det2.TAPE_FEATS)
    sp, sn, sy = [mx(r) for r in pos], [mx(r) for r in neg], [mx(r) for r in yes]
    res["max_ratio"] = {"auroc_pos_vs_neg": auroc(sp, sn), "auroc_yes_vs_neg": auroc(sy, sn),
                        "pos_ratio": [round(x, 2) for x in sp], "neg_frac_ratio_over_1": round(float(np.mean(np.array(sn) > 1.0)), 3)}
    print("TAPE max_ratio", res["max_ratio"])
    json.dump({"shape_acc_top": round(acc, 3), "shape_mismatches": mism, "tape_close": res,
               "n": {"top": len(top), "close": len(close), "tape_pos": len(pos), "tape_yes": len(yes), "tape_neg": len(neg)}},
              open(f"{WORK}/out/eval_e3.json", "w"), indent=1)
    print("WROTE e3_visual_labels.tsv and eval_e3.json")


if __name__ == "__main__":
    main()
