"""Calibrate and check the close-view tape rule (D-068) against the Opus close labels.

Re-segments the close frames (samples/6d6bdce3_<ep>_<start|end>_243722064316.png) with the current inspector,
keeps the ids of seg/bricks.json by matching (sample, bbox), and compares:
  - Opus labels: labels/opus_close_labels.tsv (the 35 close rows of shared/inspection-labels.tsv)
  - E2 labels:   labels/close_labels_E2.tsv (my G/T/NT/X labels)
Tape evaluation uses Opus singles only (GOOD, DEFECT_TAPE, DEFECT_TAPE_NOT2X2, DEFECT_NOT2X2); MERGED is excluded.
Positives = DEFECT_TAPE and DEFECT_TAPE_NOT2X2; negatives = GOOD and DEFECT_NOT2X2 (no tape).
Writes: eval/tape_silhouette_eval.json, labels/compare_opus_vs_E2.tsv, models/tape_thresholds.json.
Threshold: best balanced accuracy on the labelled singles (in-sample); leave-one-out accuracy reported as well.
"""
import csv
import glob
import json
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import inspector as I  # noqa: E402

SERIAL = "243722064316"  # close view
POS = {"DEFECT_TAPE", "DEFECT_TAPE_NOT2X2"}
NEG = {"GOOD", "DEFECT_NOT2X2"}


def auroc(pos, neg):
    """Mann-Whitney AUROC (ties count one half)."""
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return round(wins / (len(pos) * len(neg)), 4)


def acc_at(thr, pos, neg):
    tp = sum(1 for p in pos if p >= thr)
    tn = sum(1 for n in neg if n < thr)
    return {"thr": round(thr, 5), "tp": tp, "fn": len(pos) - tp, "tn": tn, "fp": len(neg) - tn,
            "acc": round((tp + tn) / max(len(pos) + len(neg), 1), 3)}


def best_threshold(pos, neg):
    """Threshold with the best balanced accuracy over the observed scores (ties go to the higher threshold)."""
    best_key, best_t = None, None
    for t in sorted(set(pos + neg)):
        r = acc_at(t, pos, neg)
        bal = 0.5 * (r["tp"] / max(len(pos), 1) + r["tn"] / max(len(neg), 1))
        if best_key is None or (bal, t) > best_key:
            best_key, best_t = (bal, t), t
    return best_t


def loo_accuracy(items):
    """Leave-one-out: pick the threshold on the other items, classify the held-out one."""
    correct = 0
    for i, (score, y) in enumerate(items):
        rest = items[:i] + items[i + 1:]
        t = best_threshold([s for s, yy in rest if yy], [s for s, yy in rest if not yy])
        correct += int((score >= t) == bool(y))
    return round(correct / len(items), 3)


def load_tsv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def coarse_opus(lab):
    return {"GOOD": "GOOD", "DEFECT_TAPE": "TAPE", "DEFECT_TAPE_NOT2X2": "TAPE",
            "DEFECT_NOT2X2": "NOT2X2", "MERGED": "MERGED"}.get(lab, "")


def coarse_e2(lab):
    return {"G": "GOOD", "T": "TAPE", "NT": "TAPE", "X": "X"}.get(lab, "")


def agree(e2, opus):
    if not e2 or not opus:
        return "n/a"
    ok = (e2 == "GOOD" and opus == "GOOD") or (e2 == "TAPE" and opus == "TAPE") or (e2 == "X" and opus == "MERGED")
    return "yes" if ok else "no"


def main():
    work = C.WORK
    seg = json.load(open(os.path.join(work, "seg", "bricks.json")))
    ref = {(b["sample"], tuple(b["bbox"])): b for b in seg["bricks"] if b["view"] == "close"}
    e2 = {r["brick_id"]: r for r in load_tsv(os.path.join(work, "labels", "close_labels_E2.tsv"))}
    opus = {r["key"]: r for r in load_tsv(os.path.join(work, "labels", "opus_close_labels.tsv"))}
    rows, unmatched = [], []
    for png in sorted(glob.glob(os.path.join(work, "samples", "6d6bdce3_*_" + SERIAL + ".png"))):
        sample = os.path.basename(png)[: -len("_" + SERIAL + ".png")]
        _, ep, kind = sample.split("_")
        seg_c = I.segment_close(cv2.imread(png))
        for b in seg_c["bricks"]:
            r = ref.get((sample, tuple(b["bbox"])))
            if r is None:
                unmatched.append([sample, b["bbox"], b["colour"], b["area"]])
                continue
            key = f"close/{ep}_{kind}/k{r['k']}"
            tf = b["tape_features"]
            rows.append({"id": r["id"], "sample": sample, "k": r["k"], "key": key, "colour": b["colour"],
                         "aspect": b["aspect"], "merged": b["merged"], "merged_reason": b["merged_reason"],
                         "sil_area": tf["sil_area"], "tape_frac": tf["tape_frac"], "dark_frac": tf["dark_frac"],
                         "glare_frac": tf["glare_frac"], "e2": e2.get(r["id"], {}).get("label", ""),
                         "opus": opus.get(key, {}).get("label", "")})

    # Opus singles (not MERGED) are the tape ground truth
    ev = [r for r in rows if r["opus"] in POS | NEG]
    pos_t = [r["tape_frac"] for r in ev if r["opus"] in POS]
    neg_t = [r["tape_frac"] for r in ev if r["opus"] in NEG]
    pos_d = [r["dark_frac"] for r in ev if r["opus"] in POS]
    neg_d = [r["dark_frac"] for r in ev if r["opus"] in NEG]
    if max(neg_t) < min(pos_t):
        thr_def = round(0.5 * (max(neg_t) + min(pos_t)), 5)  # midpoint of the empty gap (in-sample), not the weakest defect
    else:
        thr_def = best_threshold(pos_t, neg_t)
    thr_good = round(0.5 * thr_def, 5)  # conservative CONFIDENT_GOOD cut (routing unaffected: GOOD is still inspected)
    loo = loo_accuracy([(r["tape_frac"], int(r["opus"] in POS)) for r in ev])
    fit = acc_at(thr_def, pos_t, neg_t)

    # triage-level counts on the labelled singles, using the real triage (merged gate included)
    thr_eval = {"defect": thr_def, "good": thr_good}
    trig = {"defect_caught_GREEN": 0, "defect_uncertain": 0, "defect_routed_GOOD": 0,
            "good_CONFIDENT_GOOD": 0, "good_uncertain": 0, "good_CONFIDENT_DEFECT": 0}
    for r in ev:
        v_ = I.triage({"view": "close", "merged": r["merged"], "merged_reason": r["merged_reason"],
                       "tape_score": r["tape_frac"]}, thr_eval)[0]
        if r["opus"] in POS:
            key_ = {"CONFIDENT_DEFECT": "defect_caught_GREEN", "UNCERTAIN": "defect_uncertain",
                    "CONFIDENT_GOOD": "defect_routed_GOOD"}[v_]
        else:
            key_ = {"CONFIDENT_GOOD": "good_CONFIDENT_GOOD", "UNCERTAIN": "good_uncertain",
                    "CONFIDENT_DEFECT": "good_CONFIDENT_DEFECT"}[v_]
        trig[key_] += 1

    merged_check = {"opus_MERGED_flagged": sum(1 for r in rows if r["opus"] == "MERGED" and r["merged"]),
                    "opus_MERGED_total": sum(1 for r in rows if r["opus"] == "MERGED"),
                    "opus_single_flagged_merged": [r["id"] for r in rows if r["opus"] in POS | NEG and r["merged"]]}

    # comparison table: Opus vs E2 labels on the same brick ids
    cmp_rows = []
    for k_, o in sorted(opus.items()):
        rid = next((r for r in rows if r["key"] == k_), None)
        if rid is None:
            cmp_rows.append([k_, "NOT_IN_SEG", o["label"], "", "", "", "", "", "", "", "", "", ""])
            continue
        e = rid["e2"]
        cmp_rows.append([k_, rid["id"], o["label"], e, coarse_opus(o["label"]), coarse_e2(e), agree(coarse_e2(e), coarse_opus(o["label"])),
                         rid["colour"], rid["aspect"], rid["merged"], rid["merged_reason"], rid["tape_frac"], rid["dark_frac"]])
    agree_counts = {a: sum(1 for c in cmp_rows if c[6] == a) for a in ("yes", "no", "n/a")}
    with open(os.path.join(work, "labels", "compare_opus_vs_E2.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["opus_key", "brick_id", "opus_label", "e2_label", "opus_coarse", "e2_coarse", "agree",
                    "colour_fresh", "aspect", "merged", "merged_reason", "tape_frac", "dark_frac"])
        w.writerows(cmp_rows)

    thr_obj = {"rule": "close-view tape_frac: dark (V 50-130, S<90), textured (5x5 V std > 4) pixels inside the silhouette, over silhouette area",
               "defect": thr_def, "good": thr_good, "tex_sd": I.TEX_SD, "v_band": list(I.TAPE_V),
               "s_max": I.TAPE_S_MAX, "calibrated_on": "Opus close labels, singles only (in-sample; LOO reported)"}
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    with open(I.THR_PATH, "w") as fh:
        json.dump(thr_obj, fh, indent=1)

    out = {"n_close_bricks_seg": len(rows) + len(unmatched), "n_matched_to_seg_ids": len(rows),
           "unmatched": unmatched, "opus_keys": len(opus), "opus_keys_in_seg": sum(1 for c in cmp_rows if c[1] != "NOT_IN_SEG"),
           "tape_eval_singles": {"n_pos": len(pos_t), "n_neg": len(neg_t),
                                 "auroc_tape_frac": auroc(pos_t, neg_t), "auroc_dark_frac_untextured": auroc(pos_d, neg_d),
                                 "fit_threshold": fit, "loo_accuracy": loo, "triage_counts": trig,
                                 "thr_defect": thr_def, "thr_good": thr_good},
           "merged_check": merged_check, "agreement_e2_vs_opus": agree_counts,
           "per_brick": rows}
    os.makedirs(os.path.join(work, "eval"), exist_ok=True)
    with open(os.path.join(work, "eval", "tape_silhouette_eval.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    summary = {k: v for k, v in out.items() if k != "per_brick"}
    print(json.dumps(summary, indent=1))
    print("rows", len(rows), "eval singles", len(ev))


if __name__ == "__main__":
    main()
