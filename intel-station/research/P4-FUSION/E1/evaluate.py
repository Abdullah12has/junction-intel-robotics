"""P4-FUSION E1 step 3 (stage fusion-v1): LOEO metrics for the tape rule, PatchCore and PaDiM alone, and LOEO fusion.

Fused candidates, all leave-one-episode-out (the held-out group never fits its own fusion):
  <model>_LR   logistic regression on standardised [tape_score, anomaly score]; decision = probability >= 0.5
  <model>_AND  tape verdict CONFIDENT_DEFECT and anomaly score > fold threshold
  <model>_OR   tape verdict CONFIDENT_DEFECT or anomaly score > fold threshold
Single methods: tape_score (continuous), tape_verdict (binary, DET2 close rule), <model> (score > fold threshold).
Adoption rule, fixed before the first run (see CHANGES.md): a candidate is adopted only if, on every colour (each colour
must hold both DEFECT and GOOD crops), it reaches at least the F1 of the tape verdict and of the anomaly model alone, and
for LR at least their AUROC; overall it must beat both strictly; at least 5 DEFECT episodes are required.
Inputs: index.tsv (status ok), eval/scores.tsv, eval/device_check.json, P3-TAPE-EVAL metrics.json.
Outputs: metrics.json, eval/table_methods.tsv, eval/confusion_loeo.tsv, eval/roc_points.tsv, eval/roc_primary.png,
eval/roc_sensitivity.png, eval/crop_table.tsv, eval/fusion_folds.json, eval/replay_fixture.tsv,
models/fusion_params.json (deployment fit on the primary set, validated false).
"""
import csv
import datetime
import json
import os

import cv2
import numpy as np

R = "/home/intel/intel-junction-2026"
E = os.path.join(R, "work", "P4-FUSION", "E1")
EV = os.path.join(E, "eval")
PREV = os.path.join(R, "work", "P3-TAPE-EVAL", "E1", "metrics.json")
MODELS = ("patchcore", "padim")
RULES = ("LR", "AND", "OR")
LABEL_SETS = (("primary", True), ("sensitivity_incl_v1_unverified", False))
N_BOOT = 2000
SEED = 0
RNG = np.random.default_rng(SEED)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def read_tsv(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def write_tsv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh, delimiter="\t")
        wr.writerow(header)
        wr.writerows(rows)


def r4(v):
    return None if v is None else round(float(v), 4)


def auroc(y, s):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    p, n = s[y == 1], s[y == 0]
    if len(p) == 0 or len(n) == 0:
        return None
    wins = (p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()
    return float(wins) / (len(p) * len(n))


def avg_precision(y, s):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    if y.sum() == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    yy = y[order]
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    prec = tp / np.maximum(tp + fp, 1)
    return float(np.sum(prec * yy) / yy.sum())


def binary(y, p):
    y = np.asarray(y, int)
    p = np.asarray(p, int)
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tn = int(np.sum((y == 0) & (p == 0)))
    d = 2 * tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "f1": r4(2 * tp / d) if d else None}


def boot_ci(y, fn):
    """2.5 and 97.5 percentiles of fn(idx) over class-stratified bootstrap resamples of the crops."""
    y = np.asarray(y, int)
    ip = np.where(y == 1)[0]
    ineg = np.where(y == 0)[0]
    if len(ip) == 0 or len(ineg) == 0:
        return [None, None]
    vals = []
    for _ in range(N_BOOT):
        idx = np.concatenate([RNG.choice(ip, len(ip)), RNG.choice(ineg, len(ineg))])
        v = fn(idx)
        if v is not None:
            vals.append(v)
    if len(vals) < 50:
        return [None, None]
    return [r4(np.percentile(vals, 2.5)), r4(np.percentile(vals, 97.5))]


def fit_lr(X, y, iters=3000, lr=0.1, l2=1e-2):
    pos = y.sum()
    neg = len(y) - pos
    sw = np.where(y == 1, len(y) / (2.0 * pos), len(y) / (2.0 * neg))
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        g = sw * (p - y)
        w -= lr * (X.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    return w, b


def feats(rs, m):
    return np.array([[r["tape_score"], r[m + "_score"]] for r in rs], float)


def fit_params(tr, m):
    """Standardisation and LR weights fitted on the rows tr; None if one class is missing."""
    ytr = np.array([r["truth"] for r in tr], float)
    if ytr.sum() == 0 or ytr.sum() == len(ytr):
        return None
    X = feats(tr, m)
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    w, b = fit_lr((X - mu) / sd, ytr)
    return {"mu": [float(v) for v in mu], "sd": [float(v) for v in sd], "w": [float(v) for v in w], "b": float(b)}


def lr_prob(p, X):
    z = (X - np.asarray(p["mu"])) / np.asarray(p["sd"])
    return 1.0 / (1.0 + np.exp(-(z @ np.asarray(p["w"]) + p["b"])))


def build(primary_only):
    idx = {r["rec_id"]: r for r in read_tsv(os.path.join(E, "index.tsv")) if r["status"] == "ok"}
    scores = {m: {} for m in MODELS}
    for r in read_tsv(os.path.join(EV, "scores.tsv")):
        if r["rec_id"] in idx:
            scores[r["model"]][r["rec_id"]] = (float(r["score"]), float(r["threshold"]))
    rows = []
    for rid in sorted(idx):
        r = idx[rid]
        if primary_only and r["truth_status"] == "v1_unverified":
            continue
        if not all(rid in scores[m] for m in MODELS):
            continue
        row = {"rec_id": rid, "group": r["group"], "family": r["family"],
               "truth": 1 if r["truth"] == "DEFECT" else 0, "truth_status": r["truth_status"],
               "tape_score": float(r["tape_score"] or 0.0),
               "tape_defect": 1 if r["tri_verdict"] == "CONFIDENT_DEFECT" else 0,
               "tri_verdict": r["tri_verdict"]}
        for m in MODELS:
            row[m + "_score"], row[m + "_thr"] = scores[m][rid]
        rows.append(row)
    return rows


def fused_loeo(rows, m):
    """Per held-out group: LR fitted on the other groups. Returns probability per rec_id (None where the training
    groups lack one class) and the fitted parameters per group."""
    prob, params = {}, {}
    for h in sorted({r["group"] for r in rows}):
        tr = [r for r in rows if r["group"] != h]
        te = [r for r in rows if r["group"] == h]
        p = fit_params(tr, m)
        params[h] = p
        if p is None:
            for r in te:
                prob[r["rec_id"]] = None
            continue
        for r, v in zip(te, lr_prob(p, feats(te, m))):
            prob[r["rec_id"]] = float(v)
    return prob, params


def method_table(y, n, score=None, pred=None, mask=None, ci=True):
    mask = np.ones(n, bool) if mask is None else mask
    rr = np.where(mask)[0]
    yy = y[rr]
    out = {"n": int(len(rr)), "n_good": int((yy == 0).sum()), "n_defect": int((yy == 1).sum())}
    if score is not None:
        s = score[rr]
        out["auroc"] = r4(auroc(yy, s))
        out["aupr"] = r4(avg_precision(yy, s))
        out["auroc_ci95"] = boot_ci(yy, lambda idx: auroc(yy[idx], s[idx])) if ci else None
    if pred is not None:
        pp = pred[rr]
        b = binary(yy, pp)
        out.update({k: b[k] for k in ("f1", "tp", "fp", "fn", "tn")})
        out["f1_ci95"] = boot_ci(yy, lambda idx: binary(yy[idx], pp[idx])["f1"]) if ci else None
    return out


def evaluate(rows):
    n = len(rows)
    y = np.array([r["truth"] for r in rows], int)
    tape = np.array([r["tape_score"] for r in rows], float)
    tdef = np.array([r["tape_defect"] for r in rows], int)
    anom, fus, folds = {}, {}, {}
    for m in MODELS:
        s = np.array([r[m + "_score"] for r in rows], float)
        th = np.array([r[m + "_thr"] for r in rows], float)
        ap = (s > th).astype(int)
        anom[m] = (s, th, ap)
        prob, par = fused_loeo(rows, m)
        fus[m] = np.array([np.nan if prob[r["rec_id"]] is None else prob[r["rec_id"]] for r in rows], float)
        folds[m] = par
    colours = sorted({r["family"] for r in rows})
    masks = [("overall", np.ones(n, bool))] + [(c, np.array([r["family"] == c for r in rows])) for c in colours]
    tables = {"overall": {}, "by_colour": {}}
    for scope, sm in masks:
        t = {"tape_score": method_table(y, n, score=tape, mask=sm),
             "tape_verdict": method_table(y, n, score=tdef.astype(float), pred=tdef, mask=sm)}
        for m in MODELS:
            s, th, ap = anom[m]
            t[m] = method_table(y, n, score=s, pred=ap, mask=sm)
            for rule in RULES:
                if rule == "LR":
                    fm = sm & ~np.isnan(fus[m])
                    score = np.nan_to_num(fus[m])
                    pred = (score >= 0.5).astype(int)
                elif rule == "AND":
                    fm = sm
                    score = None
                    pred = (tdef.astype(bool) & ap.astype(bool)).astype(int)
                else:
                    fm = sm
                    score = None
                    pred = (tdef.astype(bool) | ap.astype(bool)).astype(int)
                cand = method_table(y, n, score=score, pred=pred, mask=fm)
                cand["paired"] = {  # the single methods on exactly the crops the candidate decides
                    "tape_verdict_f1": method_table(y, n, pred=tdef, mask=fm, ci=False)["f1"],
                    "anomaly_f1": method_table(y, n, pred=ap, mask=fm, ci=False)["f1"],
                    "tape_score_auroc": method_table(y, n, score=tape, mask=fm, ci=False)["auroc"],
                    "anomaly_auroc": method_table(y, n, score=s, mask=fm, ci=False)["auroc"],
                }
                t["%s_%s" % (m, rule)] = cand
        if scope == "overall":
            tables["overall"] = t
        else:
            tables["by_colour"][scope] = t
    arr = {"y": y, "tape": tape, "tdef": tdef, "anom": anom, "fus": fus, "folds": folds}
    return tables, arr


def _cmp(label, cand, singles, strict):
    vals = [v for v in singles if v is not None]
    if cand is None or not vals:
        return "%s not defined" % label
    best = max(vals)
    if cand < best or (strict and cand == best):
        return "%s %.4f vs best single %.4f" % (label, cand, best)
    return None


def adoption(tables, n_def_eps, assessable):
    o = tables["overall"]
    failures = {}
    for m in MODELS:
        for rule in RULES:
            name = "%s_%s" % (m, rule)
            fails = []
            c = o[name]
            p = c["paired"]
            e = _cmp("overall F1", c.get("f1"), [p["tape_verdict_f1"], p["anomaly_f1"]], True)
            if e:
                fails.append(e)
            if rule == "LR":
                e = _cmp("overall AUROC", c.get("auroc"), [p["tape_score_auroc"], p["anomaly_auroc"]], True)
                if e:
                    fails.append(e)
            for col, t in sorted(tables["by_colour"].items()):
                if col not in assessable:
                    fails.append("%s not assessable (no DEFECT or no GOOD crops)" % col)
                    continue
                cc = t[name]
                pp = cc["paired"]
                e = _cmp("%s F1" % col, cc.get("f1"), [pp["tape_verdict_f1"], pp["anomaly_f1"]], False)
                if e:
                    fails.append(e)
                if rule == "LR":
                    e = _cmp("%s AUROC" % col, cc.get("auroc"), [pp["tape_score_auroc"], pp["anomaly_auroc"]], False)
                    if e:
                        fails.append(e)
            failures[name] = fails
    passing = sorted(k for k, v in failures.items() if not v)
    reasons = []
    if n_def_eps < 5:
        reasons.append("only %d DEFECT episodes (at least 5 required)" % n_def_eps)
    if not passing:
        reasons.append("no candidate passes every check")
    decision = "ADOPT" if (passing and n_def_eps >= 5) else "NOT ADOPTED"
    return {"decision": decision, "passing_candidates": passing, "candidate_failures": failures, "reasons": reasons}


def best_candidate(o):
    """Fused candidate with the highest overall F1 (ties: AUROC). Reported for the patch default; not an adoption."""
    names = ["%s_%s" % (m, rule) for m in MODELS for rule in RULES]

    def key(k):
        f1 = o[k].get("f1")
        au = o[k].get("auroc")
        return (-1.0 if f1 is None else f1, -1.0 if au is None else au)

    return max(names, key=key)


def roc(y, s):
    order = np.argsort(-s, kind="mergesort")
    yy = y[order]
    ss = s[order]
    P = max(int(yy.sum()), 1)
    N = max(int(len(yy) - yy.sum()), 1)
    tps = np.cumsum(yy)
    fps = np.cumsum(1 - yy)
    keep = np.r_[np.where(np.diff(ss) != 0)[0], len(ss) - 1]
    return np.r_[0.0, fps[keep] / N], np.r_[0.0, tps[keep] / P]


def curves_for(arr):
    y = arr["y"]
    everyone = np.ones(len(y), bool)
    specs = [("tape score", arr["tape"], everyone)]
    for m in MODELS:
        specs.append((m, arr["anom"][m][0], everyone))
        specs.append((m + " LR", np.nan_to_num(arr["fus"][m]), ~np.isnan(arr["fus"][m])))
    out = []
    for name, s, mk in specs:
        if len(set(y[mk].tolist())) < 2:
            continue
        fpr, tpr = roc(y[mk], s[mk])
        out.append((name, fpr, tpr, float(TRAPZ(tpr, fpr))))
    return out


def plot(curves, path, title):
    W, H, M = 620, 600, 70
    x0, x1, y0, y1 = M, W - 30, H - M, M
    img = np.full((H, W, 3), 255, np.uint8)
    cv2.rectangle(img, (x0, y1), (x1, y0), (0, 0, 0), 1)
    cv2.line(img, (x0, y0), (x1, y1), (180, 180, 180), 1)
    for k in np.linspace(0, 1, 6):
        x = int(x0 + k * (x1 - x0))
        y = int(y0 - k * (y0 - y1))
        cv2.line(img, (x, y0), (x, y0 + 4), (0, 0, 0), 1)
        cv2.line(img, (x0 - 4, y), (x0, y), (0, 0, 0), 1)
        cv2.putText(img, "%.1f" % k, (x - 10, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(img, "%.1f" % k, (x0 - 36, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(img, "FPR", (W // 2 - 12, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "TPR", (6, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, title, (M, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    colours = [(0, 0, 200), (200, 80, 0), (0, 150, 0), (150, 0, 150), (0, 150, 200), (60, 60, 60)]
    for i, (name, fpr, tpr, auc) in enumerate(curves):
        pts = np.array([[x0 + f * (x1 - x0), y0 - t * (y0 - y1)] for f, t in zip(fpr, tpr)], np.int32)
        cv2.polylines(img, [pts], False, colours[i % len(colours)], 2)
        txt = "%s AUROC %s" % (name, "n/a" if auc is None else "%.3f" % auc)
        cv2.putText(img, txt, (x0 + 230, y0 - 14 - 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    colours[i % len(colours)], 1)
    cv2.imwrite(path, img)


def run_label_set(label, primary):
    rows = build(primary)
    tables, arr = evaluate(rows)
    def_groups = {r["group"] for r in rows if r["truth"] == 1}
    good_groups = {r["group"] for r in rows if r["truth"] == 0}
    def_cols = {r["family"] for r in rows if r["truth"] == 1}
    good_cols = {r["family"] for r in rows if r["truth"] == 0}
    assessable = sorted(def_cols & good_cols)
    colours = {c: {"n_good": sum(1 for r in rows if r["family"] == c and r["truth"] == 0),
                   "n_defect": sum(1 for r in rows if r["family"] == c and r["truth"] == 1)}
               for c in sorted({r["family"] for r in rows})}
    summary = {"n_crops": len(rows), "n_good_crops": int((arr["y"] == 0).sum()),
               "n_defect_crops": int((arr["y"] == 1).sum()), "n_good_episodes": len(good_groups),
               "n_defect_episodes": len(def_groups), "defect_episodes": sorted(def_groups),
               "colours": colours, "colours_assessable": assessable,
               "adoption": adoption(tables, len(def_groups), assessable), "tables": tables}
    return rows, arr, summary


def main():
    os.makedirs(EV, exist_ok=True)
    os.makedirs(os.path.join(E, "models"), exist_ok=True)
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    res = {label: run_label_set(label, primary) for label, primary in LABEL_SETS}
    prim_rows, prim_arr, prim_sum = res["primary"]
    sens_rows, sens_arr, sens_sum = res["sensitivity_incl_v1_unverified"]

    pts = []
    for name, arr, png in (("primary", prim_arr, "roc_primary.png"),
                           ("sensitivity", sens_arr, "roc_sensitivity.png")):
        cs = curves_for(arr)
        plot(cs, os.path.join(EV, png), "fusion-v1 ROC, %s set (LOEO)" % name)
        for cname, fpr, tpr, _ in cs:
            for f, t in zip(fpr, tpr):
                pts.append([name, cname, "%.4f" % f, "%.4f" % t])
    write_tsv(os.path.join(EV, "roc_points.tsv"), ["set", "method", "fpr", "tpr"], pts)

    crop_cols = ["rec_id", "group", "family", "truth", "truth_status", "in_primary", "tri_verdict", "tape_score"]
    for m in MODELS:
        crop_cols += [m + "_score", m + "_thr", m + "_LR_prob", m + "_LR", m + "_AND", m + "_OR"]
    fix_cols = ["rec_id", "group", "family", "truth", "tri_verdict", "tape_score", "patchcore_score",
                "patchcore_thr", "padim_score", "padim_thr", "patchcore_LR", "patchcore_AND", "patchcore_OR",
                "padim_LR", "padim_AND", "padim_OR"]
    crop_rows, fix_rows = [], []
    for i, r in enumerate(sens_rows):
        vals = {}
        for m in MODELS:
            s, th, ap = sens_arr["anom"][m]
            p = sens_arr["fus"][m][i]
            nan = bool(np.isnan(p))
            vals[m] = {"score": "%.5f" % s[i], "thr": "%.5f" % th[i],
                       "prob": "" if nan else "%.6f" % p, "lr": "" if nan else int(p >= 0.5),
                       "and": int(sens_arr["tdef"][i] and ap[i]), "or": int(sens_arr["tdef"][i] or ap[i])}
        crop_rows.append([r["rec_id"], r["group"], r["family"], r["truth"], r["truth_status"],
                          int(r["truth_status"] != "v1_unverified"), r["tri_verdict"], "%.6f" % r["tape_score"]]
                         + [vals[m][k] for m in MODELS for k in ("score", "thr", "prob", "lr", "and", "or")])
        fix_rows.append([r["rec_id"], r["group"], r["family"], r["truth"], r["tri_verdict"],
                         repr(float(r["tape_score"])), vals["patchcore"]["score"], vals["patchcore"]["thr"],
                         vals["padim"]["score"], vals["padim"]["thr"], vals["patchcore"]["lr"],
                         vals["patchcore"]["and"], vals["patchcore"]["or"], vals["padim"]["lr"],
                         vals["padim"]["and"], vals["padim"]["or"]])
    write_tsv(os.path.join(EV, "crop_table.tsv"), crop_cols, crop_rows)
    write_tsv(os.path.join(EV, "replay_fixture.tsv"), fix_cols, fix_rows)

    folds_out = {"stage": "fusion-v1", "sets": {
        "primary": {m: prim_arr["folds"][m] for m in MODELS},
        "sensitivity_incl_v1_unverified": {m: sens_arr["folds"][m] for m in MODELS}}}
    with open(os.path.join(EV, "fusion_folds.json"), "w", encoding="utf-8") as fh:
        json.dump(folds_out, fh, indent=1)
    dep = {m: fit_params(prim_rows, m) for m in MODELS}
    with open(os.path.join(E, "models", "fusion_params.json"), "w", encoding="utf-8") as fh:
        json.dump({"stage": "fusion-v1", "validated": False, "features": ["tape_score", "anomaly_score"],
                   "decision_threshold": 0.5,
                   "fit_on": "primary set (verified labels only), out-of-fold anomaly scores",
                   "n_fit": len(prim_rows), "n_defect_fit": int(prim_arr["y"].sum()), "models": dep},
                  fh, indent=1)

    before_after = {}
    for label, s in (("primary", prim_sum), ("sensitivity_incl_v1_unverified", sens_sum)):
        o = s["tables"]["overall"]
        before_after[label] = {"before_tape_verdict": o["tape_verdict"], "before_tape_score": o["tape_score"],
                               "before_patchcore": o["patchcore"], "before_padim": o["padim"],
                               "after": {"%s_%s" % (m, rule): o["%s_%s" % (m, rule)]
                                         for m in MODELS for rule in RULES}}

    dc = None
    dc_path = os.path.join(EV, "device_check.json")
    if os.path.exists(dc_path):
        with open(dc_path, encoding="utf-8") as fh:
            dc = json.load(fh)
    with open(PREV, encoding="utf-8") as fh:
        prev = json.load(fh)

    out = {
        "stage": "fusion-v1",
        "utc": now,
        "scope": "32 close-view brick crops (index status ok after the eye check, 14 excluded). primary = verified "
                 "labels only; sensitivity also keeps the unverified DEFECT (any_1740 ep_0017). LOEO over episodes "
                 "(Studio: participant groups).",
        "method": {"n_boot": N_BOOT, "seed": SEED,
                   "ci": "95% percentile, class-stratified bootstrap over crops (crops of one episode are "
                         "correlated, so intervals are optimistic)",
                   "anomaly_threshold": "fold threshold = max score of the fold's training GOOD crops",
                   "fusion_fit": "LR on the training folds only, standardised, class-balanced, L2 1e-2, 3000 steps"},
        "adoption_rule": "candidate adopted only if, on every colour (each with DEFECT and GOOD crops), F1 >= best "
                         "single (tape verdict, anomaly model) on the same crops and, for LR, AUROC >= best single "
                         "score; overall F1 (and AUROC for LR) strictly above; at least 5 DEFECT episodes",
        "decision": {label: s["adoption"]["decision"] for label, s in
                     (("primary", prim_sum), ("sensitivity_incl_v1_unverified", sens_sum))},
        "best_candidate_overall": {label: best_candidate(s["tables"]["overall"]) for label, s in
                                   (("primary", prim_sum), ("sensitivity_incl_v1_unverified", sens_sum))},
        "label_sets": {label: {k: v for k, v in s.items() if k != "tables"}
                       for label, s in (("primary", prim_sum), ("sensitivity_incl_v1_unverified", sens_sum))},
        "tables": {"primary": prim_sum["tables"], "sensitivity_incl_v1_unverified": sens_sum["tables"]},
        "before_after": before_after,
        "device_check": dc,
        "previous": {"stage": prev.get("stage"), "utc": prev.get("utc"), "scope": prev.get("scope"),
                     "metrics": prev.get("metrics"), "source": PREV},
        "delta_note": "before/after on the same crops is before_after (DET2 tape verdict and each Anomalib model alone "
                      "vs each fused candidate). The tape-eval-v1 numbers are a different view set (23 OURS close "
                      "frames, loop spot selection), so they are not differenced.",
    }
    with open(os.path.join(E, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    trows, crows = [], []
    for label, s in (("primary", prim_sum), ("sensitivity_incl_v1_unverified", sens_sum)):
        for scope, t in [("overall", s["tables"]["overall"])] + sorted(s["tables"]["by_colour"].items()):
            for name, v in t.items():
                trows.append([label, scope, name, v["n"], v["n_good"], v["n_defect"], v.get("auroc"),
                              json.dumps(v.get("auroc_ci95")), v.get("aupr"), v.get("f1"),
                              json.dumps(v.get("f1_ci95")), v.get("tp"), v.get("fp"), v.get("fn"), v.get("tn")])
            for name in ("tape_verdict", "patchcore", "padim") + tuple("%s_%s" % (m, r) for m in MODELS for r in RULES):
                v = t[name]
                crows.append([label, scope, name, v["n"], v["n_defect"], v["n_good"], v.get("tp"), v.get("fn"),
                              v.get("fp"), v.get("tn"), v.get("f1")])
    write_tsv(os.path.join(EV, "table_methods.tsv"),
              ["label_set", "scope", "method", "n", "n_good", "n_defect", "auroc", "auroc_ci95", "aupr", "f1",
               "f1_ci95", "tp", "fp", "fn", "tn"], trows)
    write_tsv(os.path.join(EV, "confusion_loeo.tsv"),
              ["label_set", "scope", "method", "n", "n_defect", "n_good", "tp", "fn", "fp", "tn", "f1"], crows)

    for label, s in (("primary", prim_sum), ("sensitivity_incl_v1_unverified", sens_sum)):
        print(label, "crops", s["n_crops"], "good", s["n_good_crops"], "defect", s["n_defect_crops"],
              "defect_eps", s["defect_episodes"], "assessable", s["colours_assessable"], flush=True)
        print("  decision", s["adoption"]["decision"], "passing", s["adoption"]["passing_candidates"],
              s["adoption"]["reasons"], flush=True)
        for name in ("tape_score", "tape_verdict", "patchcore", "padim", "patchcore_LR", "patchcore_AND",
                     "patchcore_OR", "padim_LR", "padim_AND", "padim_OR"):
            print("  OVERALL", name, json.dumps(s["tables"]["overall"][name]), flush=True)
    print("WROTE", os.path.join(E, "metrics.json"), flush=True)


if __name__ == "__main__":
    main()
