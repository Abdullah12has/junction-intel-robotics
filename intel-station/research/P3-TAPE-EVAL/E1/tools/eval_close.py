#!/usr/bin/env python3
"""P3-TAPE-EVAL E1: offline run of the robot's own close-view tape decision on labelled close frames.

Imports (does not copy) sort_loop.close_verdict and sort_any.close_pre_verdict, the functions the loop calls.
Truth: P3-LABEL-OURS labels_images.jsonl (shows: 'no tape' / 'with tape'). Frames read only from R/logs/sort.
Writes eval_set.tsv, results.tsv, confusion.json, metrics.json, gallery/ into the E1 folder.
usage: python3 -I eval_close.py <E1 folder>
"""
import sys, os, json, math, collections

R = "/home/intel/intel-junction-2026"
E1 = sys.argv[1]
sys.path.insert(0, f"{R}/app/tools")
sys.path.insert(0, f"{R}/app/anomaly/det2_f2")

import cv2
import numpy as np
import sort_loop as sl          # close_verdict, mask_jaws (imports grasp_auto, inspector)
import inspector as DET2        # same module object the loop uses

try:
    import sort_any as sa       # close_pre_verdict (pre-grasp rule)
    PRE_ERR = ""
except Exception as e:          # noqa: BLE001
    sa = None
    PRE_ERR = repr(e)

LAB = f"{R}/work/P3-LABEL-OURS/E1/labels_images.jsonl"
KINDS = {"pre", "station", "closed", "lifted"}   # views on which the loop calls a tape verdict


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


rows = [json.loads(l) for l in open(LAB)]
evalset = []
pre_missing = []
for r in rows:
    if r["kind"] not in KINDS or not r["image_path"].endswith("_close.jpg"):
        continue
    s = r["shows"].lower()
    if "no tape" in s:
        truth = "no"
    elif "with tape" in s:
        truth = "yes"
    else:
        continue                             # 'target unclear': no truth, excluded
    inferred = "inferred" in s
    path = r["image_path"]
    if r["kind"] == "pre":
        # the loop ran close_pre_verdict on the hover frame, saved as crops/pre_close_full.jpg (only when it had a bbox);
        # the episode's pre_close.jpg is the scan-pose snapshot and is NOT the loop input
        path = os.path.join(os.path.dirname(r["image_path"]), "crops", "pre_close_full.jpg")
        if not os.path.exists(path):
            pre_missing.append(r["episode_id"])
            continue
    evalset.append({"file": path, "episode": r["episode_id"], "kind": r["kind"],
                    "colour": r["brick"], "truth_tape": truth, "truth_note": "inferred" if inferred else "visual",
                    "source": "P3-LABEL-OURS/E1/labels_images.jsonl"})

results = []
for v in evalset:
    img = cv2.imread(v["file"])
    if img is None:
        v.update(verdict="MISSING", reasons=["unreadable"], pred="uncertain")
        results.append(v)
        continue
    if v["kind"] == "pre":
        if sa is None:
            res = {"verdict": "NOT_RUN", "reasons": ["sort_any import failed: " + PRE_ERR]}
        else:
            res = sa.close_pre_verdict(img) or {"verdict": "NONE", "reasons": ["no jaws in close view"]}
        fn = "sort_any.close_pre_verdict"
    else:
        res = sl.close_verdict(img)
        fn = "sort_loop.close_verdict"
    verdict = res.get("verdict")
    pred = {"CONFIDENT_DEFECT": "yes", "CONFIDENT_GOOD": "no"}.get(verdict, "uncertain")
    v.update(fn=fn, verdict=verdict, reasons=";".join(str(x) for x in res.get("reasons", [])),
             tape_score=res.get("tape_score"), colour_pred=res.get("colour"), dist=res.get("dist"),
             bbox=res.get("bbox"), pred=pred)
    results.append(v)

# error gallery: crop around the brick box the loop used
gal = os.path.join(E1, "gallery")
os.makedirs(gal, exist_ok=True)
errs = []
for i, v in enumerate(results):
    if v.get("pred") in ("yes", "no") and v["pred"] != v["truth_tape"]:
        errs.append(v)
        img = cv2.imread(v["file"])
        bb = v.get("bbox")
        if img is not None and bb:
            x, y, w, h = bb
            crop = img[max(0, y - 12):y + h + 12, max(0, x - 12):x + w + 12]
            crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC) if crop.size else crop
            cv2.imwrite(os.path.join(gal, f"err_{len(errs):03d}_{v['kind']}_{v['colour']}_truth{v['truth_tape']}.png"), crop)
        else:
            v["gallery"] = "no bbox"

with open(os.path.join(E1, "eval_set.tsv"), "w") as f:
    f.write("file\tepisode\tkind\tcolour\ttruth_tape\ttruth_note\tsource\n")
    for v in evalset:
        f.write("\t".join(str(v[k]) for k in ("file", "episode", "kind", "colour", "truth_tape", "truth_note", "source")) + "\n")

cols = ["file", "episode", "kind", "colour", "truth_tape", "truth_note", "fn", "verdict", "pred", "tape_score",
        "colour_pred", "dist", "bbox", "reasons"]
with open(os.path.join(E1, "results.tsv"), "w") as f:
    f.write("\t".join(cols) + "\n")
    for v in results:
        f.write("\t".join(str(v.get(c, "")) for c in cols) + "\n")


def confusion(sub):
    c = {t: {p: 0 for p in ("yes", "no", "uncertain")} for t in ("yes", "no")}
    for v in sub:
        c[v["truth_tape"]][v.get("pred", "uncertain")] += 1
    dec = c["yes"]["yes"] + c["yes"]["no"] + c["no"]["yes"] + c["no"]["no"]
    ok = c["yes"]["yes"] + c["no"]["no"]
    return {"matrix": c, "n": len(sub), "decided": dec, "correct": ok,
            "accuracy_decided": round(ok / dec, 4) if dec else None,
            "accuracy_ci95": wilson(ok, dec), "uncertain": c["yes"]["uncertain"] + c["no"]["uncertain"]}


by = collections.defaultdict(list)
out = {"overall": confusion(results), "by_colour": {}, "by_kind": {}, "by_truth_note": {}}
for v in results:
    by[("colour", v["colour"])].append(v)
    by[("kind", v["kind"])].append(v)
    by[("note", v["truth_note"])].append(v)
for (g, k), sub in sorted(by.items()):
    key = {"colour": "by_colour", "kind": "by_kind", "note": "by_truth_note"}[g]
    out[key][k] = confusion(sub)
out["errors"] = [{"file": v["file"], "kind": v["kind"], "colour": v["colour"], "truth": v["truth_tape"],
                  "pred": v["pred"], "tape_score": v.get("tape_score"), "reasons": v.get("reasons")} for v in errs]
out["sort_any_import"] = "ok" if sa is not None else PRE_ERR
out["pre_excluded_no_loop_input_saved"] = pre_missing
json.dump(out, open(os.path.join(E1, "confusion.json"), "w"), indent=1, default=str)
print(json.dumps({"n": len(results), "overall": out["overall"], "sort_any": out["sort_any_import"],
                  "errors": len(errs)}, default=str))
