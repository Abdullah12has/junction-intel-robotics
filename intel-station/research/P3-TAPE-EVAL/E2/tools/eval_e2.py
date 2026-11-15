#!/usr/bin/env python3
"""P3-TAPE-EVAL E2 harness. Runs the robot's own decision functions offline.

usage: python3 -I eval_e2.py <APP_ROOT> <OUT_DIR> [variant]
  APP_ROOT: R/app (baseline) or the patched copy under R/work/P3-TAPE-EVAL/E2/app
  variant : 'base' (as loaded) or 'p1' (the copy's sort_loop.close_verdict with the jaw-anchored fallback)
Modules are imported from APP_ROOT (sort_loop and sort_any from APP_ROOT/tools; inspector from APP_ROOT/anomaly/det2_f2).
Outputs: views.tsv (every view and its verdict), routing.tsv (every episode and its plate), studio.tsv, metrics.json.
Ground truth:
  ours    : P3-LABEL-OURS/E1/labels_images.jsonl (close views), labels_episodes.jsonl (plate, shape, tape)
  studio  : P3-LABEL-STUDIO/E1/labels_episodes.jsonl (target colour, tape, plate), frames in P3-LABEL-STUDIO/E1/frames
"""
import sys, os, json, math, collections

APP = sys.argv[1]
OUT = sys.argv[2]
VARIANT = sys.argv[3] if len(sys.argv) > 3 else "base"
R = "/home/intel/intel-junction-2026"
LOGS = f"{R}/logs/sort"
WORK = f"{R}/work"

import importlib.util
import cv2
import numpy as np


def _load(name, path):
    """Load a module from an explicit file path (so APP's own code runs, not R/app's)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, f"{APP}/tools")
sys.path.insert(0, f"{APP}/anomaly/det2_f2")
DET = _load("inspector", f"{APP}/anomaly/det2_f2/inspector.py")   # sort_loop/sort_any take this via sys.modules
sl = _load("sort_loop", f"{APP}/tools/sort_loop.py")
sa = _load("sort_any", f"{APP}/tools/sort_any.py")
pa = sys.modules["pick_any"]
kin = sys.modules["kin"]
assert os.path.abspath(sl.__file__).startswith(APP), sl.__file__
assert os.path.abspath(DET.__file__).startswith(APP), DET.__file__

os.makedirs(OUT, exist_ok=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def conf_table(rows, key_truth, key_pred):
    m = {t: {p: 0 for p in ("yes", "no", "uncertain")} for t in ("yes", "no")}
    for r in rows:
        if r[key_truth] in m:
            m[r[key_truth]][r[key_pred]] = m[r[key_truth]].get(r[key_pred], 0) + 1
    dec = sum(m[t][p] for t in m for p in ("yes", "no"))
    ok = m["yes"]["yes"] + m["no"]["no"]
    return {"matrix": m, "n": len(rows), "decided": dec, "correct": ok,
            "accuracy_decided": round(ok / dec, 4) if dec else None, "accuracy_ci95": wilson(ok, dec),
            "uncertain": sum(m[t]["uncertain"] for t in m)}


def pred_of(verdict):
    return {"CONFIDENT_DEFECT": "yes", "CONFIDENT_GOOD": "no"}.get(verdict, "uncertain")


# ---------------- A. close-view tape (ours) ----------------
LAB_IMG = f"{WORK}/P3-LABEL-OURS/E1/labels_images.jsonl"
KINDS = {"pre", "station", "closed", "lifted"}
views = []
for l in open(LAB_IMG):
    r = json.loads(l)
    if r["kind"] not in KINDS or not r["image_path"].endswith("_close.jpg"):
        continue
    s = r["shows"].lower()
    truth = "no" if "no tape" in s else ("yes" if "with tape" in s else None)
    if truth is None:
        continue
    path = r["image_path"]
    if r["kind"] == "pre":
        path = os.path.join(os.path.dirname(path), "crops", "pre_close_full.jpg")
        if not os.path.exists(path):
            continue
    img = cv2.imread(path)
    if img is None:
        continue
    if r["kind"] == "pre":
        res = sa.close_pre_verdict(img) or {"verdict": "NONE", "reasons": ["no jaws"]}
        fn = "close_pre_verdict"
    else:
        res = sl.close_verdict(img)
        fn = "close_verdict"
    views.append({"set": "ours", "file": path, "episode": r["episode_id"], "kind": r["kind"], "colour": r["brick"],
                  "truth": truth, "fn": fn, "verdict": res.get("verdict"), "pred": pred_of(res.get("verdict")),
                  "tape_score": res.get("tape_score"), "reasons": ";".join(map(str, res.get("reasons", []))),
                  "bbox": res.get("bbox")})

# ---------------- B. routing + shape (ours), replicating the loop's cycle() order ----------------
rows_ep = [json.loads(l) for l in open(f"{WORK}/P3-LABEL-OURS/E1/labels_episodes.jsonl")]
routing = []
for r in rows_ep:
    ep_dir = r["source_path"]
    wf = f"{ep_dir}/crops/wrist_hover.png"
    if not os.path.exists(wf):
        continue
    if not os.path.exists(f"{ep_dir}/episode.json"):
        routing.append({"episode": r["episode_id"], "truth_plate": r.get("correct_plate_od10"), "status": "no_episode_json"})
        continue
    ej = json.load(open(f"{ep_dir}/episode.json"))
    img_w = cv2.imread(wf)
    bl = pa.blobs(img_w, min_area=pa.SERVO_MIN_AREA) if hasattr(pa, "SERVO_MIN_AREA") else pa.blobs(img_w)
    bh = pa.nearest(bl, pa.target(), max_d=400)
    rec = {"episode": r["episode_id"], "truth_plate": r.get("correct_plate_od10"),
           "truth_shape": None, "pred_shape": None, "pred_plate": None, "status": "", "reasons": ""}
    tb = r["bricks_involved"][r["target_brick_index"]] if (r.get("target_brick_index") is not None and r.get("bricks_involved")) else None
    if tb:
        rec["truth_shape"] = tb.get("shape")
    if bh is None:
        rec["status"] = "no_hover_blob"
        routing.append(rec)
        continue
    # shape: same expression as sort_any.grasp() (inline in that function, not importable without side effects)
    q = ej.get("hover_q")
    tip = pa.tip_from_size(bh["short"])
    fk_tip = float(kin.fk(q[:5])[0][2] - kin.MAT_Z) if q else tip
    use = tip if abs(tip - fk_tip) <= 15 else fk_tip
    if bh["aspect"] > 1.4:
        shape = ("not_2x2", f"aspect {bh['aspect']}")
    elif bh["short"] < 0.72 * pa.SIZE_K / (use + pa.SIZE_C):
        shape = ("not_2x2", f"narrow short {bh['short']}px")
    else:
        shape = ("2x2", f"aspect {bh['aspect']} short {bh['short']}")
    rec["pred_shape"] = shape[0]
    pre_path = f"{ep_dir}/crops/pre_close_full.jpg"
    pre = sa.close_pre_verdict(cv2.imread(pre_path), want_hue=bh.get("hue")) if os.path.exists(pre_path) else None
    pre = pre or {}
    clean = []
    station_path = f"{ep_dir}/station_close.jpg"
    if shape[0] == "2x2" and pre.get("verdict") not in ("CONFIDENT_DEFECT", "CONFIDENT_GOOD"):
        if not os.path.exists(station_path):
            rec["status"] = "cannot_replicate_no_station_view"
            routing.append(rec)
            continue
        sv = sl.close_verdict(cv2.imread(station_path))
        clean = sa.clean_views([sv])
        rec["station_verdict"] = sv.get("verdict")
    plate, label, conf, reasons = sl.decide(pre, shape, clean)
    rec.update(pred_plate=("blue" if plate == "blue" else "green"), status="ok", reasons=";".join(map(str, reasons)),
               pre_verdict=pre.get("verdict"), label=label, conf=conf)
    routing.append(rec)

# ---------------- C. Studio close frames (per-brick tape rule, target chosen by labelled colour) ----------------
st_lab = [json.loads(l) for l in open(f"{WORK}/P3-LABEL-STUDIO/E1/labels_episodes.jsonl")]
unit_entry = {}
for e in st_lab:
    if e.get("unit_group") and e.get("entry") == e.get("representative_entry"):
        if e["unit_group"] in unit_entry:
            raise SystemExit("duplicate unit " + e["unit_group"])
        unit_entry[e["unit_group"]] = e
FR = f"{WORK}/P3-LABEL-STUDIO/E1/frames"
studio = []
thr = DET.load_thresholds()
for fn_ in sorted(os.listdir(FR)):
    if "_064316_" not in fn_:
        continue                     # 064316 = close serial 243722064316 (ours); 061421 = top
    unit, _, stage = fn_.split("_")[0], fn_.split("_")[1], fn_.split("_")[2]
    e = unit_entry.get(unit)
    if e is None or e.get("trust") not in ("training", "evaluation", "calibration"):
        continue
    tb = e.get("target_brick") or {}
    tcol = tb.get("colour")
    truth = tb.get("tape")
    if truth not in ("yes", "no") or tcol in (None, "none"):
        continue
    img = cv2.imread(os.path.join(FR, fn_))
    if img is None:
        continue
    bricks = DET.segment_close(sl.mask_jaws(img))["bricks"]
    cand = [b for b in bricks if b["colour"] == tcol and b["area"] >= 1500]
    if len(cand) != 1:
        studio.append({"set": "studio", "file": fn_, "unit": unit, "entry": e["entry"], "stage": stage, "colour": tcol,
                       "truth": truth, "verdict": "NOT_LOCATED", "pred": "uncertain", "reasons": f"{len(cand)} target-colour blobs"})
        continue
    b = cand[0]
    verdict, reasons = DET.triage(b, thr)
    studio.append({"set": "studio", "file": fn_, "unit": unit, "entry": e["entry"], "stage": stage, "colour": tcol,
                   "truth": truth, "verdict": verdict, "pred": pred_of(verdict), "reasons": ";".join(reasons),
                   "tape_score": b.get("tape_score"), "bbox": b["bbox"], "dist_to_spot": None})

# ---------------- write ----------------
def dump(rows, name, cols):
    with open(os.path.join(OUT, name), "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


dump(views, "views.tsv", ["set", "file", "episode", "kind", "colour", "truth", "fn", "verdict", "pred", "tape_score", "reasons", "bbox"])
dump(routing, "routing.tsv", ["episode", "truth_plate", "truth_shape", "pred_shape", "pred_plate", "status", "pre_verdict",
                              "station_verdict", "label", "conf", "reasons"])
dump(studio, "studio.tsv", ["set", "file", "unit", "entry", "stage", "colour", "truth", "verdict", "pred", "tape_score", "reasons", "bbox"])

rt_ok = [r for r in routing if r["status"] == "ok" and r["truth_plate"] in ("blue", "green")]
rt_hit = [r for r in rt_ok if r["pred_plate"] == r["truth_plate"]]
sh_ok = [r for r in routing if r["status"] == "ok" and r["truth_shape"]]
sh_hit = [r for r in sh_ok if (r["pred_shape"] == "2x2") == (r["truth_shape"] == "2x2")]
M = {"variant": VARIANT, "app_root": APP, "utc": None,
     "tape_ours_close": {"overall": conf_table(views, "truth", "pred"),
                         "by_kind": {k: conf_table([v for v in views if v["kind"] == k], "truth", "pred")
                                     for k in sorted(KINDS)},
                         "by_colour": {c: conf_table([v for v in views if v["colour"] == c], "truth", "pred")
                                       for c in sorted({v["colour"] for v in views})}},
     "tape_studio_close_per_brick": {"overall": conf_table(studio, "truth", "pred"),
                                     "not_located": sum(1 for s in studio if s["verdict"] == "NOT_LOCATED"),
                                     "by_stage": {s: conf_table([x for x in studio if x["stage"] == s], "truth", "pred")
                                                  for s in sorted({x["stage"] for x in studio})}},
     "routing_ours": {"episodes_with_wrist": len(routing), "replicated": len(rt_ok) + len([r for r in routing if r["status"] == "ok" and r["truth_plate"] not in ("blue", "green")]),
                      "truth_known_and_replicated": len(rt_ok), "correct": len(rt_hit),
                      "accuracy": round(len(rt_hit) / len(rt_ok), 4) if rt_ok else None,
                      "accuracy_ci95": wilson(len(rt_hit), len(rt_ok)),
                      "not_replicable": [r["episode"] for r in routing if r["status"] != "ok"]},
     "shape_ours_2x2_vs_not": {"n": len(sh_ok), "correct": len(sh_hit),
                               "accuracy": round(len(sh_hit) / len(sh_ok), 4) if sh_ok else None,
                               "accuracy_ci95": wilson(len(sh_hit), len(sh_ok))}}
json.dump(M, open(os.path.join(OUT, "metrics.json"), "w"), indent=1, default=str)
print(json.dumps({"variant": VARIANT, "ours_close_n": len(views), "ours_close_decided": M["tape_ours_close"]["overall"]["decided"],
                  "ours_close_correct": M["tape_ours_close"]["overall"]["correct"], "studio_n": len(studio),
                  "studio_decided": M["tape_studio_close_per_brick"]["overall"]["decided"],
                  "studio_correct": M["tape_studio_close_per_brick"]["overall"]["correct"],
                  "routing_n": M["routing_ours"]["truth_known_and_replicated"], "routing_correct": M["routing_ours"]["correct"],
                  "shape_n": M["shape_ours_2x2_vs_not"]["n"], "shape_correct": M["shape_ours_2x2_vs_not"]["correct"]}, default=str))
