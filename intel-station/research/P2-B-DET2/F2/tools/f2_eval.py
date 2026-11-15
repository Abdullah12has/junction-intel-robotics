"""P2-B-DET2 F2 evaluation (analysis only, not part of the app).
usage: python3 -I -B f2_eval.py APP_DIR OUT_JSON MAP_JSON ['{"CONST": value}']
Scans 16 dataset frame pairs (6d6bdce3 e03-e15, start and end) as raw PNG and as JPEG q80, scores the Opus close labels
(keys of compare_opus_vs_E2.tsv) through MAP_JSON (created once from the baseline raw run), runs the live frames.
A CONST override sets a module constant of APP_DIR/inspector.py; the key THR replaces the tape thresholds.
"""
import json
import os
import sys

import cv2
import numpy as np

APP, OUT, MAPP = sys.argv[1], sys.argv[2], sys.argv[3]
OVR = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
sys.path.insert(0, APP)
import inspector as I  # noqa: E402

for name, val in OVR.items():
    if name == "THR":
        I.load_thresholds = (lambda t: (lambda: t))(val)
    else:
        setattr(I, name, val)

E2 = "/home/intel/intel-junction-2026/work/P2-B-DET2/E2"
DET = "/home/intel/intel-junction-2026/logs/detlive"
TOP, CLOSE = "realsense 243622061421", "realsense 243722064316"
SAMP = json.load(open(os.path.join(E2, "samples", "samples.json")))

LABS = []
for line in open(os.path.join(E2, "labels", "compare_opus_vs_E2.tsv")).read().splitlines()[1:]:
    p = line.split("\t")
    if len(p) >= 13 and p[1].startswith("6d6bdce3_"):
        LABS.append({"key": p[0], "sample": p[1].rsplit("_", 2)[0], "label": p[2], "colour": p[7],
                     "aspect": float(p[8]), "tape_e2": float(p[11])})


def jpg80(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def brief(b):
    return {k: b.get(k) for k in ("id", "view", "colour", "shape", "shape_conf", "aspect", "fill", "area",
                                  "centre_px", "tape_score", "verdict", "reasons", "merged", "split", "partial")}


def scan(top, close):
    try:
        res = I.scan_full(top, close)
    except I.HandsPresent as exc:
        return {"refused": True, "reason": str(exc)[:80], "hands": 0, "bricks": []}
    return {"refused": False, "reason": "", "hands": len(res["hands"]), "bricks": [brief(b) for b in res["bricks"]]}


def top_truth(sample, tops):
    """V1 rule: white top bricks, sorted by area, are the labelled not_2x2 count first (count match only)."""
    rows = [r for r in LABS if r["sample"] == sample]
    n24 = sum(1 for r in rows if r["label"].endswith("NOT2X2"))
    n2 = sum(1 for r in rows if r["colour"] == "white" and r["label"] == "GOOD")
    truth, white = {}, []
    for b in tops:
        if b["merged"]:
            truth[b["id"]] = "merged_blob"
        elif b["colour"] == "white":
            white.append(b)
        else:
            truth[b["id"]] = "2x2"
    white.sort(key=lambda b: -b["area"])
    for i, b in enumerate(white):
        truth[b["id"]] = ("not_2x2" if i < n24 else "2x2") if len(white) == n24 + n2 else None
    return truth


def nearest(bricks, colour, centre, tol=12.0):
    best, bd = None, tol
    for b in bricks:
        if b["view"] != "close" or b["centre_px"] is None:
            continue
        d = float(np.hypot(b["centre_px"][0] - centre[0], b["centre_px"][1] - centre[1]))
        if d <= bd:
            best, bd = b, d
    return best, (round(bd, 1) if best is not None else None)


KEYS = sorted(k for k in SAMP if k.startswith("6d6bdce3_"))
RES = {}
for key in KEYS:
    top = cv2.imread(SAMP[key]["views"][TOP]["png"])
    close = cv2.imread(SAMP[key]["views"][CLOSE]["png"])
    if top is None or close is None:
        RES[key] = {"error": "cannot read PNG pair"}
        continue
    raw = scan(top, close)
    jp = scan(jpg80(top), jpg80(close))
    for side in (raw, jp):
        if not side["refused"]:
            tops = [b for b in side["bricks"] if b["view"] == "top"]
            truth = top_truth(key, tops)
            for b in tops:
                b["truth"] = truth[b["id"]]
    RES[key] = {"raw": raw, "jpg": jp}

# brick map: Opus key -> raw close brick (built once from the baseline run; features used: colour, aspect, tape_e2)
if os.path.exists(MAPP):
    MAP = json.load(open(MAPP))
else:
    MAP = {}
    for r in LABS:
        cands = [b for b in RES.get(r["sample"], {}).get("raw", {}).get("bricks", [])
                 if b["view"] == "close" and b["centre_px"] is not None
                 and abs(b["aspect"] - r["aspect"]) <= 0.02 and abs((b["tape_score"] or 0) - r["tape_e2"]) <= 0.0015]
        if cands:  # V1 renamed some hues (white read as cream, cream as yellow): colour only breaks ties
            b = min(cands, key=lambda c: (c["colour"] != r["colour"], abs((c["tape_score"] or 0) - r["tape_e2"])))
            MAP[r["key"]] = {"sample": r["sample"], "centre_px": b["centre_px"], "colour": b["colour"],
                             "area": b["area"]}
    json.dump(MAP, open(MAPP, "w"), indent=1)

ROWS = []
for r in LABS:
    rec = {"key": r["key"], "label": r["label"], "colour": r["colour"], "mapped": r["key"] in MAP}
    if r["key"] in MAP:
        m = MAP[r["key"]]
        for side in ("raw", "jpg"):
            bricks = RES.get(m["sample"], {}).get(side, {}).get("bricks", [])
            b, dist = nearest(bricks, m["colour"], m["centre_px"])
            rec[side + "_verdict"] = b["verdict"] if b else None
            rec[side + "_shape"] = b["shape"] if b else None
            rec[side + "_tape"] = b["tape_score"] if b else None
            rec[side + "_dist"] = dist
    ROWS.append(rec)


def lab_metrics(side):
    rows = [r for r in ROWS if r["mapped"]]
    goods = [r for r in rows if r["label"] == "GOOD"]
    tapes = [r for r in rows if r["label"] == "DEFECT_TAPE"]
    not2 = [r for r in rows if r["label"] in ("DEFECT_TAPE_NOT2X2", "DEFECT_NOT2X2")]
    mer = [r for r in rows if r["label"] == "MERGED"]
    cd = lambda rs: sum(r[side + "_verdict"] == "CONFIDENT_DEFECT" for r in rs)  # noqa: E731
    tv = lambda rs: [r[side + "_tape"] for r in rs if r[side + "_tape"] is not None]  # noqa: E731
    return {"labelled": len(ROWS), "unmapped": len(ROWS) - len(rows),
            "good_n": len(goods), "good_cd": cd(goods),
            "good_cg": sum(r[side + "_verdict"] == "CONFIDENT_GOOD" for r in goods),
            "good_max_tape": max(tv(goods) or [None]), "good_cd_keys": [r["key"] for r in goods if r[side + "_verdict"] == "CONFIDENT_DEFECT"],
            "tape_n": len(tapes), "tape_cd": cd(tapes), "tape_min_tape": min(tv(tapes) or [None]),
            "tape_missed_keys": [r["key"] for r in tapes if r[side + "_verdict"] != "CONFIDENT_DEFECT"],
            "not2x2_n": len(not2), "not2x2_cd": cd(not2), "merged_n": len(mer), "merged_cd": cd(mer)}


def count(side, view):
    c = {}
    for key in KEYS:
        res = RES.get(key, {}).get(side, {})
        if not res or res.get("refused"):
            continue
        for b in res["bricks"]:
            if b["view"] == view:
                c[b["verdict"]] = c.get(b["verdict"], 0) + 1
    return c


def top_agg(side):
    tot = {"n": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "n_2x2": 0, "cd_on_2x2": 0}
    for key in KEYS:
        res = RES.get(key, {}).get(side, {})
        if not res or res.get("refused"):
            continue
        for b in res["bricks"]:
            t = b.get("truth")
            if b["view"] != "top" or t is None or t == "merged_blob":
                continue
            pred = b["verdict"] == "CONFIDENT_DEFECT"
            tot["n"] += 1
            if t == "not_2x2":
                tot["tp" if pred else "fn"] += 1
            else:
                tot["n_2x2"] += 1
                tot["fp" if pred else "tn"] += 1
                tot["cd_on_2x2"] += int(pred)
    return tot


def flips():
    """Label-free check: raw and JPEG verdicts of the same brick (centre within 12 px, same view)."""
    out = {"matched": 0, "verdict_changes": 0, "raw_not_cd_jpg_cd": [], "raw_cd_jpg_not_cd": []}
    for key in KEYS:
        R, J = RES.get(key, {}).get("raw", {}), RES.get(key, {}).get("jpg", {})
        if not R or not J or R.get("refused") or J.get("refused"):
            continue
        for view in ("top", "close"):
            A = [b for b in R["bricks"] if b["view"] == view]
            B = [b for b in J["bricks"] if b["view"] == view]
            used = set()
            for a in A:
                best, bd, bj = None, 1e9, -1
                for j, c in enumerate(B):
                    if j in used or a["centre_px"] is None or c["centre_px"] is None:
                        continue
                    d = float(np.hypot(a["centre_px"][0] - c["centre_px"][0], a["centre_px"][1] - c["centre_px"][1]))
                    if d < bd:
                        best, bd, bj = c, d, j
                if best is None or bd > 12.0:
                    if a["verdict"] == "CONFIDENT_DEFECT":
                        out.setdefault("raw_cd_unmatched_jpg", []).append([key, view, a["id"], a["colour"], a["area"]])
                    continue
                used.add(bj)
                out["matched"] += 1
                out["verdict_changes"] += int(a["verdict"] != best["verdict"])
                row = [key, view, a["id"], a["colour"], a["area"], a["tape_score"], best["tape_score"]]
                if a["verdict"] != "CONFIDENT_DEFECT" and best["verdict"] == "CONFIDENT_DEFECT":
                    out["raw_not_cd_jpg_cd"].append(row)
                if a["verdict"] == "CONFIDENT_DEFECT" and best["verdict"] != "CONFIDENT_DEFECT":
                    out["raw_cd_jpg_not_cd"].append(row)
            for j, c in enumerate(B):
                if j not in used and c["verdict"] == "CONFIDENT_DEFECT":
                    out.setdefault("jpg_cd_unmatched_raw", []).append([key, view, c["id"], c["colour"], c["area"]])
    return out


LIVE = {}
for tag in ("153426", "155631"):
    top = cv2.imread(os.path.join(DET, "top_%s.jpg" % tag))
    close = cv2.imread(os.path.join(DET, "close_%s.jpg" % tag))
    LIVE[tag] = scan(top, close) if top is not None and close is not None else {"error": "unreadable"}

summary = {
    "variant": OVR,
    "refused": {s: [k for k in KEYS if RES[k].get(s, {}).get("refused")] for s in ("raw", "jpg")},
    "close_counts": {s: count(s, "close") for s in ("raw", "jpg")},
    "top_counts": {s: count(s, "top") for s in ("raw", "jpg")},
    "top_metrics": {s: top_agg(s) for s in ("raw", "jpg")},
    "labels": {s: lab_metrics(s) for s in ("raw", "jpg")},
    "flips": flips(),
    "live": {t: {"refused": v.get("refused"), "bricks": [
        [b["id"], b["view"], b["colour"], b["shape"], b["shape_conf"], b["tape_score"], b["verdict"], b["reasons"]]
        for b in v.get("bricks", [])]} for t, v in LIVE.items()},
}
with open(OUT, "w") as fh:
    json.dump({"summary": summary, "label_rows": ROWS, "frames": RES, "live_full": LIVE}, fh, indent=1, default=str)
print(json.dumps(summary, default=str))
