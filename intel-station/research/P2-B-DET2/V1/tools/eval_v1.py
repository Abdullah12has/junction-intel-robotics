"""P2-B-DET2 V1 evaluation harness (analysis only; not part of the app).
usage: python3 -I -B eval_v1.py APP_DIR OUT_JSON
Scans every 6d6bdce3 home-still frame pair as PNG and as JPEG q80, compares the two, checks top-view shape
against the close-view labels (labels/compare_opus_vs_E2.tsv) and dumps skin and hue diagnostics."""
import json
import os
import sys
import time

import cv2
import numpy as np

APP, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, APP)
import inspector as I  # noqa: E402

E2 = "/home/intel/intel-junction-2026/work/P2-B-DET2/E2"
TOP, CLOSE = "realsense 243622061421", "realsense 243722064316"
KEEP = ("id", "view", "colour", "shape", "shape_conf", "aspect", "area", "merged", "merged_reason",
        "tape_score", "verdict", "reasons", "centre_px", "bbox", "footprint_px")
SAMP = json.load(open(os.path.join(E2, "samples", "samples.json")))
ROWS = []
for line in open(os.path.join(E2, "labels", "compare_opus_vs_E2.tsv")).read().splitlines()[1:]:
    p = line.split("\t") + [""] * 13
    if p[1].startswith("6d6bdce3_"):
        ROWS.append({"sample": p[1].rsplit("_", 2)[0], "opus": p[2], "colour": p[7]})


def jpg80(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def compact(b):
    return {k: b.get(k) for k in KEEP}


def run(top, close):
    t0 = time.perf_counter()
    try:
        res = I.scan_full(top, close)
    except I.HandsPresent as exc:
        return {"refused": True, "reason": str(exc), "hands": 0, "bricks": [], "ms": None}
    return {"refused": False, "reason": "", "hands": len(res["hands"]),
            "bricks": [compact(b) for b in res["bricks"]],
            "ms": round((time.perf_counter() - t0) * 1000.0, 1)}


def pair_views(a, b, view, tol=12.0):
    A = [x for x in a["bricks"] if x["view"] == view]
    B = [x for x in b["bricks"] if x["view"] == view]
    used, pairs = set(), []
    for x in A:
        best, bd, bj = None, 1e9, -1
        for j, y in enumerate(B):
            if j in used:
                continue
            d = float(np.hypot(x["centre_px"][0] - y["centre_px"][0], x["centre_px"][1] - y["centre_px"][1]))
            if d < bd:
                best, bd, bj = y, d, j
        if best is not None and bd <= tol:
            used.add(bj)
            pairs.append((x, best, round(bd, 1)))
        else:
            pairs.append((x, None, None))
    return pairs, [y for j, y in enumerate(B) if j not in used]


def diffs(raw, jp):
    out = {"changes": [], "max_tape_delta": 0.0, "matched": 0}
    if raw["refused"] or jp["refused"]:
        return out
    for view in ("top", "close"):
        pairs, extra = pair_views(raw, jp, view)
        for x, y, d in pairs:
            if y is None:
                out["changes"].append({"view": view, "raw_id": x["id"], "change": "missing_in_jpeg",
                                       "raw": [x["verdict"], x["shape"], x["colour"]]})
                continue
            out["matched"] += 1
            if x["tape_score"] is not None and y["tape_score"] is not None:
                out["max_tape_delta"] = max(out["max_tape_delta"], abs(x["tape_score"] - y["tape_score"]))
            if (x["verdict"], x["shape"], x["colour"]) != (y["verdict"], y["shape"], y["colour"]):
                out["changes"].append({"view": view, "raw_id": x["id"], "jpeg_id": y["id"], "shift_px": d,
                                       "change": "verdict_shape_or_colour",
                                       "raw": [x["verdict"], x["shape"], x["colour"]],
                                       "jpeg": [y["verdict"], y["shape"], y["colour"]]})
        for y in extra:
            out["changes"].append({"view": view, "jpeg_id": y["id"], "change": "extra_in_jpeg",
                                   "jpeg": [y["verdict"], y["shape"], y["colour"]]})
    return out


def truth_for(sample, tops):
    rows = [r for r in ROWS if r["sample"] == sample]
    n24 = sum(1 for r in rows if r["opus"].endswith("NOT2X2"))
    n2 = sum(1 for r in rows if r["colour"] == "white" and r["opus"] == "GOOD")
    truth, white = {}, []
    for b in tops:
        if b["merged"]:
            truth[b["id"]] = "merged_blob"
        elif b["colour"] == "white":
            white.append(b)
        else:
            truth[b["id"]] = "2x2"  # prior: only the white 2x4s are not 2x2 in this brick set
    white.sort(key=lambda b: -b["area"])
    if len(white) == n24 + n2:
        for i, b in enumerate(white):
            truth[b["id"]] = "not_2x2" if i < n24 else "2x2"
        method = "count_match" if white else "no_white"
    else:
        for b in white:
            truth[b["id"]] = None
        method = "count_mismatch_skipped"
    return truth, method


def top_metrics(tops):
    m = {"n": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "n_2x2": 0, "conf_defect_on_2x2": 0, "unknown_on_2x2": 0}
    for b in tops:
        t = b.get("truth")
        if t is None or t == "merged_blob":
            continue
        pred = b["verdict"] == "CONFIDENT_DEFECT"
        m["n"] += 1
        if t == "not_2x2":
            m["tp" if pred else "fn"] += 1
        else:
            m["n_2x2"] += 1
            m["fp" if pred else "tn"] += 1
            m["conf_defect_on_2x2"] += int(pred)
            m["unknown_on_2x2"] += int(b["shape"] == "unknown")
    return m


def agg(ms):
    tot = {}
    for m in ms:
        for k, v in m.items():
            tot[k] = tot.get(k, 0) + v
    if tot:
        tot["precision_not_2x2"] = round(tot["tp"] / (tot["tp"] + tot["fp"]), 3) if tot["tp"] + tot["fp"] else None
        tot["recall_not_2x2"] = round(tot["tp"] / (tot["tp"] + tot["fn"]), 3) if tot["tp"] + tot["fn"] else None
        tot["conf_defect_rate_on_2x2"] = round(tot["conf_defect_on_2x2"] / tot["n_2x2"], 3) if tot["n_2x2"] else None
    return tot


def skin_diag(top):
    h, s, v = I._split(top)
    skin = ((h <= 18) | (h >= 172)) & (s >= 45) & (s <= I.HANDS_MAX_S) & (v >= 90)
    roi = I._roi(h.shape, I.TOP_ROI)
    H, W = h.shape
    out = []
    for m, a, st in I._components(skin, 300):
        x, y, w, hh = (int(t) for t in st[:4])
        pts = np.column_stack(np.nonzero(m)[::-1]).astype(np.float32)
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
        out.append({"area": int(a), "in_mat": int((m & roi).sum()), "bbox": [x, y, w, hh],
                    "touches_border": bool(x <= 0 or y <= 0 or x + w >= W or y + hh >= H),
                    "rel_2x2": round(float(a) / I.A2X2_TOP, 2),
                    "fill": round(float(a) / max(float(rw) * float(rh), 1.0), 2),
                    "aspect": round(max(float(rw), float(rh)) / max(min(float(rw), float(rh)), 1.0), 2),
                    "med_hsv": [float(np.median(h[m])), float(np.median(s[m])), float(np.median(v[m]))]})
    return out


def hue_diag(close):
    h, s, v = I._split(close)
    out = []
    for i, b in enumerate(I.segment_close(close)["bricks"]):
        m = b["_mask"]
        ch = m & (s >= 55) & (v >= 80)
        n = int(ch.sum())
        rec = {"k": i, "area": b["area"], "centre_px": b["centre_px"], "colour": b["colour"],
               "merged": b["merged"], "merged_reason": b["merged_reason"], "shape": b["shape"], "chroma_px": n}
        if n >= 30:
            hh = h[ch].astype(np.int64)
            hist = np.bincount(np.minimum(hh // 6, 29), minlength=30)
            p1 = int(np.argmax(hist))
            dist = np.minimum(np.abs(np.arange(30) - p1), 30 - np.abs(np.arange(30) - p1))
            cand = np.where(dist >= 3)[0]
            p2 = int(cand[np.argmax(hist[cand])]) if len(cand) else -1
            rec.update(peak1_hue=p1 * 6 + 3, share1=round(float(hist[p1]) / n, 3),
                       peak2_hue=(p2 * 6 + 3) if p2 >= 0 else None,
                       share2=round(float(hist[p2]) / n, 3) if p2 >= 0 else 0.0)
        out.append(rec)
    return out


def count(res_list, view):
    c = {}
    for res in res_list:
        if res["refused"]:
            continue
        for b in res["bricks"]:
            if b["view"] == view:
                c[b["verdict"]] = c.get(b["verdict"], 0) + 1
    return c


results = []
for key in sorted(k for k in SAMP if k.startswith("6d6bdce3_")):
    rec = {"sample": key}
    try:
        top = cv2.imread(SAMP[key]["views"][TOP]["png"])
        close = cv2.imread(SAMP[key]["views"][CLOSE]["png"])
        if top is None or close is None:
            raise RuntimeError("cannot read PNG pair")
        raw = run(top, close)
        jp = run(jpg80(top), jpg80(close))
        for res in (raw, jp):
            if not res["refused"]:
                tops = [b for b in res["bricks"] if b["view"] == "top"]
                truth, method = truth_for(key, tops)
                for b in tops:
                    b["truth"] = truth[b["id"]]
                res["top_method"] = method
                res["top_metrics"] = top_metrics(tops)
        rec.update(raw=raw, jpg=jp, diffs=diffs(raw, jp), skin_top=skin_diag(top))
        if not raw["refused"]:
            rec["close_hue"] = hue_diag(close)
    except Exception as exc:  # noqa: BLE001
        rec["error"] = repr(exc)[:300]
    results.append(rec)

ok = [r for r in results if "raw" in r]
ms = [r["raw"]["ms"] for r in ok if r["raw"]["ms"] is not None]
summary = {
    "app": APP, "frames": len(results), "errors": [r["sample"] for r in results if "error" in r],
    "raw_refused": [r["sample"] for r in ok if r["raw"]["refused"]],
    "jpg_refused": [r["sample"] for r in ok if r["jpg"]["refused"]],
    "raw_vs_jpg_changes": sum(len(r["diffs"]["changes"]) for r in ok),
    "max_tape_delta_raw_vs_jpg": max([r["diffs"]["max_tape_delta"] for r in ok] or [0.0]),
    "raw_top_metrics": agg([r["raw"]["top_metrics"] for r in ok if "top_metrics" in r["raw"]]),
    "jpg_top_metrics": agg([r["jpg"]["top_metrics"] for r in ok if "top_metrics" in r["jpg"]]),
    "raw_top_method": {m: sum(1 for r in ok if r["raw"].get("top_method") == m)
                       for m in ("count_match", "count_mismatch_skipped", "no_white")},
    "raw_verdicts_top": count([r["raw"] for r in ok], "top"),
    "raw_verdicts_close": count([r["raw"] for r in ok], "close"),
    "jpg_verdicts_top": count([r["jpg"] for r in ok], "top"),
    "jpg_verdicts_close": count([r["jpg"] for r in ok], "close"),
    "raw_scan_ms_mean": round(sum(ms) / len(ms), 1) if ms else None,
}
with open(OUT, "w") as fh:
    json.dump({"summary": summary, "samples": results}, fh, indent=1, default=str)
print(json.dumps(summary, indent=1, default=str))
