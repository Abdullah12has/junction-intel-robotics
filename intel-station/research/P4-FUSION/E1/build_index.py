"""P4-FUSION E1 step 1: close-view brick index with truth, tape-rule outputs and loop-made crops.

Truth: OURS close frames from R/work/P3-TAPE-EVAL/E1/eval_set.tsv (visual tape truth from P3-LABEL-OURS E1 labels;
V1 status from the sandbox V1 verdicts.tsv); Studio entries from R/work/P3-LABEL-STUDIO/E1/labels_episodes.jsonl
(representative entries, 2x2 target, tape yes/no), frames U<nn>_064316_<phase>_f<frame>.png (close camera).
Brick: DET2.segment_close(sl.mask_jaws(frame)); the largest brick whose colour name matches the labelled colour.
Tape rule: DET2.triage on that brick (the loop's rule; no spot search). Crop: sl.save_crop (the loop's crop code).
Optional review: R/work/P4-FUSION/E1/review_exclude.tsv (rec_id, reason) marks crops removed after the eye check.
"""
import csv
import glob
import json
import os
import sys
from collections import Counter

import cv2
import numpy as np

R = "/home/intel/intel-junction-2026"
E = os.path.join(R, "work", "P4-FUSION", "E1")
sys.path.insert(0, os.path.join(R, "app", "tools"))
sys.path.insert(0, os.path.join(R, "app", "anomaly", "det2_f2"))
import sort_loop as sl  # noqa: E402
import inspector as DET2  # noqa: E402

FAM = {"yellow": ("yellow", "cream"), "cream/yellow": ("yellow", "cream"), "orange": ("orange",),
       "red": ("red",), "white": ("white",), "blue": ("blue",)}
PHASES = ("start", "pregrasp", "grasp", "carry", "release", "end")
OURS_STATUS = {"any_1740/ep_0006": "v1_confirmed", "any_1740/ep_0017": "v1_unverified"}
STUDIO_BASE = os.path.join(R, "work", "P3-LABEL-STUDIO", "E1")
COLS = ["rec_id", "source", "group", "family", "kind", "colour", "truth", "truth_status", "file", "status",
        "n_bricks", "n_match", "brick_colours", "detected_colour", "bbox", "tape_score", "tri_verdict",
        "tri_reasons", "crop"]


def family(colour):
    c = str(colour).lower()
    if "yellow" in c or "cream" in c:
        return "yellow"
    return c


def read_ours():
    out = []
    path = os.path.join(R, "work", "P3-TAPE-EVAL", "E1", "eval_set.tsv")
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            ep = row["episode"]
            stem = os.path.splitext(os.path.basename(row["file"]))[0]
            out.append({"rec_id": "O_" + ep.replace("/", "_") + "_" + row["kind"] + "_" + stem,
                        "source": "ours", "group": ep, "kind": row["kind"], "colour": row["colour"],
                        "truth": "DEFECT" if row["truth_tape"] == "yes" else "GOOD",
                        "truth_status": OURS_STATUS.get(ep, "e1_visual"), "file": row["file"]})
    return out


def read_studio():
    out = []
    frames = os.path.join(STUDIO_BASE, "frames")
    with open(os.path.join(STUDIO_BASE, "labels_episodes.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("dup_of") or e.get("trust") == "nothing":
                continue
            bb = e.get("target_brick") or {}
            tape = str(bb.get("tape", "")).lower()
            colour = str(bb.get("colour", "")).lower()
            if bb.get("shape") != "2x2" or tape not in ("yes", "no") or colour in ("", "none"):
                continue
            unit = str(e.get("unit_group"))
            for f in sorted(glob.glob(os.path.join(frames, unit + "_064316_*.png"))):
                stem = os.path.splitext(os.path.basename(f))[0]
                phase = stem.split("_")[2]
                if phase not in PHASES:
                    continue
                out.append({"rec_id": "S_" + str(e.get("entry")).replace(":", "-") + "_" + stem,
                            "source": "studio", "group": "studio/" + str(e.get("physical_episode")),
                            "kind": phase, "colour": colour,
                            "truth": "DEFECT" if tape == "yes" else "GOOD",
                            "truth_status": "studio_e1_verified", "file": f})
    return out


def pick_brick(img, colour):
    seg = DET2.segment_close(sl.mask_jaws(img))
    bricks = list(seg.get("bricks", []))
    names = sorted({str(b.get("colour")) for b in bricks})
    fam = FAM.get(str(colour).lower(), (str(colour).lower(),))
    match = [b for b in bricks if any(f in str(b.get("colour", "")).lower() for f in fam)]
    if not match:
        return None, len(bricks), 0, names
    best = max(match, key=lambda b: float(b["bbox"][2]) * float(b["bbox"][3]))
    return best, len(bricks), len(match), names


def process(rec, thr, crop_dir):
    row = {c: "" for c in COLS}
    row.update({k: rec.get(k, "") for k in ("rec_id", "source", "group", "kind", "colour", "truth",
                                             "truth_status", "file")})
    row["family"] = family(rec["colour"])
    img = cv2.imread(rec["file"], cv2.IMREAD_COLOR)
    if img is None:
        row["status"] = "unreadable"
        return row
    try:
        b, nb, nm, names = pick_brick(img, rec["colour"])
    except Exception as exc:  # noqa: BLE001
        row["status"] = "seg_error"
        row["tri_reasons"] = repr(exc)[:120]
        return row
    row.update({"n_bricks": nb, "n_match": nm, "brick_colours": ";".join(names)})
    if b is None:
        row["status"] = "no_colour_match"
        return row
    verdict, reasons = DET2.triage(b, thr)
    bbox = [int(v) for v in b["bbox"]]
    crop = os.path.join(crop_dir, rec["rec_id"] + ".png")
    try:
        sl.save_crop(img, bbox, crop)
        row["crop"] = crop
    except Exception as exc:  # noqa: BLE001
        row["tri_reasons"] = "crop_error " + repr(exc)[:100]
    ts = b.get("tape_score")
    row.update({"status": "ok" if row["crop"] else "crop_failed", "detected_colour": str(b.get("colour")),
                "bbox": ",".join(map(str, bbox)), "tape_score": "" if ts is None else "%.5f" % float(ts),
                "tri_verdict": verdict, "tri_reasons": row["tri_reasons"] or ";".join(map(str, reasons))})
    return row


def review_map():
    out = {}
    path = os.path.join(E, "review_exclude.tsv")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                out[r["rec_id"]] = r["reason"]
    return out


def sheet(rows, path):
    tiles = []
    for r in rows:
        im = cv2.imread(r["crop"], cv2.IMREAD_COLOR)
        if im is None:
            continue
        im = cv2.resize(im, (128, 128), interpolation=cv2.INTER_AREA)
        bar = np.zeros((26, 128, 3), np.uint8)
        col = (0, 0, 255) if r["truth"] == "DEFECT" else (0, 200, 0)
        cv2.putText(bar, "%s %s" % (r["truth"][:1], r["family"][:6]), (2, 11), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, col, 1, cv2.LINE_AA)
        cv2.putText(bar, r["group"][-10:], (2, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1,
                    cv2.LINE_AA)
        tiles.append(np.vstack([im, bar]))
    if not tiles:
        return 0
    cols = 10
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(path, np.vstack(grid))
    return len(tiles)


def main():
    cv2.setNumThreads(4)
    thr = DET2.load_thresholds()
    crop_dir = os.path.join(E, "crops")
    os.makedirs(os.path.join(E, "sheets"), exist_ok=True)
    review = review_map()
    recs = read_ours() + read_studio()
    rows = []
    for rec in recs:
        os.makedirs(crop_dir, exist_ok=True)
        row = process(rec, thr, crop_dir)
        if row["rec_id"] in review and row["status"] == "ok":
            row["status"] = "excluded_by_eye"
            row["tri_reasons"] = review[row["rec_id"]]
        rows.append(row)
        print("REC", row["rec_id"], row["truth"], row["family"], row["status"], row["tri_verdict"],
              row["tape_score"], flush=True)
    with open(os.path.join(E, "index.tsv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=COLS, delimiter="\t")
        wr.writeheader()
        wr.writerows(rows)
    print("COUNT_STATUS", dict(Counter(r["status"] for r in rows)), flush=True)
    ok = [r for r in rows if r["status"] == "ok"]
    for key, cnt in sorted(Counter((r["truth"], r["truth_status"], r["family"]) for r in ok).items()):
        print("OK_CROPS", key, cnt, flush=True)
    print("OK_GROUPS", sorted(Counter((r["group"], r["truth"]) for r in ok).items()), flush=True)
    print("TRI_VERDICT_OK", dict(Counter(r["tri_verdict"] for r in ok)), flush=True)
    order = sorted(ok, key=lambda r: (r["truth"], r["family"], r["group"], r["rec_id"]))
    n = sheet(order, os.path.join(E, "sheets", "crops_all_ok.png"))
    print("SHEET_TILES", n, flush=True)


if __name__ == "__main__":
    main()
