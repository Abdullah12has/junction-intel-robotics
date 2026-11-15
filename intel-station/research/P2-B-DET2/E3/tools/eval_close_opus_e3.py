"""Close-view tape check against the Opus close labels (shared/inspection-labels.tsv, 35 crops, D-068).
Label keys are mapped to segmented bricks by episode, frame and colour (MAP). Labels are Opus's; they are not
verified here beyond the mapping. Run: envs/data-env/bin/python -I -B tools/eval_close_opus_e3.py"""
import json
import os
import sys

WORK = "/home/intel/intel-junction-2026/work/P2-B-DET2/E3"
APP = "/home/intel/intel-junction-2026/app/anomaly/det2_e3"
sys.path.insert(0, APP)
import cv2  # noqa: E402
import inspector  # noqa: E402

# (label key, Opus label, frame, segmented close brick k or None, note)
MAP = [
    ("close/e03_end/k0", "GOOD", "e03_end", 0, "partial cream at edge; mine orange, touches edge"),
    ("close/e03_end/k1", "GOOD", "e03_end", 2, "red 2x2 with printed 2"),
    ("close/e03_end/k2", "GOOD", "e03_end", 3, "blue 2x2"),
    ("close/e03_start/k0", "MERGED", "e03_start", None, "red 2 + yellow"),
    ("close/e03_start/k1", "GOOD", "e03_start", 2, "blue 2x2"),
    ("close/e04_start/k0", "GOOD", "e04_start", 1, "blue 2x2"),
    ("close/e05_end/k0", "GOOD", "e05_end", 0, "red 2x2; mine orange"),
    ("close/e05_start/k0", "MERGED", "e05_start", None, "red + yellow"),
    ("close/e06_end/k0", "GOOD", "e06_end", 1, "red 2 lying on its side"),
    ("close/e06_start/k1", "GOOD", "e06_start", 0, "red 2x2"),
    ("close/e07_start/k0", "DEFECT_TAPE_NOT2X2", "e07_start", 3, "white 2x4 with tape; mine merged with blue"),
    ("close/e07_start/k1", "GOOD", "e07_start", 1, "orange 2x2 (top left)"),
    ("close/e07_start/k2", "DEFECT_TAPE", "e07_start", None, "blue 2x2 taped: no blue brick segmented"),
    ("close/e07_start/k3", "DEFECT_TAPE", "e07_start", 2, "yellow 2x2 taped; mine cream"),
    ("close/e07_start/k4", "GOOD", "e07_start", 0, "red 2 with print; mine orange"),
    ("close/e09_end/k0", "DEFECT_TAPE", "e09_end", 0, "blue 2x2 taped"),
    ("close/e09_end/k1", "GOOD", "e09_end", 1, "brown 2x2 upside down; mine orange"),
    ("close/e10_end/k0", "DEFECT_NOT2X2", "e10_end", 0, "white 2x4 (shape defect, not tape); mine cream"),
    ("close/e10_end/k1", "GOOD", "e10_end", None, "brown 2x2 upside down: not segmented"),
    ("close/e11_end/k0", "GOOD", "e11_end", 1, "white 2x2"),
    ("close/e11_end/k1", "MERGED", "e11_end", None, "white 2x4 taped + orange + brown"),
    ("close/e11_end/k2", "GOOD", "e11_end", None, "orange 2x2: inside merged blob k0"),
    ("close/e11_end/k3", "DEFECT_TAPE", "e11_end", 2, "blue 2x2 taped"),
    ("close/e11_end/k4", "DEFECT_TAPE", "e11_end", None, "yellow 2x2 taped: not segmented"),
    ("close/e11_start/k0", "GOOD", "e11_start", 2, "white 2x2"),
    ("close/e11_start/k1", "MERGED", "e11_start", None, "red + taped yellow + taped blue"),
    ("close/e12_start/k0", "GOOD", "e12_start", 3, "white 2x2"),
    ("close/e12_start/k1", "MERGED", "e12_start", None, "taped yellow + taped blue"),
    ("close/e12_start/k2", "DEFECT_TAPE", "e12_start", None, "cream/yellow 2x2 taped; in merged blob k1 (old k0 was an orange+white blob)"),
    ("close/e12_start/k3", "GOOD", "e12_start", 2, "red 2 with print"),
    ("close/e13_start/k0", "GOOD", "e13_start", 1, "red 2x2"),
    ("close/e13_start/k1", "GOOD", "e13_start", 0, "orange 2x2"),
    ("close/e13_start/k2", "GOOD", "e13_start", None, "brown 2x2: inside merged blob k2"),
    ("close/e14_start/k0", "GOOD", "e14_start", 1, "red 2x2"),
    ("close/e14_start/k1", "GOOD", "e14_start", 0, "orange 2x2"),
]


def auroc(pos, neg):
    if not pos or not neg:
        return None
    s = 0.0
    for p in pos:
        for n in neg:
            s += 1.0 if p > n else (0.5 if p == n else 0.0)
    return round(s / (len(pos) * len(neg)), 3)


def main():
    frame_recs = {}
    rows = []
    for key, label, frame, k, note in MAP:
        if label in ("MERGED", "DEFECT_NOT2X2"):
            rows.append({"key": key, "label": label, "frame": frame, "k": k, "note": note, "used": False})
            continue
        if frame not in frame_recs:
            img = cv2.imread(f"{WORK}/frames/6d6bdce3_{frame}_243722064316.png")
            frame_recs[frame] = {r["id"]: r for r in inspector.scan(close_rgb=img)}
        rec = frame_recs[frame].get(f"scan_close_{k}") if k is not None else None
        rows.append({"key": key, "label": label, "frame": frame, "k": k, "note": note, "used": True,
                     "segmented": rec is not None,
                     "dark_grey": None if rec is None else rec["tape_score"],
                     "verdict": None if rec is None else rec["verdict"],
                     "route": None if rec is None else rec["route"],
                     "reasons": None if rec is None else rec["reasons"]})
    used = [r for r in rows if r["used"]]
    pos = [r for r in used if r["label"].startswith("DEFECT_TAPE")]
    neg = [r for r in used if r["label"] == "GOOD"]
    pos_s = [r["dark_grey"] for r in pos if r["segmented"]]
    neg_s = [r["dark_grey"] for r in neg if r["segmented"]]
    summary = {
        "labels_mapped": len(used), "labels_excluded": len(rows) - len(used),
        "tape_positive": len(pos), "tape_negative_GOOD": len(neg),
        "positives_not_segmented": sum(1 for r in pos if not r["segmented"]),
        "good_not_segmented": sum(1 for r in neg if not r["segmented"]),
        "auroc_dark_grey": auroc(pos_s, neg_s),
        "flag_TAPE_FLAG": {"tp": sum(1 for r in pos if r["segmented"] and r["dark_grey"] >= inspector.TAPE_FLAG),
                           "fn": sum(1 for r in pos if not r["segmented"] or r["dark_grey"] < inspector.TAPE_FLAG),
                           "fp": sum(1 for r in neg if r["segmented"] and r["dark_grey"] >= inspector.TAPE_FLAG),
                           "tn": sum(1 for r in neg if r["segmented"] and r["dark_grey"] < inspector.TAPE_FLAG)},
        "green_TAPE_CONFIDENT": {"correct_green": sum(1 for r in pos if r["route"] == "GREEN"),
                                 "wrong_green_on_GOOD": sum(1 for r in neg if r["route"] == "GREEN")},
    }
    print(json.dumps(summary, indent=1))
    for r in rows:
        if r["used"]:
            dg = "-" if r["dark_grey"] is None else f"{r['dark_grey']:.4f}"
            print(f"{r['key']:24s} {r['label']:18s} k={r['k']!s:5s} dg={dg:8s} route={r['route']} {r['note']}")
    with open(f"{WORK}/out/eval_close_opus_e3.json", "w") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
