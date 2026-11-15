"""Scan one saved frame pair and write verdict.json plus an annotated image (CLI for the E2 detector).

usage: python3 -I scan_once.py --top PATH [--close PATH] --out DIR
Exit code 3 when hands are present (frame refused; route nothing).
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inspector as I  # noqa: E402

COL = {"CONFIDENT_DEFECT": (0, 0, 255), "CONFIDENT_GOOD": (0, 255, 0), "UNCERTAIN": (0, 255, 255)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", required=True)
    ap.add_argument("--close")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    top = cv2.imread(a.top)
    close = cv2.imread(a.close) if a.close else None
    if top is None or (a.close and close is None):
        raise SystemExit("cannot read input image")
    os.makedirs(a.out, exist_ok=True)
    t0 = time.perf_counter()
    try:
        res = I.scan_full(top, close)
    except I.HandsPresent as e:
        json.dump({"refused": True, "reason": str(e), "bricks": []}, open(os.path.join(a.out, "verdict.json"), "w"),
                  indent=1)
        print("REFUSED", e)
        sys.exit(3)
    ms = (time.perf_counter() - t0) * 1000.0
    bricks = [{k: v for k, v in b.items() if k != "_mask"} for b in res["bricks"]]
    out = {"refused": False, "latency_ms_scan_cpu": round(ms, 1), "threshold_loaded": res["threshold_loaded"],
           "hands": res["hands"], "plates_rejected_top": res["plates_rejected"],
           "partial_at_border_close": res["partial_at_border"], "rejected_close": res["rejected"], "bricks": bricks}
    json.dump(out, open(os.path.join(a.out, "verdict.json"), "w"), indent=1, default=str)
    vis = top.copy()
    for b in res["bricks"]:
        if b["view"] != "top":
            continue
        box = cv2.boxPoints(((b["centre_px"][0], b["centre_px"][1]), tuple(b["footprint_px"]), b["angle_deg"]))
        cv2.polylines(vis, [box.astype(np.int32)], True, COL[b["verdict"]], 1)
        cv2.putText(vis, b["id"], (int(b["centre_px"][0]) + 4, int(b["centre_px"][1]) - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(a.out, "annotated_top.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"bricks={len(bricks)} scan_ms={ms:.1f} out={a.out}")


if __name__ == "__main__":
    main()
