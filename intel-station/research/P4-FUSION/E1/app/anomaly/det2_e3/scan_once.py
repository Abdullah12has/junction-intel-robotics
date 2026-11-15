"""scan_once.py --top PATH [--close PATH] --out DIR
Runs inspector.scan on one frame pair on CPU. Writes DIR/verdict.json, DIR/annotated.jpg and DIR/crops/<id>.jpg."""
import argparse
import json
import os
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import inspector  # noqa: E402

COLOURS = {"CONFIDENT_DEFECT": (0, 0, 255), "UNCERTAIN": (0, 215, 255), "CONFIDENT_GOOD": (0, 200, 0)}


def _annotate(img, recs, view):
    out = img.copy()
    for r in recs:
        if r["view"] != view or r.get("bbox") is None:
            continue
        x, y, w, h = r["bbox"]
        c = COLOURS.get(r["verdict"], (255, 255, 255))
        cv2.rectangle(out, (x, y), (x + w, y + h), c, 2)
        cv2.putText(out, f"{r['id'].split('_')[-1]} {r['verdict'][:9]}", (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top")
    ap.add_argument("--close")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not a.top and not a.close:
        ap.error("give --top and/or --close")
    top = cv2.imread(a.top) if a.top else None
    close = cv2.imread(a.close) if a.close else None
    if (a.top and top is None) or (a.close and close is None):
        ap.error("cannot read an image")
    os.makedirs(os.path.join(a.out, "crops"), exist_ok=True)
    t0 = time.perf_counter()
    recs = inspector.scan(top_rgb=top, close_rgb=close)
    scan_ms = (time.perf_counter() - t0) * 1000.0
    for r in recs:
        if r.get("bbox") is None:
            continue
        src = top if r["view"] == "top" else close
        x, y, w, h = r["bbox"]
        p = os.path.join(a.out, "crops", f"{r['id']}.jpg")
        cv2.imwrite(p, src[y:y + h, x:x + w])
        r["crop_paths"] = [p]
    route = inspector.route(recs)
    with open(os.path.join(a.out, "verdict.json"), "w") as fh:
        json.dump({"route": route, "scan_ms": round(scan_ms, 1), "bricks": recs}, fh, indent=1, default=float)
    panels = []
    if top is not None:
        panels.append(_annotate(top, recs, "top"))
    if close is not None:
        panels.append(_annotate(close, recs, "close"))
    cv2.imwrite(os.path.join(a.out, "annotated.jpg"), cv2.hconcat(panels) if len(panels) > 1 else panels[0])
    print(json.dumps({"route": route, "scan_ms": round(scan_ms, 1), "records": len(recs)}))


if __name__ == "__main__":
    main()
