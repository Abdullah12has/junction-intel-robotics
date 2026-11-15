"""Calibrate det2 from the 6d6bdce3 reference frames (home-and-still windows): static masks, mat ROI,
2x2 reference area (top view, squarish mid-size blobs) and tape percentiles (non-yellow bricks, assumed good).
Run: envs/data-env/bin/python -I -B app/anomaly/det2_e3/build_model.py  (writes models/ and the work folder)."""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import det2  # noqa: E402

WORK = "/home/intel/intel-junction-2026/work/P2-B-DET2/E3"
DS = "6d6bdce3"


def main():
    idx = json.load(open(f"{WORK}/out/samples_index.json"))
    os.makedirs(det2.MODEL_DIR, exist_ok=True)
    models, frames = {}, {}
    for view in ("top", "close"):
        recs = [r for r in idx if r["ds"] == DS and r["cam"] == view]
        bgrs = [cv2.imread(r["png"]) for r in recs]
        static, roi = det2.build_static(bgrs, view)
        np.savez(os.path.join(det2.MODEL_DIR, f"{view}.npz"), static=static, roi=roi)
        models[view] = {"static": static, "roi": roi}
        frames[view] = (recs, bgrs)
        print(view, "frames", len(recs), "static px", int(static.sum()), "roi px", int(roi.sum()))
    areas = []
    recs, bgrs = frames["top"]
    for b in bgrs:
        for bl in det2.find_bricks(b, "top", models["top"]["static"], models["top"]["roi"]):
            (_, _), (rw, rh), _ = bl["rect"]
            asp = max(rw, rh) / max(min(rw, rh), 1e-6)
            if not bl["touch"] and 0.85 <= asp <= 1.18 and 900 <= bl["area"] <= 2200:
                areas.append(bl["area"])
    a_ref = float(np.median(areas))
    print("a_ref (2x2 px area, top)", round(a_ref, 1), "n", len(areas), "p10/p90", np.percentile(areas, [10, 90]).round(0).tolist())
    cal = {"ds": DS, "a_ref_top_px": round(a_ref, 1), "a_ref_samples": len(areas), "views": {}}
    allrecs = []
    for view in ("top", "close"):
        recs, bgrs = frames[view]
        for r, b in zip(recs, bgrs):
            for rec in det2.analyse(b, view, models[view], a_ref=a_ref):
                rec.update({"id": f"{DS}_e{r['ep']:02d}_{r['which']}_{r['serial']}_{rec['k']}", "ep": r["ep"],
                            "which": r["which"], "serial": r["serial"], "png": r["png"]})
                allrecs.append(rec)
    for view in ("top", "close"):
        rv = [x for x in allrecs if x["view"] == view]
        nonyellow = [x for x in rv if x["colour"] not in ("yellow", "cream") and (view == "close" or x.get("shape") == "2x2")]
        p = {}
        for f in det2.TAPE_FEATS:
            vals = [x["tape_feats"][f] for x in nonyellow]
            p[f] = {"p95": round(float(np.percentile(vals, 95)), 4), "p99": round(float(np.percentile(vals, 99)), 4),
                    "max": round(float(np.max(vals)), 4), "median": round(float(np.median(vals)), 4)}
        shape_counts = {}
        for x in rv:
            shape_counts[x.get("shape", "-")] = shape_counts.get(x.get("shape", "-"), 0) + 1
        colours = {}
        for x in rv:
            colours[x["colour"]] = colours.get(x["colour"], 0) + 1
        cal["views"][view] = {"bricks": len(rv), "colours": colours, "shapes": shape_counts,
                              "tape_negatives": len(nonyellow), "tape_stats": p}
        print(view, "bricks", len(rv), "colours", colours, "shapes", shape_counts, "negatives", len(nonyellow))
        for f in det2.TAPE_FEATS:
            print("   tape", f, p[f])
    with open(os.path.join(det2.MODEL_DIR, "cal.json"), "w") as fh:
        json.dump(cal, fh, indent=1)
    with open(f"{WORK}/out/bricks_build.json", "w") as fh:
        json.dump(allrecs, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("WROTE models and bricks_build.json, records", len(allrecs))


if __name__ == "__main__":
    main()
