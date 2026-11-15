#!/usr/bin/env python3
"""P3-TWIN-CAMS E1 step 11a: inventory of picked-brick correspondences (read-only).

For every logs/sort/any_*/ep_*/episode.json: the brick's base xy from the kin FK at the lift configuration (kin.fk, the
validated convention; the TCP stays at the brick while it is lifted), compared with the scan estimate brick_xy_est, plus
the pre/released top frames that exist. Writes out/s11_inventory.json. Run with envs/anomalib-env (kin imports scipy).
Run: envs/anomalib-env/bin/python -I tools/s11_inventory.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/intel/intel-junction-2026/app/tools")
import kin  # noqa: E402

R = "/home/intel/intel-junction-2026"
E1 = f"{R}/work/P3-TWIN-CAMS/E1"


def main():
    rows = []
    for ej in sorted(glob.glob(f"{R}/logs/sort/any_*/ep_*/episode.json")):
        d = json.load(open(ej))
        folder = os.path.dirname(ej)
        ep = os.path.relpath(folder, f"{R}/logs/sort")
        lift = d.get("lift_q")
        p = np.asarray(kin.fk(lift)[0], float) if lift else None
        scan = d.get("brick_xy_est")
        diff = None
        if p is not None and scan:
            diff = round(float(np.hypot(p[0] - scan[0], p[1] - scan[1])), 1)
        rows.append({
            "ep": ep, "success": d.get("success"), "label": d.get("label"), "colour": d.get("colour"),
            "plate": d.get("plate"), "shape": d.get("shape"), "t_start": d.get("t_start"), "t_end": d.get("t_end"),
            "lift_tcp_mm": None if p is None else [round(float(v), 1) for v in p],
            "brick_xy_est_mm": scan, "kin_vs_scan_mm": diff,
            "pre_top": os.path.exists(os.path.join(folder, "pre_top.jpg")),
            "released_top": os.path.exists(os.path.join(folder, "released_top.jpg")),
        })
    with open(f"{E1}/out/s11_inventory.json", "w") as fh:
        json.dump(rows, fh, indent=1)
    n_ok = sum(1 for r in rows if r["success"] and r["pre_top"] and r["released_top"])
    print("episodes", len(rows), "with success and both top frames", n_ok, flush=True)
    for r in rows:
        print(r["ep"], "succ", r["success"], r["colour"], r["plate"], r["shape"], "lift TCP mm", r["lift_tcp_mm"],
              "scan xy", r["brick_xy_est_mm"], "diff", r["kin_vs_scan_mm"], "pre/rel", r["pre_top"],
              r["released_top"], flush=True)


if __name__ == "__main__":
    main()
