"""Leave-one-out check of the learned scan map: for every recorded scan pair, fit the map (same code as the loop:
homography with >= 15 pairs, else the similarity correction) on the other pairs and predict this one. Large errors
mark pairs from another camera pose or a biased servo target. Prints one line per pair and the medians by period."""
import json
import os
import shutil
import statistics
import sys
import tempfile

import numpy as np

R = "/home/intel/intel-junction-2026"
sys.path.insert(0, f"{R}/app/tools")
import sort_any as sa  # noqa: E402

pairs = [json.loads(x) for x in open(f"{R}/app/tools/scan_pairs.jsonl") if x.strip()]
tmpd = tempfile.mkdtemp()
sa.RUN = tmpd
c = sa.scan_cal()
p0, _ = sa.kin.fk(c["q_scan"])
rows = []
for i, pr in enumerate(pairs):
    others = pairs[:i] + pairs[i + 1:]
    f = os.path.join(tmpd, "pairs.jsonl")
    with open(f, "w") as fh:
        for o in others:
            fh.write(json.dumps(o) + "\n")
    sa.SCAN_PAIRS = f
    sa.fit_scan_corr()
    d = sa.brick_xy_from_scan(c["q_scan"], {"centre": pr["px"]}, c)
    xy = (p0[0] + d[0], p0[1] + d[1])
    err = float(np.hypot(xy[0] - pr["xy"][0], xy[1] - pr["xy"][1]))
    rows.append((pr.get("t"), pr.get("run"), pr.get("ep"), pr.get("target_px"), round(err, 1)))
    print(f"{pr.get('t')} {pr.get('run')} ep{pr.get('ep')} target {pr.get('target_px')} px {pr['px']} -> LOO error {err:.1f} mm "
          f"({'homography' if sa.SCAN_H is not None else 'similarity'})")
old = [r[4] for r in rows if (r[1] or "") < "any_2055"]
new = [r[4] for r in rows if (r[1] or "") >= "any_2055"]
print("median LOO error: before 20:46Z", statistics.median(old) if old else None, "mm; after", statistics.median(new) if new else None, "mm")
shutil.rmtree(tmpd)
