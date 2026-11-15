"""Map sort_any.grasp_reachable over the mat (base frame), print the reach boundary and the time per call."""
import json
import math
import sys
import time

import numpy as np

sys.path.insert(0, "/home/intel/intel-junction-2026/app/tools")
import kin  # noqa: E402
import sort_any as sa  # noqa: E402

c = json.load(open("/home/intel/intel-junction-2026/app/tools/scan_calib.json"))
p, _ = kin.fk(c["q_scan"])
print("scan TCP", [round(float(v), 1) for v in p])
t0 = time.time()
n = 0
rows = []
for x in range(60, 400, 20):
    line = ""
    for y in range(-200, 300, 20):
        d = (x - p[0], y - p[1])
        ok = sa.grasp_reachable(c["q_scan"], d)
        n += 1
        r = math.hypot(x - kin.PAN_XY_MM[0], y - kin.PAN_XY_MM[1])
        rows.append((r, ok, x, y))
        line += "#" if ok else "."
    print(f"x={x:4d} {line}")
dt = (time.time() - t0) / n
print(f"{n} cells, {dt * 1000:.1f} ms per uncached call")
ok_r = [r for r, ok, *_ in rows if ok]
no_r = [r for r, ok, *_ in rows if not ok and r >= 115]
print("max reachable r", round(max(ok_r), 1), "min unreachable r (>=115)", round(min(no_r), 1))
for lo in range(240, 360, 10):
    sel = [ok for r, ok, *_ in rows if lo <= r < lo + 10]
    if sel:
        print(f"r {lo}-{lo + 10}: {sum(sel)}/{len(sel)} reachable")
for pl, xy in sa.PLATES_XY.items():
    print(pl, "plate centre reachable:", sa.grasp_reachable(c["q_scan"], (xy[0] - p[0], xy[1] - p[1])))
