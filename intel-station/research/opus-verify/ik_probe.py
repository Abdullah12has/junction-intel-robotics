"""Probe: does ik_rel accept wrist-yaw changes at the station hover, what are the roll bounds, how fast is one call,
and how far from the pan axis are the station and the plates."""
import math
import sys
import time

import numpy as np

APP = "/home/intel/intel-junction-2026/app"
sys.path[:0] = [f"{APP}/tools"]
import kin  # noqa: E402
import sort_any as sa  # noqa: E402

q0 = list(sa.SPOT_HOVER)
print("norm bounds", [tuple(np.round(b, 1)) for b in zip(*kin.CAL.norm_bounds(3))])
print("pan axis", kin.PAN_XY_MM, "station r", round(math.hypot(291.2 - kin.PAN_XY_MM[0], 65.4 - kin.PAN_XY_MM[1]), 1))
print("plates", getattr(sa, "PLATES_XY", None))
for dyaw in range(-120, 121, 20):
    t = time.time()
    q = kin.ik_rel(q0, [0, 0, 0], math.radians(dyaw))
    dt = (time.time() - t) * 1000
    print(f"dyaw {dyaw:5d}: {q} ({dt:.0f} ms)", "psi", None if q is None else round(math.degrees(kin.psi_of(q)), 1))
for r in (160, 200, 240, 280):
    for ang in (-60, -30, 0, 30, 60):
        x = kin.PAN_XY_MM[0] + r * math.cos(math.radians(ang))
        y = kin.PAN_XY_MM[1] + r * math.sin(math.radians(ang))
        q = kin.ik(x, y, kin.MAT_Z + 57.7, kin.psi_of(q0), q0)
        print(f"hover reach r {r} ang {ang}: {'ok' if q else 'NO'} {q}")
