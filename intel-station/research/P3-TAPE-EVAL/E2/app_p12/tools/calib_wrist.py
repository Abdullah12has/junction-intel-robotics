#!/usr/bin/env python3
"""Opus I-60: wrist-camera Cartesian calibration with a brick under the hover.
IK moves of +-D mm along base x and y at constant height and yaw; brick pixel shift -> M_base (px/mm),
M_tool = M_base @ R2(psi). Saves app/tools/wrist_calib.json. usage: calib_wrist.py [--d 10]"""
import argparse
import json
import math
import sys
import time

import numpy as np

sys.path.insert(0, "/home/intel/intel-junction-2026/app/tools")
import grasp_auto as ga  # noqa: E402
import kin  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--d", type=float, default=10.0)
a = ap.parse_args()
ga.OUT = f"{ga.R}/logs/grasp_auto/calib_{time.strftime('%H%M%S', time.gmtime())}"
import os  # noqa: E402
os.makedirs(ga.OUT, exist_ok=True)
st = ga.move(ga.HOVER, speed=8, comp=True)
q0 = st["present"][:5]
p0, Rm = kin.fk(q0)
psi = math.atan2(Rm[1, 0], Rm[0, 0])
z = kin.MAT_Z + kin.HOVER_DZ
base_ik = kin.ik(p0[0], p0[1], z, psi, q0)
print("hover fk", p0.round(1), "psi deg", round(math.degrees(psi), 1), "ik at hover height", base_ik)
ga.move(base_ik + [25.0], speed=6, comp=True)
w0 = ga.wrist_brick(ga.snap("c0")["wrist"])
cols = []
for k, (dx, dy) in enumerate(((a.d, 0.0), (0.0, a.d), (-a.d, 0.0), (0.0, -a.d))):
    q = kin.ik(p0[0] + dx, p0[1] + dy, z, psi, base_ik)
    ga.move(q + [25.0], speed=6, comp=True)
    w = ga.wrist_brick(ga.snap(f"c{k + 1}")["wrist"])
    ga.move(base_ik + [25.0], speed=6, comp=True)
    cols.append(((dx, dy), [w["centre"][0] - w0["centre"][0], w["centre"][1] - w0["centre"][1]]))
    print("probe", dx, dy, "px", cols[-1][1], "brick", w)
# least squares: px = M_base @ [dx, dy]
A = np.array([c[0] for c in cols])
B = np.array([c[1] for c in cols])
Mb = np.linalg.lstsq(A, B, rcond=None)[0].T
c, s = math.cos(psi), math.sin(psi)
R2 = np.array([[c, -s], [s, c]])
Mt = Mb @ R2
res = (A @ Mb.T - B)
out = {"t": time.strftime("%H:%M:%SZ", time.gmtime()), "psi_deg": round(math.degrees(psi), 2), "M_base": Mb.round(4).tolist(),
       "M_tool": Mt.round(4).tolist(), "fit_rms_px": round(float(np.sqrt((res ** 2).mean())), 2), "d_mm": a.d,
       "hover_ik": base_ik, "brick_px_at_hover": w0["centre"], "brick_size_px": w0["size"], "target_px": list(ga.TARGET_PX)}
json.dump(out, open(kin.CALF, "w"), indent=1)
print(json.dumps(out))
