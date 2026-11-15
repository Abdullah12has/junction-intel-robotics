#!/usr/bin/env python3
"""Opus I-60: wrist-camera image Jacobian at the hover by differential-IK probes (+-D mm along base x, y) and
whole-image phase correlation (jaws masked), robust to whatever bricks are in view.
Saves app/tools/wrist_calib.json: M_base (px per mm, base x/y), M_tool (= M_base @ R2(psi)), psi, tcp at calib.
usage: calib_M.py [--d 10]"""
import argparse
import json
import math
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grasp_auto as ga  # noqa: E402
import pick_any as pa  # noqa: E402
import kin  # noqa: E402


def gray(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if ga.JAW_MASK is not None:
        g[ga.JAW_MASK] = float(g[~ga.JAW_MASK].mean())
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=float, default=10.0)
    a = ap.parse_args()
    st = ga.state()
    q0 = st["present"][:5]
    g = st["target"][5]
    p0, Rm = kin.fk(q0)
    psi = math.atan2(Rm[1, 0], Rm[0, 0])
    w0 = pa.brick_at_target(ga.fresh("wrist", time.time() + 0.3))
    print("brick at start", w0, flush=True)
    A, B, resp = [], [], []
    for d in ((a.d, 0, 0), (0, a.d, 0), (-a.d, 0, 0), (0, -a.d, 0)):
        q = kin.ik_rel(q0, d)
        if q is None:
            print("ik fail", d)
            continue
        ga.move(q + [g], speed=6, comp=True)
        time.sleep(0.3)
        w = pa.brick_at_target(ga.fresh("wrist", time.time() + 0.3))
        sx, sy = w["centre"][0] - w0["centre"][0], w["centre"][1] - w0["centre"][1]
        r = w["dist_to_target"]
        A.append(d[:2])
        B.append([sx, sy])
        resp.append(round(float(r), 1))
        ga.move(q0 + [g], speed=6, comp=True)
        print("probe", d, "shift px", round(sx, 1), round(sy, 1), "resp", round(r, 3), flush=True)
    A, B = np.array(A, float), np.array(B, float)
    Mb = np.linalg.lstsq(A, B, rcond=None)[0].T
    c, s = math.cos(psi), math.sin(psi)
    Mt = Mb @ np.array([[c, -s], [s, c]])
    res = A @ Mb.T - B
    out = {"t": time.strftime("%H:%M:%SZ", time.gmtime()), "psi_deg": round(math.degrees(psi), 2),
           "M_base": Mb.round(4).tolist(), "M_tool": Mt.round(4).tolist(), "fit_rms_px": round(float(np.sqrt((res ** 2).mean())), 2),
           "resp": resp, "q_calib": q0, "tcp_calib_mm": p0.round(1).tolist(), "d_mm": a.d}
    json.dump(out, open(kin.CALF, "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
