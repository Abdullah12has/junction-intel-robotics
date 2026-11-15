"""Eye-in-hand calibration v2 (Opus I-61): adds the scan-pose frames and validates on the RUNTIME path.

v1 (calib.py) fitted only still windows inside joints.csv, which starts after the scan, so the scan height was
extrapolated: the runtime path (scan blob at q_scan -> TCP hover target) was 12.6 mm median / 30.7 mm max off.
v2 adds, per episode, the scan frame observation (q = scan_calib q_scan, blob = episode target_scan_blob, association
fixed) and reports leave-one-episode-out errors of exactly what sort_any will do at runtime.
"""
import json
import math
import os
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

R = "/home/intel/intel-junction-2026"
OUT = f"{R}/work/eyeinhand"
sys.path[:0] = [f"{R}/app/tools"]
import kin  # noqa: E402

TOP_DZ = float(os.environ.get("TOP_DZ", "20.0"))
Z_TOP = kin.MAT_Z + TOP_DZ
COMPAT = {"red": ("red", "orange"), "orange": ("orange", "red", "yellow"), "yellow": ("yellow", "white", "orange"),
          "white": ("white", "yellow"), "green": ("green",), "blue": ("blue",)}
SC = json.load(open(f"{R}/app/tools/scan_calib.json"))
JC = json.load(open(f"{R}/app/tools/jaw_calib.json"))
QS = SC["q_scan"]
SCAN_UP = 70.0

obs = json.load(open(f"{OUT}/obs.json"))
for o in obs:
    o["fixed"] = None
# scan-frame observations with the association known (the brick the loop chose)
eps_json = {}
for o in obs:
    eps_json.setdefault(o["ep"], (o.get("hover_q"), o.get("want")))
for e in sorted(eps_json):
    ef = f"{R}/logs/sort/{e}/episode.json"
    d = json.load(open(ef))
    sb = d.get("target_scan_blob")
    if sb:
        obs.append({"ep": e, "t": 0.0, "q": QS, "frame": f"{R}/logs/sort/{e}/pre_wrist.jpg", "want": eps_json[e][1],
                    "hover_q": d.get("hover_q"), "blobs": [], "fixed": sb["centre"], "scan": True})
eps = sorted({o["ep"] for o in obs})
EPI = {e: i for i, e in enumerate(eps)}
TCP = {}


def tcp_of(q):
    k = tuple(round(float(x), 4) for x in q[:5])
    if k not in TCP:
        p, Rm = kin.fk(list(q)[:5])
        TCP[k] = (np.asarray(p, float), np.asarray(Rm, float))
    return TCP[k]


def rod(r):
    return cv2.Rodrigues(np.asarray(r, float).reshape(3, 1))[0]


def project(par, q, P):
    f, cx, cy, k1 = par[:4]
    p, Rm = tcp_of(q)
    Rbc = Rm @ rod(par[4:7])
    cb = p + Rm @ par[7:10]
    X = Rbc.T @ (np.asarray(P, float) - cb)
    if X[2] <= 1e-3:
        return np.array([1e4, 1e4])
    x, y = X[0] / X[2], X[1] / X[2]
    dd = 1 + k1 * (x * x + y * y)
    return np.array([f * x * dd + cx, f * y * dd + cy])


def pix2mat(par, q, u, z=Z_TOP):
    f, cx, cy, k1 = par[:4]
    xd, yd = (u[0] - cx) / f, (u[1] - cy) / f
    x, y = xd, yd
    for _ in range(8):
        dd = 1 + k1 * (x * x + y * y)
        x, y = xd / dd, yd / dd
    p, Rm = tcp_of(q)
    Rbc = Rm @ rod(par[4:7])
    cb = p + Rm @ par[7:10]
    ray = Rbc @ np.array([x, y, 1.0])
    s = (z - cb[2]) / ray[2]
    return (cb + s * ray)[:2]


def brick(par, e):
    i = 10 + 2 * EPI[e]
    return np.array([par[i], par[i + 1], Z_TOP])


def associate(par, gate, use):
    pairs = []
    for o in obs:
        if o["ep"] not in use:
            continue
        if o["fixed"] is not None:
            pairs.append((o, np.array(o["fixed"], float), 1.0))
            continue
        up = project(par, o["q"], brick(par, o["ep"]))
        ok = COMPAT.get(o["want"])
        best, bd = None, gate
        for b in o["blobs"]:
            if b["plate_frac"] >= 0.3 or (ok and b["colour"] not in ok):
                continue
            dd = math.hypot(b["centre"][0] - up[0], b["centre"][1] - up[1])
            if dd < bd:
                best, bd = b, dd
        if best is not None:
            pairs.append((o, np.array(best["centre"], float), 1.0))
    return pairs


def resid(par, pairs):
    r = []
    for o, u, w in pairs:
        r.extend(w * (project(par, o["q"], brick(par, o["ep"])) - u))
    r += [(par[1] - 320.0) / 60.0, (par[2] - 240.0) / 60.0, par[3] / 0.5]
    return np.array(r)


def fit(par, use, gates=(140, 80, 45, 30, 25)):
    pairs = []
    for g in gates:
        pairs = associate(par, g, use)
        if len(pairs) < 12:
            break
        par = least_squares(resid, par, args=(pairs,), loss="soft_l1", f_scale=4.0, max_nfev=500).x
    return par, pairs


def runtime_tcp_target(par, sb_centre):
    """What sort_any does: brick xy from the scan blob at q_scan, then the TCP hover target via ik_rel."""
    qh = kin.ik_rel(QS, [0, 0, -SCAN_UP])
    bxy = pix2mat(par, QS, sb_centre)
    ph, _ = tcp_of(qh)
    t = bxy - (pix2mat(par, qh, JC["target_px"]) - ph[:2])
    for _ in range(3):
        qt = kin.ik_rel(qh, [t[0] - ph[0], t[1] - ph[1], 0.0])
        if qt is None:
            break
        pt, _ = tcp_of(qt)
        t = bxy - (pix2mat(par, qt, JC["target_px"]) - pt[:2])
    return t


# initial parameters: the v1 solution for the camera, hover TCP for the bricks
v1 = json.load(open(f"{OUT}/wrist_cam_model.json"))
par0 = [v1["f"], v1["cx"], v1["cy"], v1["k1"], *v1["rvec_tcp_cam"], *v1["t_tcp_cam_mm"]]
for e in eps:
    hq = next((o["hover_q"] for o in obs if o["ep"] == e and o.get("hover_q")), None)
    if e in v1["bricks"]:
        par0 += list(v1["bricks"][e])
    else:
        ph, _ = tcp_of(hq or next(o["q"] for o in obs if o["ep"] == e))
        par0 += [ph[0], ph[1]]
par0 = np.array(par0, float)
cnt = {}
for o, _, _ in associate(par0, 45, set(eps)):
    cnt[o["ep"]] = cnt.get(o["ep"], 0) + 1
good = {e for e, n in cnt.items() if n >= 5}
print(f"Z_TOP {Z_TOP:.1f} (mat + {TOP_DZ}); {len(obs)} observations; episodes with >= 5 associated: {len(good)} {sorted(good)}")
par, pairs = fit(par0, good, gates=(60, 40, 30, 25))
res = np.array([np.linalg.norm(project(par, o["q"], brick(par, o["ep"])) - u) for o, u, _ in pairs])
sres = np.array([np.linalg.norm(project(par, o["q"], brick(par, o["ep"])) - u) for o, u, _ in pairs if o.get("scan")])
print(f"fit: {len(pairs)} pairs, reprojection median {np.median(res):.1f} px p90 {np.percentile(res, 90):.1f}; "
      f"scan frames median {np.median(sres):.1f} px max {sres.max():.1f}")
print(f"f {par[0]:.1f} cx {par[1]:.1f} cy {par[2]:.1f} k1 {par[3]:.3f} t {np.round(par[7:10], 1).tolist()}")

rows = []
for e in sorted(good):
    hq = next((o["hover_q"] for o in obs if o["ep"] == e and o.get("hover_q")), None)
    sb = next((o["fixed"] for o in obs if o["ep"] == e and o.get("scan")), None)
    if not hq or sb is None:
        continue
    pe, _ = fit(par.copy(), good - {e}, gates=(30, 25))
    t = runtime_tcp_target(pe, sb)
    ph, _ = tcp_of(hq)
    rows.append((e, float(np.linalg.norm(t - ph[:2]))))
    print(f"LOO-runtime {e}: scan px {np.round(sb, 1).tolist()} -> TCP target {np.round(t, 1).tolist()} vs servo hover "
          f"{np.round(ph[:2], 1).tolist()}: {rows[-1][1]:.1f} mm")
if rows:
    print(f"LOO-runtime median {np.median([r[1] for r in rows]):.1f} mm, max {max(r[1] for r in rows):.1f} mm")
json.dump({"f": par[0], "cx": par[1], "cy": par[2], "k1": par[3], "rvec_tcp_cam": par[4:7].tolist(),
           "t_tcp_cam_mm": par[7:10].tolist(), "z_top_mm": Z_TOP, "n_pairs": len(pairs),
           "reproj_median_px": float(np.median(res)), "loo_runtime_mm": rows, "episodes": sorted(good),
           "bricks": {e: brick(par, e)[:2].tolist() for e in eps if e in good},
           "fitted": "work/eyeinhand/calib2.py"}, open(f"{OUT}/wrist_cam_model_v2.json", "w"), indent=1)
