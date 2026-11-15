"""Eye-in-hand calibration of the wrist camera from natural sort-loop data (Opus I-61, offline).

Unknowns: pinhole f, cx, cy, radial k1; camera pose in the TCP frame (rotation vector, translation mm); the mat
position of the target brick of every episode (x, y; top face at MAT_Z + 20 mm). Observations: target-brick blob
centres in still-arm wrist frames (collect.py) with the joint state from joints.csv. Robust least squares with
iterative data association (blob of the target's colour class nearest the prediction, shrinking gate).

Validation: leave-one-episode-out (brick position from the remaining episodes' camera model, using only the
episode's own scan frame) and the target px under the TCP at the hover.
"""
import json
import math
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

R = "/home/intel/intel-junction-2026"
OUT = f"{R}/work/eyeinhand"
sys.path[:0] = [f"{R}/app/tools"]
import kin  # noqa: E402

TOP_DZ = 20.0  # brick top face above the mat (19.2 body + studs averaged)
Z_TOP = kin.MAT_Z + TOP_DZ
COMPAT = {"red": ("red", "orange"), "orange": ("orange", "red", "yellow"), "yellow": ("yellow", "white", "orange"),
          "white": ("white", "yellow"), "green": ("green",), "blue": ("blue",), None: None}

obs = json.load(open(f"{OUT}/obs.json"))
eps = sorted({o["ep"] for o in obs})
EPI = {e: i for i, e in enumerate(eps)}
TCP = {}
for o in obs:
    p, Rm = kin.fk(o["q"])
    TCP[id(o)] = (np.asarray(p, float), np.asarray(Rm, float))


def rodrigues(r):
    return cv2.Rodrigues(np.asarray(r, float).reshape(3, 1))[0]


def project(par, o, P):
    f, cx, cy, k1 = par[:4]
    Rc = rodrigues(par[4:7])
    tc = par[7:10]
    p, Rm = TCP[id(o)]
    Rbc = Rm @ Rc                 # camera axes in base
    cb = p + Rm @ tc              # camera centre in base (mm)
    X = Rbc.T @ (np.asarray(P, float) - cb)
    if X[2] <= 1e-3:
        return np.array([1e4, 1e4])
    x, y = X[0] / X[2], X[1] / X[2]
    d = 1 + k1 * (x * x + y * y)
    return np.array([f * x * d + cx, f * y * d + cy])


def brick_P(par, e):
    i = 10 + 2 * EPI[e]
    return np.array([par[i], par[i + 1], Z_TOP])


def init_params():
    sc = json.load(open(f"{R}/app/tools/scan_calib.json"))
    jc = json.load(open(f"{R}/app/tools/jaw_calib.json"))
    M = np.array(kin.load_calib()["M_base"], float) if "M_base" in kin.load_calib() else np.array([[-0.775, 2.52], [1.96, 1.19]])
    qh = sc["q_hover"]
    p, Rm = kin.fk(qh)
    ex = -np.r_[M[0], 0.0]
    ex /= np.linalg.norm(ex)
    ey = -np.r_[M[1], 0.0]
    ey -= ex * (ex @ ey)
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    if ez[2] > 0:  # optical axis must point down; keep right-handed by flipping the v axis
        ey = -ey
        ez = np.cross(ex, ey)
    Rbc = np.stack([ex, ey, ez], 1)
    Rc = Rm.T @ Rbc
    f, cx, cy = 312.0, 320.0, 240.0
    tx, ty = jc["target_px"]
    ray = Rbc @ np.array([(tx - cx) / f, (ty - cy) / f, 1.0])
    P = np.array([p[0], p[1], Z_TOP])
    s = (P[2] - (p[2] + 54.0)) / ray[2]  # camera ~54 mm above the TCP (size model 9886/short - 33)
    cb = P - s * ray
    tc = Rm.T @ (cb - p)
    par = [f, cx, cy, 0.0, *cv2.Rodrigues(Rc)[0].ravel(), *tc]
    for e in eps:
        hq = next((o["hover_q"] for o in obs if o["ep"] == e and o.get("hover_q")), None)
        q0 = hq or next(o["q"] for o in obs if o["ep"] == e)
        ph, _ = kin.fk(q0)
        par += [ph[0], ph[1]]
    return np.array(par, float)


def associate(par, gate, use=None):
    pairs = []
    for o in obs:
        if use is not None and o["ep"] not in use:
            continue
        up = project(par, o, brick_P(par, o["ep"]))
        ok = COMPAT.get(o["want"]) if o["want"] in COMPAT else None
        best, bd = None, gate
        for b in o["blobs"]:
            if b["plate_frac"] >= 0.3 or (ok and b["colour"] not in ok):
                continue
            d = math.hypot(b["centre"][0] - up[0], b["centre"][1] - up[1])
            if d < bd:
                best, bd = b, d
        if best is not None:
            pairs.append((o, np.array(best["centre"], float)))
    return pairs


def resid(par, pairs):
    r = []
    for o, u in pairs:
        r.extend(project(par, o, brick_P(par, o["ep"])) - u)
    # weak priors keep the intrinsics physical when few views constrain them
    r += [(par[1] - 320.0) / 40.0, (par[2] - 240.0) / 40.0, par[3] / 0.5]
    return np.array(r)


def fit(par, use=None, gates=(140, 80, 45, 30, 25)):
    pairs = []
    for g in gates:
        pairs = associate(par, g, use)
        if len(pairs) < 12:
            break
        sol = least_squares(resid, par, args=(pairs,), loss="soft_l1", f_scale=4.0, max_nfev=400)
        par = sol.x
    return par, pairs


par0 = init_params()
pairs0 = associate(par0, 140)
print(f"{len(obs)} observations, {len(eps)} episodes, {len(pairs0)} associated at init")
par, pairs = fit(par0)
res = np.array([np.linalg.norm(project(par, o, brick_P(par, o["ep"])) - u) for o, u in pairs])
print(f"fit: {len(pairs)} pairs, reprojection median {np.median(res):.1f} px, p90 {np.percentile(res, 90):.1f}, max {res.max():.1f}")
f, cx, cy, k1 = par[:4]
Rc = rodrigues(par[4:7])
print(f"f {f:.1f} cx {cx:.1f} cy {cy:.1f} k1 {k1:.3f} cam-in-TCP t {np.round(par[7:10], 1).tolist()} mm, optical axis in TCP {np.round(Rc[:, 2], 3).tolist()}")

# hover check: where does the point under the TCP project at the hover?
sc = json.load(open(f"{R}/app/tools/scan_calib.json"))
o_h = {"q": sc["q_hover"]}
p, Rm = kin.fk(sc["q_hover"])
TCP[id(o_h)] = (np.asarray(p, float), np.asarray(Rm, float))
print("point under TCP at hover projects to", np.round(project(par, o_h, [p[0], p[1], Z_TOP]), 1).tolist(), "(servo target 403.1, 268)")


def ray_to_plane(par, o, u, z=Z_TOP):
    f, cx, cy, k1 = par[:4]
    x, y = (u[0] - cx) / f, (u[1] - cy) / f
    for _ in range(5):  # undo k1
        d = 1 + k1 * (x * x + y * y)
        x, y = (u[0] - cx) / f / d, (u[1] - cy) / f / d
    p, Rm = TCP[id(o)]
    Rbc = Rm @ rodrigues(par[4:7])
    cb = p + Rm @ par[7:10]
    ray = Rbc @ np.array([x, y, 1.0])
    s = (z - cb[2]) / ray[2]
    return (cb + s * ray)[:2]


# leave-one-episode-out: camera from the other episodes, brick from this episode's FIRST still frame (the scan)
loo = []
for e in eps:
    others = [x for x in eps if x != e]
    pe, _ = fit(par.copy(), use=set(others), gates=(40, 25))
    mine = [(o, u) for o, u in pairs if o["ep"] == e]
    if not mine:
        continue
    o1, u1 = min(mine, key=lambda ou: ou[0]["t"])
    xy = ray_to_plane(pe, o1, u1)
    ref = brick_P(par, e)[:2]
    loo.append((e, float(np.linalg.norm(xy - ref)), len(mine)))
for e, err, n in loo:
    print(f"LOO {e}: first-frame estimate off by {err:.1f} mm ({n} frames)")
if loo:
    print(f"LOO median {np.median([x[1] for x in loo]):.1f} mm, max {max(x[1] for x in loo):.1f} mm")
json.dump({"f": f, "cx": cx, "cy": cy, "k1": k1, "rvec_tcp_cam": par[4:7].tolist(), "t_tcp_cam_mm": par[7:10].tolist(),
           "z_top_mm": Z_TOP, "n_pairs": len(pairs), "reproj_median_px": float(np.median(res)),
           "reproj_p90_px": float(np.percentile(res, 90)), "loo": loo,
           "bricks": {e: brick_P(par, e)[:2].tolist() for e in eps}},
          open(f"{OUT}/wrist_cam_model.json", "w"), indent=1)
