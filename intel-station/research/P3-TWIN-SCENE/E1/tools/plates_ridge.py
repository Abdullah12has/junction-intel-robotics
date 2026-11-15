#!/usr/bin/env python3
"""E1 P3-TWIN-SCENE: plate rims from the top depth map (ridge method, v1).
Reads logs/depth/g1_162903/top_depth.npy only. Helpers (intrinsics, plane fit, RANSAC circle) are imported from measure_scene.py.
Seeds = ring centres placed by eye on overlay_top_depth_v2.png (depth pixels); the seed pixel's mat-plane position is the
initial centre. Rays from the centre find the rim height peak (H of the rim top), a RANSAC circle fits the peaks (rim top radius),
and the same rays give the half-height outer and inner radii (depth-edge halo removed).
Writes plates_ridge_v1.json into OUT. Run with sim-env python -I from the E1 folder.
"""
import json
import math
import sys
import time

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial import cKDTree

sys.path.insert(0, "/home/intel/intel-junction-2026/work/P3-TWIN-SCENE/E1/tools")
import measure_scene as ms  # noqa: E402

OUT = ms.OUT
DEP = ms.CAMS["top"]["depth"]
SEEDS = {"ring_left": (316, 145), "ring_right": (673, 170)}
WIN = 140
PLATES_XY = ms.PLATES_XY


def mat_frame(dep):
    h, w = dep.shape
    fx, fy, cx, cy = ms.intr(w, h)
    valid = np.isfinite(dep) & (dep > 0)
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    Pf = np.dstack([(uu - cx) * dep / fx, (vv - cy) * dep / fy, dep])
    rng = np.random.default_rng(0)
    Pv = Pf[valid]
    sub = Pv[rng.choice(len(Pv), min(40000, len(Pv)), replace=False)]
    nv, d = ms.fit_plane(sub, rng)
    H = Pf @ nv + d
    H[~valid] = np.nan
    right = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, -1.0, 0.0])
    ex = right - (right @ nv) * nv
    ex /= np.linalg.norm(ex)
    ey = up - (up @ nv) * nv
    ey -= (ey @ ex) * ex
    ey /= np.linalg.norm(ey)
    rel = Pf - H[..., None] * nv - (-d * nv)
    return {"H": H, "X": rel @ ex, "Y": rel @ ey, "valid": valid, "d": float(d), "nv": nv}


def ridge_rays(tree, Hv, c, R0, dirs, rad_span=45.0):
    radii = np.arange(max(20.0, R0 - rad_span), R0 + rad_span, 0.5)
    X = c[0] + radii[None, :] * dirs[:, 0:1]
    Y = c[1] + radii[None, :] * dirs[:, 1:2]
    dist, ix = tree.query(np.stack([X.ravel(), Y.ravel()], 1), k=1, distance_upper_bound=2.0)
    g = np.isfinite(dist)
    hv = np.full(dist.shape, -1.0)
    hv[g] = Hv[ix[g]]
    hv = hv.reshape(X.shape)
    hs = uniform_filter1d(hv, size=5, axis=1)
    kp = np.argmax(hs, axis=1)
    return radii, hs, kp


def ridge_plate(Xs, Ys, Hv, c0, R0, rng):
    tree = cKDTree(np.stack([Xs, Ys], 1))
    th = np.deg2rad(np.arange(0.0, 360.0, 2.0))
    dirs = np.stack([np.cos(th), np.sin(th)], 1)
    c = (float(c0[0]), float(c0[1]))
    R_est = R0
    for _ in range(4):
        radii, hs, kp = ridge_rays(tree, Hv, c, R_est, dirs)
        hp = hs[np.arange(hs.shape[0]), kp]
        ok = hp > 4.0
        rp = radii[kp]
        xr = c[0] + rp[ok] * dirs[ok, 0]
        yr = c[1] + rp[ok] * dirs[ok, 1]
        rc = ms.ransac_circle(xr, yr, rng, iters=400, thr=2.0)
        if rc is None:
            break
        c = (rc["cx"], rc["cy"])
        R_est = rc["R"]
    radii, hs, kp = ridge_rays(tree, Hv, c, R_est, dirs)
    r_out, r_in, peaks, ridge_r, ridge_xy = [], [], [], [], []
    for t in range(hs.shape[0]):
        j = int(kp[t])
        hp = float(hs[t, j])
        if hp <= 4.0:
            continue
        half = 0.5 * hp
        jo = j
        while jo < len(radii) - 1 and hs[t, jo] >= half:
            jo += 1
        ji = j
        while ji > 0 and hs[t, ji] >= half:
            ji -= 1
        r_out.append(float(radii[jo]))
        r_in.append(float(radii[ji]))
        peaks.append(hp)
        ridge_r.append(float(radii[j]))
        ridge_xy.append((c[0] + radii[j] * dirs[t, 0], c[1] + radii[j] * dirs[t, 1]))
    rxy = np.array(ridge_xy)
    cov = np.cov((rxy - rxy.mean(0)).T)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    return {"centre_mat_mm": [round(c[0], 2), round(c[1], 2)], "R_rim_top_fit_mm": round(R_est, 2),
            "rim_top_rms_mm": round(float(ms.circle_fit(rxy[:, 0], rxy[:, 1])[3]), 3),
            "rim_top_axis_ratio_pca": round(float(math.sqrt(ev[1] / ev[0])), 4),
            "rays_total": int(hs.shape[0]), "rays_with_rim": int(len(peaks)),
            "rim_peak_H_median_mm": round(float(np.median(peaks)), 2),
            "rim_peak_H_p10_p90_mm": [round(float(np.percentile(peaks, 10)), 2), round(float(np.percentile(peaks, 90)), 2)],
            "R_peak_median_mm": round(float(np.median(ridge_r)), 2),
            "R_outer_half_height_median_mm": round(float(np.median(r_out)), 2),
            "R_outer_half_height_p10_p90_mm": [round(float(np.percentile(r_out, 10)), 2), round(float(np.percentile(r_out, 90)), 2)],
            "R_inner_half_height_median_mm": round(float(np.median(r_in)), 2),
            "wall_width_half_height_mm": round(float(np.median(r_out) - np.median(r_in)), 2),
            "diameter_outer_half_height_mm": round(2 * float(np.median(r_out)), 2)}


def main():
    dep = np.load(DEP).astype(np.float64)
    fr = mat_frame(dep)
    H, X, Y, valid = fr["H"], fr["X"], fr["Y"], fr["valid"]
    h, w = dep.shape
    rng = np.random.default_rng(1)
    out = {"stage": "twin-scene-v1", "method": "plates_ridge v1 (see docstring)", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "source_depth": DEP, "plane": {"cam_to_mat_mm": round(fr["d"], 2)}, "plates": {}}
    for name, (su, sv) in SEEDS.items():
        x0, x1 = max(0, su - WIN), min(w, su + WIN)
        y0, y1 = max(0, sv - WIN), min(h, sv + WIN)
        win = np.zeros((h, w), bool)
        win[y0:y1, x0:x1] = True
        sel = win & valid & np.isfinite(H) & (H > 0.0) & (H < 45.0) & np.isfinite(X) & np.isfinite(Y)
        seed_valid = np.argwhere(valid[sv - 6:sv + 7, su - 6:su + 7])
        if len(seed_valid) == 0:
            out["plates"][name] = {"error": "no depth at seed"}
            continue
        yy, xx = seed_valid[0][0] + sv - 6, seed_valid[0][1] + su - 6
        c0 = (X[yy, xx], Y[yy, xx])
        rec = ridge_plate(X[sel], Y[sel], H[sel], c0, 80.0, rng)
        rec.update({"seed_px_depth": [su, sv], "window_px": [int(x0), int(y0), int(x1), int(y1)],
                    "initial_centre_from_seed_mm": [round(float(c0[0]), 2), round(float(c0[1]), 2)]})
        # row check: horizontal width at the rim centre row, at half of the rim peak
        half = 0.5 * rec["rim_peak_H_median_mm"]
        row = H[yy, :].copy()
        row[~win[yy, :]] = np.nan
        cols = np.where(np.isfinite(row) & (row >= half) & (row < 45.0))[0]
        if cols.size:
            rec["row_check_width_px_at_seed_row"] = int(cols.max() - cols.min() + 1)
            rec["row_check_width_mm_at_seed_row"] = round(float(np.hypot(X[yy, cols.max()] - X[yy, cols.min()],
                                                                        Y[yy, cols.max()] - Y[yy, cols.min()])), 2)
        out["plates"][name] = rec
    names = sorted([k for k in out["plates"] if "centre_mat_mm" in out["plates"][k]],
                   key=lambda k: out["plates"][k]["centre_mat_mm"][0])
    if len(names) == 2:
        b, g = out["plates"][names[0]], out["plates"][names[1]]
        dx = g["centre_mat_mm"][0] - b["centre_mat_mm"][0]
        dy = g["centre_mat_mm"][1] - b["centre_mat_mm"][1]
        dxb, dyb = PLATES_XY["green"][0] - PLATES_XY["blue"][0], PLATES_XY["green"][1] - PLATES_XY["blue"][1]
        out["left_is"] = "blue (smaller mat-x; image-left, matches colour frame)"
        out["separation_mm"] = round(math.hypot(dx, dy), 2)
        out["PLATES_XY_separation_mm"] = round(math.hypot(dxb, dyb), 2)
        out["separation_diff_mm"] = round(math.hypot(dx, dy) - math.hypot(dxb, dyb), 2)
    with open(f"{OUT}/plates_ridge_v1.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "plates"}))
    for k, v in out["plates"].items():
        print(k, {kk: vv for kk, vv in v.items() if kk not in ("window_px",)})


if __name__ == "__main__":
    main()
