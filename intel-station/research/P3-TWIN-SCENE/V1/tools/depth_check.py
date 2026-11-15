#!/usr/bin/env python3
"""V1 independent recomputation of the top and close depth numbers (own code path; depth .npy read-only).
Steps: plane fit by least-median-of-squares + trimmed SVD refinement (not the builder's RANSAC), sensitivity to
threshold and seed; brick blobs (height 9.6-30 mm), footprint from in-plane PCA, top plateau from H > 14 mm;
ring-template search on the top map for the two plate centres (rim wall 19.5 mm wide, inner and outer annuli).
Writes outputs/depth_check.json, outputs/mat_frame_top_height.png, outputs/ring_score_top_R82.png."""
import json
import math

import cv2
import numpy as np
from scipy.signal import fftconvolve

DEP = "/home/intel/intel-junction-2026/logs/depth/g1_162903"
FILES = {"top": DEP + "/top_depth.npy", "close": DEP + "/c424x240@15/close_depth.npy"}
OUT = "/home/intel/intel-junction-2026/work/P3-TWIN-SCENE/V1/outputs"
BUILDER_CANDIDATES = {
    "blue_ridge_v1": (-119.55, 317.41),
    "blue_outline_v4_rejected": (-202.94, 226.06),
    "green_ridge_v1": (219.43, 277.6),
    "green_outline_v4": (224.92, 283.74),
}
XR = (-400.0, 400.0)
YR = (-150.0, 450.0)
CELL = 2.0


def points(dep):
    h, w = dep.shape
    f = 0.5 * w / math.tan(math.radians(65.0) / 2.0)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    valid = np.isfinite(dep) & (dep > 0)
    P = np.dstack([(u - cx) * dep / f, (v - cy) * dep / f, dep])
    return P, valid, f, cx, cy


def lmeds(X, rng, trials=2500, sub=60000):
    S = X[rng.choice(len(X), min(sub, len(X)), replace=False)]
    q_at = int(0.3 * len(S))
    best, bq = None, np.inf
    for _ in range(trials):
        s = S[rng.choice(len(S), 3, replace=False)]
        nv = np.cross(s[1] - s[0], s[2] - s[0])
        nn = float(np.linalg.norm(nv))
        if nn < 1e-9:
            continue
        nv = nv / nn
        r = np.abs(S @ nv - float(nv @ s[0]))
        q = float(np.partition(r, q_at)[q_at])
        if q < bq:
            bq, best = q, (nv, -float(nv @ s[0]))
    return best, bq


def refine(X, nv, d, thr, iters=8):
    for _ in range(iters):
        Q = X[np.abs(X @ nv + d) < thr]
        c0 = Q.mean(axis=0)
        nv = np.linalg.svd(Q - c0, full_matrices=False)[2][2]
        d = -float(nv @ c0)
    if d < 0:
        nv, d = -nv, -d
    return nv, d


def fit(X, thr, seed):
    (nv, d), lmed = lmeds(X, np.random.default_rng(seed))
    nv, d = refine(X, nv, d, thr)
    r = X @ nv + d
    inl = np.abs(r) < thr
    rec = {"thr_mm": thr, "seed": seed, "lmeds_q30_mm": round(lmed, 3), "cam_to_mat_mm": round(d, 2),
           "tilt_deg": round(math.degrees(math.acos(min(1.0, abs(float(nv[2]))))), 2),
           "inlier_frac_of_valid": round(float(inl.mean()), 4),
           "inlier_rms_mm": round(float(np.sqrt(np.mean(r[inl] ** 2))), 3)}
    return nv, d, rec


def basis(nv):
    right = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, -1.0, 0.0])
    ex = right - float(right @ nv) * nv
    ex = ex / np.linalg.norm(ex)
    ey = up - float(up @ nv) * nv
    ey = ey - float(ey @ ex) * ex
    ey = ey / np.linalg.norm(ey)
    return ex, ey


def blobs(P, valid, nv, d, ex, ey):
    H = np.where(valid, P @ nv + d, -99.0)
    m = ((H > 9.6) & (H < 30.0)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = [int(v) for v in st[i]]
        if area < 150:
            continue
        sel = lab == i
        Hb = H[sel]
        Pb = P[sel]
        xm, ym = Pb @ ex, Pb @ ey
        A = np.stack([xm, ym], 1)
        _, vecs = np.linalg.eigh(np.cov(A.T))
        a = A @ vecs
        ext = np.percentile(a, 99.5, axis=0) - np.percentile(a, 0.5, axis=0)
        ext_full = a.max(axis=0) - a.min(axis=0)
        top = Hb[Hb > 14.0]
        if top.size < 50:
            continue
        out.append({"bbox_px_xywh": [x, y, bw, bh], "area_px": area,
                    "centre_mat_mm": [round(float(xm.mean()), 2), round(float(ym.mean()), 2)],
                    "long_mm_p99": round(float(ext.max()), 2), "short_mm_p99": round(float(ext.min()), 2),
                    "long_mm_full": round(float(ext_full.max()), 2), "short_mm_full": round(float(ext_full.min()), 2),
                    "top_p50_mm": round(float(np.median(top)), 2),
                    "top_p10_p90_mm": [round(float(np.percentile(top, 10)), 2), round(float(np.percentile(top, 90)), 2)],
                    "n_top_px": int(top.size)})
    out.sort(key=lambda r: -r["area_px"])
    return out


def ann_kernel(r0, r1, cell):
    rr = int(math.ceil(r1 / cell)) + 1
    g = np.arange(-rr, rr + 1) * cell
    XX, YY = np.meshgrid(g, g)
    dist = np.hypot(XX, YY)
    return ((dist >= r0) & (dist < r1)).astype(float)


def raster(xm, ym, hv):
    nx = int(round((XR[1] - XR[0]) / CELL))
    ny = int(round((YR[1] - YR[0]) / CELL))
    ix = np.floor((xm - XR[0]) / CELL).astype(int)
    iy = np.floor((ym - YR[0]) / CELL).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    idx = iy[ok] * nx + ix[ok]
    hc = np.clip(hv[ok], -3.0, 12.0)
    cnt = np.bincount(idx, minlength=nx * ny).astype(float)
    sm = np.bincount(idx, weights=hc, minlength=nx * ny)
    has = cnt > 0
    G = np.zeros(nx * ny)
    G[has] = sm[has] / cnt[has]
    return G.reshape(ny, nx), has.reshape(ny, nx).astype(float), nx, ny


def ring_scores(G, M, R):
    GM = G * M

    def mean_cov(K):
        s = fftconvolve(GM, K, mode="same")
        c = fftconvolve(M, K, mode="same")
        return s / np.maximum(c, 1e-9), c / K.sum()

    mw, cw = mean_cov(ann_kernel(R - 9.75, R + 9.75, CELL))
    mi, ci = mean_cov(ann_kernel(R - 30.0, R - 12.0, CELL))
    mo, co = mean_cov(ann_kernel(R + 12.0, R + 30.0, CELL))
    score = mw - 0.5 * (mi + mo)
    ok = (cw > 0.5) & (ci > 0.3) & (co > 0.3)
    return np.where(ok, score, -99.0)


def peaks(score, nx, ny):
    gx = XR[0] + (np.arange(nx) + 0.5) * CELL
    gy = YR[0] + (np.arange(ny) + 0.5) * CELL
    GX, _ = np.meshgrid(gx, gy)
    res = {}
    for side, mask in (("left_X_lt_50", GX < 50.0), ("right_X_ge_50", GX >= 50.0)):
        s = np.where(mask, score, -99.0)
        k = int(np.argmax(s))
        iy, ix = divmod(k, nx)
        res[side] = {"centre_mm": [round(float(gx[ix]), 1), round(float(gy[iy]), 1)],
                     "score_mm": round(float(s.flat[k]), 3)}
    return res


def score_at(score, xy):
    ix = int(math.floor((xy[0] - XR[0]) / CELL))
    iy = int(math.floor((xy[1] - YR[0]) / CELL))
    r = 3
    sub = score[max(0, iy - r):iy + r + 1, max(0, ix - r):ix + r + 1]
    return round(float(sub.max()), 3)


def main():
    out = {"stage": "twin-scene-v1", "worker": "V1",
           "method": "own: LMedS plane + trimmed SVD refinement; blob PCA footprint; ring template on the top map",
           "builder_claims": {"top_cam_to_mat_mm": 589.46, "top_tilt_deg": 20.32, "close_cam_to_mat_mm": 147.34,
                              "close_tilt_deg": 50.78, "top_valid_frac": 0.8313, "close_valid_frac": 0.6059},
           "cams": {}}
    store = {}
    for cam, path in FILES.items():
        dep = np.load(path).astype(np.float64)
        P, valid, f, cx, cy = points(dep)
        X = P[valid]
        rec = {"file": path, "shape_hw": list(dep.shape), "valid_frac": round(float(valid.mean()), 4),
               "median_valid_depth_mm": round(float(np.median(dep[valid])), 1),
               "min_valid_mm": round(float(dep[valid].min()), 1), "max_valid_mm": round(float(dep[valid].max()), 1),
               "intrinsics_fx_cx_cy": [round(f, 2), cx, cy], "fits": []}
        nv_b, d_b = None, None
        for thr in (1.5, 2.5, 4.0):
            nv, d, r = fit(X, thr, seed=7)
            rec["fits"].append(r)
            if thr == 2.5:
                nv_b, d_b = nv, d
        for seed in (1, 2):
            _, _, r = fit(X, 2.5, seed=seed)
            rec["fits"].append(r)
        ex, ey = basis(nv_b)
        rec["plane_used_thr2p5_seed7"] = {"normal_cam": [round(float(v), 5) for v in nv_b],
                                          "cam_to_mat_mm": round(d_b, 2)}
        rec["blobs"] = blobs(P, valid, nv_b, d_b, ex, ey)
        store[cam] = {"P": P, "valid": valid, "nv": nv_b, "d": d_b, "ex": ex, "ey": ey}
        out["cams"][cam] = rec
    st = store["top"]
    P, valid, nv, d, ex, ey = st["P"], st["valid"], st["nv"], st["d"], st["ex"], st["ey"]
    Hm = P @ nv + d
    G, M, nx, ny = raster(P[valid] @ ex, P[valid] @ ey, Hm[valid])
    per_R = {}
    scores = {}
    for R in np.arange(74.0, 92.0, 2.0):
        s = ring_scores(G, M, float(R))
        scores[float(R)] = s
        pk = peaks(s, nx, ny)
        sep = math.hypot(pk["right_X_ge_50"]["centre_mm"][0] - pk["left_X_lt_50"]["centre_mm"][0],
                         pk["right_X_ge_50"]["centre_mm"][1] - pk["left_X_lt_50"]["centre_mm"][1])
        per_R[str(float(R))] = {"peaks": pk, "separation_mm": round(sep, 2),
                                "score_at_builder_candidates_mm": {k: score_at(s, xy) for k, xy in BUILDER_CANDIDATES.items()}}
    out["ring_template_top"] = {"cell_mm": CELL, "height_clip_mm": [-3, 12], "wall_halfwidth_mm": 9.75,
                                "inner_annulus_mm": "R-30..R-12", "outer_annulus_mm": "R+12..R+30",
                                "R_scan_mm": "74..90 step 2", "grid_x_mm": list(XR), "grid_y_mm": list(YR),
                                "per_R": per_R}
    im = cv2.applyColorMap(((np.clip(G, -3, 12) + 3) / 15.0 * 255).astype(np.uint8), cv2.COLORMAP_JET)
    im[M == 0] = (0, 0, 0)
    im = cv2.flip(im, 0)
    for k, (x, y) in BUILDER_CANDIDATES.items():
        px = int((x - XR[0]) / CELL)
        py = ny - 1 - int((y - YR[0]) / CELL)
        cv2.circle(im, (px, py), 5, (255, 255, 255), 2)
        cv2.putText(im, k, (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.imwrite(OUT + "/mat_frame_top_height.png", im)
    s82 = scores[82.0]
    sim = np.clip((s82 + 2.0) / 10.0, 0.0, 1.0)
    im2 = cv2.flip(cv2.applyColorMap((sim * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), 0)
    cv2.imwrite(OUT + "/ring_score_top_R82.png", im2)
    with open(OUT + "/depth_check.json", "w") as fh:
        json.dump(out, fh, indent=1)
    for cam in ("top", "close"):
        r = out["cams"][cam]
        print(cam, "valid", r["valid_frac"], "shape", r["shape_hw"], "plane", r["plane_used_thr2p5_seed7"])
        for fr in r["fits"]:
            print("  fit", fr)
        for b in r["blobs"]:
            print("  blob", b)
    for R, v in per_R.items():
        print("R", R, "peaks", v["peaks"], "sep", v["separation_mm"], "cand", v["score_at_builder_candidates_mm"])


if __name__ == "__main__":
    main()
