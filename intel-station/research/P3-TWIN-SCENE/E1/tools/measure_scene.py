#!/usr/bin/env python3
"""E1 P3-TWIN-SCENE: scene measurements from recorded data only (v4).
Inputs (read-only): logs/depth/g1_162903 (top and close depth, colour), logs/depth/servo stills.
Outputs (written only into OUT): scene_measurements_raw_v4.json, overlays, crops.
Plate windows are seeds placed by eye on overlay_top_depth_v2.png (depth pixels): blue ring (316,145), green ring (673,170).
Inside each window the plate is fitted from data: RANSAC circle on the 4-45 mm outline (mat-plane coordinates), then the
radial height profile gives the rim peak and the half-height outer and inner radii (removes the depth-edge halo).
Run: nice -n 19 env OMP_NUM_THREADS=4 /home/intel/intel-junction-2026/envs/sim-env/bin/python -I tools/measure_scene.py
"""
import json
import math
import os
import time

import cv2
import numpy as np

OUT = "/home/intel/intel-junction-2026/work/P3-TWIN-SCENE/E1"
G = "/home/intel/intel-junction-2026/logs/depth/g1_162903"
SV = "/home/intel/intel-junction-2026/logs/depth/servo"
HFOV_DEPTH = 65.0
PLATES_XY = {"blue": (187.1, -86.1), "green": (194.6, 218.3)}
PLATE_SEEDS_DEPTH_PX = {"ring_left": (316, 145), "ring_right": (673, 170)}
PLATE_WIN_HALF_PX = 140
CAMS = {
    "top": {"depth": f"{G}/top_depth.npy", "color": f"{G}/top_color.jpg"},
    "close": {"depth": f"{G}/c424x240@15/close_depth.npy", "color": f"{G}/close_color.jpg"},
}
STILLS = {
    "servo_163312_p0_base_top": f"{SV}/163312_p0_base/live_top.jpg",
    "servo_163312_p0_base_close": f"{SV}/163312_p0_base/live_close.jpg",
    "servo_163312_p0_base_wrist": f"{SV}/163312_p0_base/live_wrist.jpg",
    "servo_165016_a6_lift_top": f"{SV}/165016_a6_lift/live_top.jpg",
    "servo_165016_a6_lift_close": f"{SV}/165016_a6_lift/live_close.jpg",
    "servo_165016_a6_lift_wrist": f"{SV}/165016_a6_lift/live_wrist.jpg",
}
ROI = [  # (colour image key, name, x0, y0, x1, y1) in colour-image pixels, placed by eye; crops saved to check
    ("close", "close_blue_2x2_left_face_dark_tape", 318, 240, 356, 300),
    ("close", "close_blue_2x2_studs_top", 345, 205, 430, 240),
    ("close", "close_yellow_2x2_tape_area", 545, 170, 615, 232),
    ("close", "close_white_2x2_lower_right", 520, 330, 600, 380),
    ("close", "close_mat_lower_left", 0, 300, 200, 480),
    ("close", "close_mat_centre", 180, 250, 300, 330),
    ("top", "top_mat_centre", 230, 300, 330, 340),
    ("top", "top_blue_plate_interior", 205, 115, 275, 175),
    ("top", "top_green_plate_interior", 530, 140, 600, 210),
]


def intr(w, h):
    f = 0.5 * w / math.tan(math.radians(HFOV_DEPTH) / 2.0)
    return f, f, (w - 1) / 2.0, (h - 1) / 2.0


def fit_plane(P, rng, iters=800, thr=2.5):
    best, bestc = None, -1
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        nv = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(nv)
        if nn < 1e-9:
            continue
        nv = nv / nn
        d = -float(nv @ s[0])
        c = int((np.abs(P @ nv + d) < thr).sum())
        if c > bestc:
            best, bestc = (nv, d), c
    nv, d = best
    for _ in range(3):
        inl = np.abs(P @ nv + d) < thr
        Q = P[inl]
        c0 = Q.mean(0)
        nv = np.linalg.svd(Q - c0, full_matrices=False)[2][2]
        d = -float(nv @ c0)
    if d < 0:
        nv, d = -nv, -d
    return nv, d


def circle_fit(x, y):
    A = np.stack([x, y, np.ones_like(x)], 1)
    a, b, c = np.linalg.lstsq(A, x * x + y * y, rcond=None)[0]
    cx, cy = a / 2.0, b / 2.0
    r2 = c + cx * cx + cy * cy
    if r2 <= 0:
        return float(cx), float(cy), 0.0, 1e9
    r = math.sqrt(r2)
    res = np.hypot(x - cx, y - cy) - r
    return float(cx), float(cy), float(r), float(np.sqrt(np.mean(res ** 2)))


def ring_profile(r, hh, R0, step=0.5):
    edges = np.arange(max(5.0, R0 - 60.0), R0 + 60.0 + step, step)
    centres = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(r, edges) - 1
    med = np.full(len(centres), np.nan)
    for k in range(len(centres)):
        s = hh[idx == k]
        if s.size >= 8:
            med[k] = np.median(s)
    sm = med.copy()
    for k in range(1, len(med) - 1):
        w3 = med[k - 1:k + 2]
        if np.isfinite(w3).sum() >= 2:
            sm[k] = np.nanmedian(w3)
    kpk = int(np.nanargmax(np.where(np.isfinite(sm), sm, -1e9)))
    hpk = float(sm[kpk])
    half = 0.5 * hpk
    k_out = kpk
    while k_out < len(sm) - 1 and np.isfinite(sm[k_out]) and sm[k_out] >= half:
        k_out += 1
    k_in = kpk
    while k_in > 0 and np.isfinite(sm[k_in]) and sm[k_in] >= half:
        k_in -= 1
    r_out = float(centres[min(k_out, len(centres) - 1)])
    r_in = float(centres[max(k_in, 0)])
    return {"rim_peak_H_mm": round(hpk, 2), "rim_peak_radius_mm": round(float(centres[kpk]), 2),
            "R_outer_half_height_mm": round(r_out, 2), "R_inner_half_height_mm": round(r_in, 2),
            "rim_wall_width_half_height_mm": round(r_out - r_in, 2),
            "profile_bins_mm": [round(float(c), 1) for c in centres[::4]],
            "profile_median_H_mm": [round(float(v), 2) if np.isfinite(v) else None for v in sm[::4]]}


def ransac_circle(x, y, rng, iters=500, thr=3.0):
    n = len(x)
    if n < 12:
        return None
    best_inl, best_n = None, 0
    for _ in range(iters):
        i1, i2, i3 = rng.choice(n, 3, replace=False)
        x1, y1, x2, y2, x3, y3 = x[i1], y[i1], x[i2], y[i2], x[i3], y[i3]
        dd = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if abs(dd) < 1e-9:
            continue
        s1, s2, s3 = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2, x3 * x3 + y3 * y3
        ux = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / dd
        uy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / dd
        R = math.hypot(x1 - ux, y1 - uy)
        if R < 35.0 or R > 140.0:
            continue
        inl = np.abs(np.hypot(x - ux, y - uy) - R) < thr
        k = int(inl.sum())
        if k > best_n:
            best_n, best_inl = k, inl
    if best_inl is None or best_n < 12:
        return None
    cx, cy, R, rms = circle_fit(x[best_inl], y[best_inl])
    return {"cx": cx, "cy": cy, "R": R, "rms": rms, "n_inl": best_n, "n_pts": int(n),
            "inlier_frac": round(best_n / n, 3)}


def overlay_depth(cam, H, plates, bricks, tag):
    Hn = np.nan_to_num(H, nan=-5.0)
    vis = np.clip(Hn / 45.0, 0.0, 1.0)
    col = cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
    col[Hn < -3.0] = (60, 60, 60)
    for p in plates:
        x0, y0, x1, y1 = p["window_px"]
        cv2.rectangle(col, (x0, y0), (x1, y1), (255, 255, 255), 1)
        cv2.putText(col, f"{p['seed']} Rout {p['R_outer_half_height_mm']:.0f}", (x0 + 3, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    for k, b in enumerate(bricks):
        x, y, bw, bh = b["bbox_px"]
        cv2.rectangle(col, (x, y), (x + bw, y + bh), (0, 0, 0), 1)
        cv2.putText(col, f"B{k} {b['footprint_short_mm']:.0f}x{b['footprint_long_mm']:.0f}", (x, max(12, y - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.imwrite(f"{OUT}/overlay_{cam}_depth_{tag}.png", col)


def analyse(cam, tag):
    rng = np.random.default_rng(0)
    dep = np.load(CAMS[cam]["depth"]).astype(np.float64)
    h, w = dep.shape
    fx, fy, cx, cy = intr(w, h)
    valid = np.isfinite(dep) & (dep > 0)
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    Pf = np.dstack([(uu - cx) * dep / fx, (vv - cy) * dep / fy, dep])
    Pv = Pf[valid]
    sub = Pv[rng.choice(len(Pv), min(40000, len(Pv)), replace=False)]
    nv, d = fit_plane(sub, rng)
    H = Pf @ nv + d
    H[~valid] = np.nan
    Hn = np.nan_to_num(H, nan=-99.0)
    inl = valid & (np.abs(Hn) < 2.5)
    right = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, -1.0, 0.0])
    ex = right - (right @ nv) * nv
    ex /= np.linalg.norm(ex)
    ey = up - (up @ nv) * nv
    ey -= (ey @ ex) * ex
    ey /= np.linalg.norm(ey)
    P0 = -d * nv
    rel = Pf - H[..., None] * nv - P0
    Xm = rel @ ex
    Ym = rel @ ey
    pa = (dep / fx) * (dep / fy)
    plane = {
        "cam_to_mat_mm": round(float(d), 2),
        "tilt_deg_normal_vs_optical_axis": round(math.degrees(math.acos(min(1.0, abs(float(nv[2]))))), 2),
        "normal_cam_frame": [round(float(v), 5) for v in nv],
        "intrinsics_fx_fy_cx_cy_nominal": [round(fx, 2), round(fy, 2), round(cx, 2), round(cy, 2)],
        "hfov_assumed_deg": HFOV_DEPTH,
        "valid_frac": round(float(valid.mean()), 4),
        "inlier_frac_thr2p5mm": round(float(inl.sum() / valid.sum()), 4),
        "inlier_rms_mm": round(float(np.sqrt(np.mean(H[inl] ** 2))), 3),
        "inlier_median_mm": round(float(np.median(H[inl])), 3),
        "inlier_p5_p95_mm": [round(float(np.percentile(H[inl], 5)), 2), round(float(np.percentile(H[inl], 95)), 2)],
    }
    plates, plate_rejects = [], []
    if cam == "top":
        for seed_name, (su, sv) in PLATE_SEEDS_DEPTH_PX.items():
            x0, x1 = max(0, su - PLATE_WIN_HALF_PX), min(w, su + PLATE_WIN_HALF_PX)
            y0, y1 = max(0, sv - PLATE_WIN_HALF_PX), min(h, sv + PLATE_WIN_HALF_PX)
            win = np.zeros((h, w), bool)
            win[y0:y1, x0:x1] = True
            m = ((Hn > 4.0) & (Hn < 45.0) & win).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            if hier is None or len(cnts) == 0:
                plate_rejects.append({"seed": seed_name, "reason": "no contours in window"})
                continue
            hier = hier[0]
            outers, holes = [], []
            for k in range(len(cnts)):
                if cv2.contourArea(cnts[k]) < 150:
                    continue
                p = cnts[k][:, 0, :]
                xm, ym = Xm[p[:, 1], p[:, 0]], Ym[p[:, 1], p[:, 0]]
                ok = np.isfinite(xm) & np.isfinite(ym)
                if ok.sum() < 40:
                    continue
                rc = ransac_circle(xm[ok], ym[ok], rng)
                if rc is None:
                    continue
                rc["area_px"] = int(cv2.contourArea(cnts[k]))
                (outers if hier[k][3] == -1 else holes).append(rc)
            good = [o for o in sorted(outers, key=lambda r: -r["n_inl"])
                    if 40.0 <= o["R"] <= 130.0 and o["rms"] <= 3.0 and o["inlier_frac"] >= 0.3]
            if not good:
                plate_rejects.append({"seed": seed_name, "reason": "no outer circle R 40-130 mm, rms<=3, inliers>=0.3",
                                      "n_outer_candidates": len(outers)})
                continue
            o = good[0]
            rad = np.hypot(Xm - o["cx"], Ym - o["cy"])
            sel = win & valid & np.isfinite(H) & (H < 45.0) & np.isfinite(rad)
            prof = ring_profile(rad[sel], H[sel], o["R"])
            pr = {"seed": seed_name, "window_px": [int(x0), int(y0), int(x1), int(y1)],
                  "cx_mm": round(o["cx"], 2), "cy_mm": round(o["cy"], 2),
                  "R_outline_fit_mm": round(o["R"], 2), "outline_rms_mm": round(o["rms"], 3),
                  "outline_inlier_frac": o["inlier_frac"], "outline_pts": o["n_pts"]}
            pr.update(prof)
            pr["diameter_outer_half_height_mm"] = round(2 * prof["R_outer_half_height_mm"], 2)
            pr["R_outer_half_height_mm_alias"] = prof["R_outer_half_height_mm"]
            inside = valid & (rad < 0.6 * prof["R_inner_half_height_mm"])
            if inside.any():
                pr["floor_H_p50_mm"] = round(float(np.median(H[inside])), 2)
                pr["floor_H_p10_p90_mm"] = [round(float(np.percentile(H[inside], 10)), 2),
                                            round(float(np.percentile(H[inside], 90)), 2)]
            plates.append(pr)
    plates.sort(key=lambda r: r["cx_mm"])
    # bricks: blobs between half brick height (9.6 mm) and 30 mm; footprint from the whole blob at half height
    mb = ((Hn > 9.6) & (Hn < 30.0)).astype(np.uint8)
    mb = cv2.morphologyEx(mb, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    nb, labb, stb, _ = cv2.connectedComponentsWithStats(mb, 8)
    min_area = 300 if cam == "top" else 800
    bricks, brick_rejects = [], {}
    for i in range(1, nb):
        x, y, bw, bh, area = stb[i]
        if area < min_area or bw > 200 or bh > 200:
            continue
        sel = labb == i
        mx = float(np.nanmean(Xm[sel]))
        my = float(np.nanmean(Ym[sel]))
        if any(math.hypot(mx - p["cx_mm"], my - p["cy_mm"]) < p["R_outer_half_height_mm"] + 15.0 for p in plates):
            brick_rejects["inside_plate_radius"] = brick_rejects.get("inside_plate_radius", 0) + 1
            continue
        ok_all = sel & np.isfinite(Xm) & np.isfinite(Ym)
        pts = np.stack([Xm[ok_all], Ym[ok_all]], 1).astype(np.float32)
        if len(pts) < 50:
            brick_rejects["too_few_points"] = brick_rejects.get("too_few_points", 0) + 1
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
        short, long_ = sorted([float(rw), float(rh)])
        plateau = sel & (Hn > 17.5) & (Hn < 21.0)
        if plateau.sum() < 40:
            brick_rejects["no_top_plateau"] = brick_rejects.get("no_top_plateau", 0) + 1
            continue
        h_top = float(np.median(H[plateau]))
        blob_area_mm2 = float(np.nansum(pa[ok_all]))
        fill = blob_area_mm2 / max(short * long_, 1e-6)
        why = None
        if not (24.0 <= short <= 40.0 and 24.0 <= long_ <= 75.0):
            why = "footprint outside 24-40 x 24-75 mm"
        elif long_ / max(short, 1e-6) > 2.4:
            why = "aspect > 2.4"
        elif not (17.8 <= h_top <= 20.8):
            why = "top plateau median outside 17.8-20.8 mm"
        elif fill < 0.6:
            why = "fill ratio < 0.6 (ragged blob)"
        if why is not None:
            brick_rejects[why] = brick_rejects.get(why, 0) + 1
            continue
        ratio = long_ / max(short, 1e-6)
        rec = {"bbox_px": [int(x), int(y), int(bw), int(bh)], "area_px": int(area),
               "centre_mat_mm": [round(mx, 1), round(my, 1)],
               "footprint_short_mm": round(short, 2), "footprint_long_mm": round(long_, 2),
               "aspect_long_over_short": round(ratio, 3),
               "blob_area_halfheight_mm2": round(blob_area_mm2, 1), "fill_ratio": round(fill, 3),
               "H_top_plateau_p50_mm": round(h_top, 2), "H_top_plateau_px": int(plateau.sum()),
               "H_top_plateau_p10_p90_mm": [round(float(np.percentile(H[plateau], 10)), 2),
                                            round(float(np.percentile(H[plateau], 90)), 2)],
               "H_hist_0p5mm_bins_9p5_to_30": [int(c) for c in np.histogram(
                   H[sel & np.isfinite(H)], bins=np.arange(9.5, 30.01, 0.5))[0]],
               "kind_guess": "2x2" if ratio < 1.35 else ("2x4" if 1.8 <= ratio <= 2.2 else "unknown")}
        bricks.append(rec)
    bricks.sort(key=lambda r: -r["area_px"])
    overlay_depth(cam, H, plates, bricks, tag)
    return {"plane": plane, "plates": plates, "plate_rejects": plate_rejects, "bricks": bricks,
            "brick_reject_counts": brick_rejects, "intrinsics_used": {"fx": fx, "cx": cx, "cy": cy, "w": w, "h": h}}


def colour_blobs(img, min_area=150):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    Hh, Ss, Vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((Ss >= 90) & (Vv >= 60)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    rows = []
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if area < min_area:
            continue
        sel = lab == i
        hh, ss, vv = Hh[sel], Ss[sel], Vv[sel]
        rgb = img[sel][:, ::-1].astype(np.float64)
        topm = vv >= np.percentile(vv, 70)
        rows.append({
            "bbox_px": [int(x), int(y), int(bw), int(bh)], "area_px": int(area),
            "hue_med": round(float(np.median(hh)), 1), "sat_med": round(float(np.median(ss)), 1),
            "val_med": round(float(np.median(vv)), 1),
            "hue_p10_p90": [round(float(np.percentile(hh, 10)), 1), round(float(np.percentile(hh, 90)), 1)],
            "rgb_mean": [round(float(c), 1) for c in rgb.mean(0)],
            "rgb_mean_brightest30pct_top_proxy": [round(float(c), 1) for c in rgb[topm].mean(0)],
            "rgb_mean_darkest70pct_side_proxy": [round(float(c), 1) for c in rgb[~topm].mean(0)] if (~topm).any() else None,
        })
    rows.sort(key=lambda r: -r["area_px"])
    return rows


def lighting(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    Ss, Vv = hsv[..., 1], hsv[..., 2]
    rgb = img[..., ::-1].astype(np.float64)
    mat = (Vv < 70) & (Ss < 110)
    wht = (Ss < 40) & (Vv > 170)
    out = {"mat_px_frac": round(float(mat.mean()), 4),
           "mat_V_p5_p50_p95": [float(np.percentile(Vv[mat], q)) for q in (5, 50, 95)],
           "mat_rgb_mean": [round(float(c), 1) for c in rgb[mat].mean(0)]}
    if wht.sum() > 50:
        wm = rgb[wht].mean(0)
        out.update({"white_px_frac": round(float(wht.mean()), 4),
                    "white_rgb_mean": [round(float(c), 1) for c in wm],
                    "white_V_p50": float(np.median(Vv[wht])),
                    "grayworld_gain_R_to_match_G": round(float(wm[1] / wm[0]), 3),
                    "grayworld_gain_B_to_match_G": round(float(wm[1] / wm[2]), 3)})
    Y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    out["luma_mean_all"] = round(float(Y.mean()), 1)
    out["luma_p50_p95_all"] = [round(float(np.percentile(Y, 50)), 1), round(float(np.percentile(Y, 95)), 1)]
    return out


def roi_stats(img, box):
    x0, y0, x1, y1 = box
    c = img[y0:y1, x0:x1]
    rgb = c[..., ::-1].reshape(-1, 3).astype(np.float64)
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64)
    return {"box_xyxy": [x0, y0, x1, y1], "n_px": int(rgb.shape[0]),
            "rgb_mean": [round(float(v), 1) for v in rgb.mean(0)], "rgb_std": [round(float(v), 1) for v in rgb.std(0)],
            "hsv_mean_opencv": [round(float(v), 1) for v in hsv.mean(0)], "hsv_std": [round(float(v), 1) for v in hsv.std(0)]}


def overlay_colour(tag, img, rows):
    o = img.copy()
    for k, r in enumerate(rows[:25]):
        x, y, bw, bh = r["bbox_px"]
        cv2.rectangle(o, (x, y), (x + bw, y + bh), (255, 255, 255), 1)
        cv2.putText(o, str(k), (x, max(12, y - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imwrite(f"{OUT}/overlay_{tag}_colour_v4.png", o)


def save_crop(img, box, scale, path):
    x0, y0, x1, y1 = box
    c = cv2.resize(img[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(path, c)


def main():
    tag = "v4"
    os.makedirs(f"{OUT}/crops", exist_ok=True)
    res = {"stage": "twin-scene-v1", "version": tag, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "inputs": {"depth_top": CAMS["top"]["depth"], "depth_close": CAMS["close"]["depth"],
                      "colour_top": CAMS["top"]["color"], "colour_close": CAMS["close"]["color"]},
           "depth": {}, "colour": {}, "lighting": {}, "tape_roi": {}}
    for cam in ("top", "close"):
        res["depth"][cam] = analyse(cam, tag)
    pl = res["depth"]["top"]["plates"]
    if len(pl) == 2:
        b, g = pl[0], pl[1]
        dx, dy = g["cx_mm"] - b["cx_mm"], g["cy_mm"] - b["cy_mm"]
        dxb, dyb = PLATES_XY["green"][0] - PLATES_XY["blue"][0], PLATES_XY["green"][1] - PLATES_XY["blue"][1]
        res["plates_vs_PLATES_XY"] = {
            "left_plate_centre_mat_mm": [b["cx_mm"], b["cy_mm"]], "right_plate_centre_mat_mm": [g["cx_mm"], g["cy_mm"]],
            "measured_separation_mm": round(math.hypot(dx, dy), 2),
            "PLATES_XY_separation_mm": round(math.hypot(dxb, dyb), 2),
            "separation_diff_mm": round(math.hypot(dx, dy) - math.hypot(dxb, dyb), 2),
            "measured_vector_angle_from_mat_x_deg": round(math.degrees(math.atan2(dy, dx)), 2),
            "PLATES_XY_vector_angle_from_base_y_deg": round(math.degrees(math.atan2(dxb, dyb)), 2),
            "note": "Only the separation is frame-free. Angles use different axes (mat x vs base y)."}
    tops = {}
    for cam in ("top", "close"):
        img = cv2.imread(CAMS[cam]["color"])
        rows = colour_blobs(img)
        res["colour"][cam] = {"image": CAMS[cam]["color"], "shape_hw": list(img.shape[:2]), "blobs": rows[:25]}
        res["lighting"][cam] = lighting(img)
        overlay_colour(cam, img, rows)
        tops[cam] = img
    for name, path in STILLS.items():
        img = cv2.imread(path)
        if img is None:
            res["colour"][name] = {"image": path, "error": "unreadable"}
            continue
        res["colour"][name] = {"image": path, "shape_hw": list(img.shape[:2]), "blobs": colour_blobs(img)[:12]}
        res["lighting"][name] = lighting(img)
    for key, nm, x0, y0, x1, y1 in ROI:
        res["tape_roi"][nm] = roi_stats(tops[key], (x0, y0, x1, y1))
    save_crop(tops["close"], (300, 170, 470, 340), 3, f"{OUT}/crops/close_blue_brick_context_x3.png")
    save_crop(tops["top"], (330, 190, 510, 360), 3, f"{OUT}/crops/top_bricks_context_x3.png")
    with open(f"{OUT}/scene_measurements_raw_{tag}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    for cam in ("top", "close"):
        print(cam, "plane", {k: res["depth"][cam]["plane"][k] for k in ("cam_to_mat_mm", "tilt_deg_normal_vs_optical_axis",
                                                                       "inlier_rms_mm", "inlier_frac_thr2p5mm")})
        for p in res["depth"][cam]["plates"]:
            print(cam, "plate", {kk: p.get(kk) for kk in ("seed", "cx_mm", "cy_mm", "R_outline_fit_mm", "rim_peak_H_mm",
                                                          "R_outer_half_height_mm", "R_inner_half_height_mm",
                                                          "rim_wall_width_half_height_mm", "floor_H_p50_mm")})
        print(cam, "plate_rejects", res["depth"][cam]["plate_rejects"])
        print(cam, "brick_reject_counts", res["depth"][cam]["brick_reject_counts"], "n_bricks", len(res["depth"][cam]["bricks"]))
        for k, b in enumerate(res["depth"][cam]["bricks"][:14]):
            print(cam, "brick", k, {kk: b[kk] for kk in ("bbox_px", "centre_mat_mm", "footprint_short_mm", "footprint_long_mm",
                                                       "H_top_plateau_p50_mm", "fill_ratio", "kind_guess")})
    print("plates_vs_PLATES_XY", res.get("plates_vs_PLATES_XY"))


if __name__ == "__main__":
    main()
