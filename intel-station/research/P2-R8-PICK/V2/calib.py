"""Verifier V2 (P2-R8-PICK): real-data calibration and offline pick check, dataset 6d6bdce3, top view.
Usage: python3 -I calib.py HERE   (HERE holds episodes.json and frames/). Writes calib.json, offline_eval.json, overlays.
Labels: grasped brick = the start-frame blob that has no blob within 18 px in the end frame (ambiguous ones resolved by
the fitted prior homography, flagged). Every pair is checked on an overlay sheet by a person before it is trusted."""
import json
import math
import os
import sys

import cv2
import numpy as np
import pyarrow.parquet as pq

HERE = sys.argv[1]
ROOT = "/home/intel/intel-junction-2026/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af"
eps = json.load(open(os.path.join(HERE, "episodes.json")))
PROJ = json.load(open("/home/intel/intel-junction-2026/app/motion/project_robots.json"))
CAL = [c for c in PROJ if c["name"] == "SO101 Follower"][0]["payload"]["calibration"]
BODY = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
SCALE = [(CAL[j]["range_max"] - CAL[j]["range_min"]) / 200.0 * 360.0 / 4095.0 for j in BODY]
LIM = [(-1.91986, 1.91986), (-1.74533, 1.74533), (-1.69, 1.69), (-1.65806, 1.65806), (-2.74385, 2.84121)]
LO = np.array([a for a, _ in LIM])
HI = np.array([b for _, b in LIM])
HYP = {
    "A_no_sign_no_offset": ((1, 1, 1, 1, 1), (0, 0, 0, 0, 0)),
    "B_studio_sign_no_offset": ((1, -1, 1, 1, 1), (0, 0, 0, 0, 0)),
    "C_studio_sign_and_offset": ((1, -1, 1, 1, 1), (0, 90, 90, 0, 0)),
    "D_offset_no_sign": ((1, 1, 1, 1, 1), (0, 90, 90, 0, 0)),
    "E_pan_flip_and_lift_sign": ((-1, -1, 1, 1, 1), (0, 0, 0, 0, 0)),
}
# URDF chain (W1 copy of so101.urdf, lines 293-303 and 333-439): all revolute axes are local z
JOINTS = [
    ((0.0388353, -8.97657e-09, 0.0624), (3.14159, 4.18253e-17, -3.14159)),
    ((-0.0303992, -0.0182778, -0.0542), (-1.5708, -1.5708, 0)),
    ((-0.11257, -0.028, 1.73763e-16), (-3.63608e-16, 8.74301e-16, 1.5708)),
    ((-0.1349, 0.0052, 3.62355e-17), (4.02456e-15, 8.67362e-16, -1.5708)),
    ((5.55112e-17, -0.0611, 0.0181), (1.5708, 0.0486795, 3.14159)),
]
ARM_BOX = (330, 0, 480, 170)  # top-centre region where the arm sits at home (start frames)


def rot_rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def tf(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = rot_rpy(*rpy)
    T[:3, 3] = xyz
    return T


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    T[0, 0], T[0, 1], T[1, 0], T[1, 1] = c, -s, s, c
    return T


TCP_T = tf((-0.0079, -0.000218121, -0.0981274), (0, 3.14159, 0))


def fk_T(q):
    T = np.eye(4)
    for (xyz, rpy), a in zip(JOINTS, q):
        T = T @ tf(xyz, rpy) @ rot_z(a)
    return T @ TCP_T


def to_rad(v, hyp):
    sg, of = HYP[hyp]
    return np.array([math.radians(sg[i] * SCALE[i] * float(v[i]) + of[i]) for i in range(5)])


def fitH(uv, xy):
    src = np.asarray(uv, float) / np.array([640.0, 480.0])
    H, _ = cv2.findHomography(src, np.asarray(xy, float), 0)
    return H


def mapH(H, uv):
    p = np.c_[np.asarray(uv, float) / np.array([640.0, 480.0]), np.ones(len(uv))] @ H.T
    return p[:, :2] / p[:, 2:3]


def invH(H, xy):
    p = np.c_[np.asarray(xy, float), np.ones(len(xy))] @ np.linalg.inv(H).T
    return (p[:, :2] / p[:, 2:3]) * np.array([640.0, 480.0])


def loo_errors(uv, xy):
    errs = []
    n = len(uv)
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        H = fitH(uv[idx], xy[idx])
        pred = mapH(H, uv[i:i + 1])[0]
        errs.append(1000.0 * float(np.linalg.norm(pred - xy[i])))
    return errs


def find_plate(img, hlo, hhi, smin=90, vmin=90):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    m = ((h >= hlo) & (h <= hhi) & (s >= smin) & (v >= vmin)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    pts = np.column_stack(np.where(lab == i))[:, ::-1].astype(np.float32)
    (cx, cy), r = cv2.minEnclosingCircle(pts)
    return (float(cx), float(cy), float(r))


YMAX, PLATE_EXCL = 330, 1.08


def blobs(img, plates):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    m = (((s > 110) & (v > 110)) | ((v > 175) & (s < 60))).astype(np.uint8) * 255
    m[YMAX:, :] = 0
    m[:, :30] = 0
    m[:, 610:] = 0
    for (cx, cy, r) in plates:
        cv2.circle(m, (int(round(cx)), int(round(cy))), int(round(r * PLATE_EXCL)), 0, -1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, ce = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        x, y, w, hh, a = [int(t) for t in st[i]]
        if 150 <= a <= 4000 and 0.3 <= w / max(hh, 1) <= 3.5:
            out.append({"u": round(float(ce[i][0]), 1), "v": round(float(ce[i][1]), 1), "x": x, "y": y,
                        "w": w, "h": hh, "area": a})
    return out


def in_arm(b):
    return ARM_BOX[0] <= b["u"] <= ARM_BOX[2] and ARM_BOX[1] <= b["v"] <= ARM_BOX[3]


# ---- load states from the parquet (independent of the extraction step)
states = {}
for f in sorted(os.listdir(os.path.join(ROOT, "data", "chunk-000"))):
    if not f.endswith(".parquet"):
        continue
    d = pq.read_table(os.path.join(ROOT, "data", "chunk-000", f),
                      columns=["observation.state", "frame_index", "episode_index"]).to_pydict()
    for i in range(len(d["episode_index"])):
        states.setdefault(int(d["episode_index"][i]), []).append((int(d["frame_index"][i]), d["observation.state"][i]))
for ep in list(states):
    states[ep].sort(key=lambda r: r[0])
    states[ep] = np.array([s for _, s in states[ep]], float)

usable = [k for k in sorted(eps, key=int) if eps[k]["status"] == "ok"]
PAIRS = []
for k in usable:
    e = eps[k]
    img_s = cv2.imread(os.path.join(HERE, e["frame_start"]))
    img_e = cv2.imread(os.path.join(HERE, e["frame_end"]))
    pl_s = [find_plate(img_s, 88, 104), find_plate(img_s, 60, 86)]
    pl_e = [find_plate(img_e, 88, 104), find_plate(img_e, 60, 86)]
    bs_all = blobs(img_s, pl_s)
    bs = [b for b in bs_all if not in_arm(b)]
    be = blobs(img_e, pl_e)
    cand = []
    for b in bs:
        d = min([math.hypot(b["u"] - c["u"], b["v"] - c["v"]) for c in be] or [1e9])
        if d > 18:
            cand.append(dict(b, dist_end_px=round(d, 1)))
    gq = np.array(e["grasp_state"][:5], float)
    S = states[int(k)]
    g = S[:, 5]
    rel = next((t for t in range(e["grasp_frame"] + 15, len(g)) if g[t] > 7.0), None)
    PAIRS.append({"ep": int(k), "tasks": e["tasks"], "plates_start": pl_s, "plates_end": pl_e,
                  "n_start_all": len(bs_all), "n_start": len(bs), "n_arm_excluded": len(bs_all) - len(bs),
                  "n_end": len(be), "blobs_start": bs, "end_blobs": be, "cand": cand,
                  "grasp_q_norm": gq.tolist(), "release_q_norm": (S[rel, :5].tolist() if rel is not None else None),
                  "release_frame": rel})

for p in PAIRS:
    p["status"] = {1: "unique", 0: "none"}.get(len(p["cand"]), "ambiguous")
    p["chosen"] = p["cand"][0] if len(p["cand"]) == 1 else None

diag = [{"ep": p["ep"], "status": p["status"], "n_start_all": p["n_start_all"], "n_arm_excluded": p["n_arm_excluded"],
         "n_start": p["n_start"], "n_end": p["n_end"],
         "cand": [{k: c[k] for k in ("u", "v", "area", "w", "h", "dist_end_px")} for c in p["cand"]]} for p in PAIRS]
json.dump(diag, open(os.path.join(HERE, "label_diagnostics.json"), "w"), indent=1)
uniq = [p for p in PAIRS if p["status"] == "unique"]
print("labels:", {s: sum(1 for p in PAIRS if p["status"] == s) for s in ("unique", "ambiguous", "none")}, flush=True)
if len(uniq) < 4:
    for d in diag:
        print("ep", d["ep"], d["status"], "start", d["n_start"], "end", d["n_end"], [(c["u"], c["v"], c["area"]) for c in d["cand"]], flush=True)
    sys.exit(0)

# ---- hypothesis comparison on unique pairs
hyp_report = {}
for hyp in HYP:
    uv = np.array([[p["chosen"]["u"], p["chosen"]["v"]] for p in uniq], float)
    xyz = np.array([fk_T(to_rad(p["grasp_q_norm"], hyp))[:3, 3] for p in uniq])
    xy = xyz[:, :2]
    H = fitH(uv, xy)
    res_in = 1000.0 * np.linalg.norm(mapH(H, uv) - xy, axis=1)
    loo = np.array(loo_errors(uv, xy))
    rel_hits, rel_detail = 0, []
    for p in uniq:
        if p["release_q_norm"] is None:
            continue
        ptxy = fk_T(to_rad(p["release_q_norm"], hyp))[:2, 3]
        px = invH(H, ptxy[None, :])[0]
        pl = p["plates_start"]
        dists = [math.hypot(px[0] - c[0], px[1] - c[1]) for c in pl]
        near = ["blue", "green"][int(np.argmin(dists))]
        task = " ".join(p["tasks"])
        want = "blue" if ("blue" in task or "left plate" in task) else ("green" if "green" in task else None)
        ok = (want == near) if want else None
        rel_detail.append({"ep": p["ep"], "task": task, "pix_pred": [round(float(px[0]), 1), round(float(px[1]), 1)],
                           "nearest_plate": near, "dist_blue_px": round(dists[0], 1), "dist_green_px": round(dists[1], 1),
                           "match_task": ok})
        if ok:
            rel_hits += 1
    hyp_report[hyp] = {"n_pairs": len(uniq), "rms_in_sample_mm": round(float(np.sqrt(np.mean(res_in ** 2))), 2),
                       "loo_rms_mm": round(float(np.sqrt(np.mean(loo ** 2))), 2),
                       "loo_median_mm": round(float(np.median(loo)), 2), "loo_max_mm": round(float(np.max(loo)), 2),
                       "loo_per_pair_mm": {str(p["ep"]): round(float(v), 2) for p, v in zip(uniq, loo)},
                       "release_task_matches": rel_hits, "release_checks": rel_detail,
                       "home_tcp_xyz_m": [round(float(v), 4) for v in fk_T(to_rad(eps["0"]["start_state"][:5], hyp))[:3, 3]]}
best = min(HYP, key=lambda h: hyp_report[h]["loo_rms_mm"])

# ---- resolve ambiguous pairs with the best hypothesis prior (flagged), then refit on all chosen pairs
uv_u = np.array([[p["chosen"]["u"], p["chosen"]["v"]] for p in uniq], float)
xy_u = np.array([fk_T(to_rad(p["grasp_q_norm"], best))[:2, 3] for p in uniq])
H_prior = fitH(uv_u, xy_u)
for p in PAIRS:
    if p["status"] != "ambiguous":
        continue
    xy_g = fk_T(to_rad(p["grasp_q_norm"], best))[:2, 3]
    px = invH(H_prior, xy_g[None, :])[0]
    dists = [math.hypot(c["u"] - px[0], c["v"] - px[1]) for c in p["cand"]]
    j = int(np.argmin(dists))
    p["chosen"] = dict(p["cand"][j], resolved_by="nearest to prior homography", prior_dist_px=round(dists[j], 1))
    p["status"] = "ambiguous_resolved_visual_check"
chosen = [p for p in PAIRS if p["chosen"] is not None]
uv_all = np.array([[p["chosen"]["u"], p["chosen"]["v"]] for p in chosen], float)
xyz_all = np.array([fk_T(to_rad(p["grasp_q_norm"], best))[:3, 3] for p in chosen])
H_all = fitH(uv_all, xyz_all[:, :2])
loo_all = np.array(loo_errors(uv_all, xyz_all[:, :2]))
inl = loo_all <= 15.0
H_in = fitH(uv_all[inl], xyz_all[inl, :2]) if inl.sum() >= 8 else H_all
loo_in = np.array(loo_errors(uv_all[inl], xyz_all[inl, :2])) if inl.sum() >= 8 else loo_all
z_grasp = float(np.median(xyz_all[:, 2]))

pairs_out = []
for p, lo in zip(chosen, loo_all):
    pairs_out.append({"ep": p["ep"], "status": p["status"], "pixel_uv": [p["chosen"]["u"], p["chosen"]["v"]],
                      "blob_wh": [p["chosen"]["w"], p["chosen"]["h"]],
                      "tcp_xyz_m": [round(float(v), 4) for v in fk_T(to_rad(p["grasp_q_norm"], best))[:3, 3]],
                      "loo_err_mm": round(float(lo), 2), "kept_in_fit": bool(lo <= 15.0),
                      "prior_dist_px": p["chosen"].get("prior_dist_px")})
dropped = [{"ep": p["ep"], "reason": "no start blob vanished by end frame"} for p in PAIRS if p["chosen"] is None]
dropped += [{"ep": int(k), "reason": eps[k]["status"]} for k in sorted(eps, key=int) if eps[k]["status"] != "ok"]
for o in pairs_out:
    if not o["kept_in_fit"]:
        dropped.append({"ep": o["ep"], "reason": "LOO error %.1f mm > 15 mm" % o["loo_err_mm"]})

calib = {"unit_note": "pixel normalised by 640x480; XY in robot base frame (m) from URDF FK; joints via SCALE from project_robots.json",
         "scale_deg_per_runtime_unit": [round(s, 6) for s in SCALE], "hypotheses": hyp_report,
         "best_hypothesis_by_loo_rms": best, "z_grasp_median_m": round(z_grasp, 4),
         "homography_all_pairs_3x3": H_all.round(8).tolist(), "homography_inliers_3x3": H_in.round(8).tolist(),
         "loo_all_rms_mm": round(float(np.sqrt(np.mean(loo_all ** 2))), 2),
         "loo_all_median_mm": round(float(np.median(loo_all)), 2), "loo_all_max_mm": round(float(np.max(loo_all)), 2),
         "loo_inliers_rms_mm": round(float(np.sqrt(np.mean(loo_in ** 2))), 2),
         "pairs": pairs_out, "dropped": dropped,
         "plate_pixels_per_episode": {str(p["ep"]): [[round(c, 1) for c in q] for q in p["plates_start"]] for p in PAIRS}}
json.dump(calib, open(os.path.join(HERE, "calib.json"), "w"), indent=1)


# ---- offline pick check: held-out homography -> top-down IK at the predicted XY -> compare with the human grasp
def residual(q, target, w=0.05):
    T = fk_T(q)
    return np.r_[T[:3, 3] - target, w * (T[:3, 2] - np.array([0.0, 0.0, -1.0]))]


def ik_topdown(target, seeds):
    best_c, best_q = None, None
    for q0 in seeds:
        q = np.clip(np.array(q0, float), LO, HI)
        lam = 1e-3
        for _ in range(300):
            r = residual(q, target)
            J = np.zeros((6, 5))
            for j in range(5):
                dq = np.zeros(5)
                dq[j] = 1e-6
                J[:, j] = (residual(q + dq, target) - r) / 1e-6
            step = -np.linalg.solve(J.T @ J + lam * np.eye(5), J.T @ r)
            qn = np.clip(q + step, LO, HI)
            if np.linalg.norm(residual(qn, target)) < np.linalg.norm(r):
                q = qn
                lam = max(lam / 3.0, 1e-9)
            else:
                lam *= 4.0
            if np.linalg.norm(step) < 1e-10 or lam > 1e7:
                break
        c = float(np.linalg.norm(residual(q, target)))
        if best_c is None or c < best_c:
            best_c, best_q = c, q.copy()
    return best_c, best_q


offline = []
for i, p in enumerate(chosen):
    others = [j for j in range(len(chosen)) if j != i and chosen[j]["status"] != "ambiguous_resolved_visual_check"]
    H_i = fitH(uv_all[others], xyz_all[others, :2])
    xy_pred = mapH(H_i, uv_all[i:i + 1])[0]
    tgt = np.array([xy_pred[0], xy_pred[1], z_grasp])
    q_h = to_rad(p["grasp_q_norm"], best)
    T_h = fk_T(q_h)
    seeds = []
    pan0 = math.atan2(xy_pred[1], xy_pred[0])
    for pan in (pan0, pan0 + 0.3, pan0 - 0.3):
        for lift, elbow in ((0.6, 0.6), (-0.3, 0.9), (1.0, 0.2), (0.2, -0.4)):
            seeds.append([pan, lift, elbow, 0.5, 0.0])
    cost, q_ik = ik_topdown(tgt, seeds)
    T_ik = fk_T(q_ik)
    cart_mm = 1000.0 * float(np.linalg.norm(T_ik[:3, 3] - T_h[:3, 3]))
    plan_xy_mm = 1000.0 * float(np.linalg.norm(xy_pred - T_h[:2, 3]))
    ang = math.degrees(math.acos(max(-1.0, min(1.0, -float(T_h[2, 2])))))
    offline.append({"ep": p["ep"], "pred_xy_m": [round(float(v), 4) for v in xy_pred],
                    "human_xy_m": [round(float(v), 4) for v in T_h[:2, 3]], "plan_xy_err_mm": round(plan_xy_mm, 2),
                    "ik_residual_norm": round(cost, 6), "ik_tcp_xyz_m": [round(float(v), 4) for v in T_ik[:3, 3]],
                    "cartesian_err_mm": round(cart_mm, 2),
                    "joint_err_deg": [round(math.degrees(a - b), 2) for a, b in zip(q_ik, q_h)],
                    "human_tool_tilt_from_vertical_deg": round(ang, 1),
                    "pass_6mm": bool(cart_mm <= 6.0 and cost < 0.005)})
cart = np.array([o["cartesian_err_mm"] for o in offline])
json.dump({"method": "LOO homography (best hypothesis) -> top-down IK (5 joints, DLS, URDF limits) -> FK compare",
           "criterion": "cartesian_err_mm <= 6 and IK residual < 0.005", "best_hypothesis": best,
           "n": len(offline), "pass_count": int(sum(o["pass_6mm"] for o in offline)),
           "cartesian_err_mm_median": round(float(np.median(cart)), 2), "cartesian_err_mm_max": round(float(np.max(cart)), 2),
           "plan_xy_err_mm_median": round(float(np.median([o["plan_xy_err_mm"] for o in offline])), 2),
           "per_grasp": offline}, open(os.path.join(HERE, "offline_eval.json"), "w"), indent=1)

# ---- overlays for visual verification
ov_dir = os.path.join(HERE, "overlays")
os.makedirs(ov_dir, exist_ok=True)
ov_paths = []
for p, o in zip(chosen, pairs_out):
    img = cv2.imread(os.path.join(HERE, eps[str(p["ep"])]["frame_start"]))
    for b in p["blobs_start"]:
        cv2.rectangle(img, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), (0, 255, 255), 1)
    for c in p["cand"]:
        cv2.rectangle(img, (c["x"] - 2, c["y"] - 2), (c["x"] + c["w"] + 2, c["y"] + c["h"] + 2), (0, 255, 0), 2)
    cu, cv_ = p["chosen"]["u"], p["chosen"]["v"]
    cv2.circle(img, (int(cu), int(cv_)), 5, (255, 0, 255), -1)
    for (cx, cy, r), col in zip(p["plates_start"], [(255, 255, 0), (0, 128, 255)]):
        cv2.circle(img, (int(cx), int(cy)), int(r), col, 2)
    gp = invH(H_all, np.array([[o["tcp_xyz_m"][0], o["tcp_xyz_m"][1]]]))[0]
    cv2.drawMarker(img, (int(gp[0]), int(gp[1])), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    cv2.putText(img, "ep%d %s" % (p["ep"], p["status"][:12]), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    out = os.path.join(ov_dir, "ov_ep%02d.png" % p["ep"])
    cv2.imwrite(out, img)
    ov_paths.append(out)
cells = [cv2.resize(cv2.imread(q), (320, 240)) for q in ov_paths]
while len(cells) % 4:
    cells.append(np.zeros((240, 320, 3), np.uint8))
rows = [np.hstack(cells[i:i + 4]) for i in range(0, len(cells), 4)]
cv2.imwrite(os.path.join(HERE, "overlay_sheet.png"), np.vstack(rows))
print("best", best, "pairs", len(chosen), "loo_all_rms", calib["loo_all_rms_mm"], "loo_in_rms", calib["loo_inliers_rms_mm"])
for h in HYP:
    print(h, hyp_report[h]["loo_rms_mm"], hyp_report[h]["loo_median_mm"], hyp_report[h]["release_task_matches"], hyp_report[h]["home_tcp_xyz_m"])
print("offline pass", sum(o["pass_6mm"] for o in offline), "of", len(offline), "median cart mm", float(np.median(cart)))
