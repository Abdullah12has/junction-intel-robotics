"""Verifier V2: pixel-to-table fit from the 12 grasp instants (convention A) with a plate-derived prior.
Prior (2 plate correspondences, isotropic scale): u = U0 + S*y, v = V0 + S*x  (base frame, metres).
Association: nearest start-frame blob to the prior pixel, then refit and re-associate (2 rounds). Labels are checked on
an overlay sheet. Usage: python3 -I fit2.py HERE"""
import json
import math
import os
import sys

import cv2
import numpy as np

HERE = sys.argv[1]
pc = json.load(open(os.path.join(HERE, "plates_check.json")))
ids = json.load(open(os.path.join(HERE, "blob_ids_start.json")))
eps = json.load(open(os.path.join(HERE, "episodes.json")))
rows = [r for r in pc["rows"] if "grasp_xyz_m" in r]
S, U0, V0 = 1043.7, 336.7, -43.5
NORM = np.array([640.0, 480.0])


def prior_px(x, y):
    return (U0 + S * y, V0 + S * x)


def fitH(uv, xy):
    H, _ = cv2.findHomography(np.asarray(uv, float) / NORM, np.asarray(xy, float), 0)
    return H


def mapH(H, uv):
    p = np.c_[np.asarray(uv, float) / NORM, np.ones(len(uv))] @ H.T
    return p[:, :2] / p[:, 2:3]


def invH(H, xy):
    p = np.c_[np.asarray(xy, float), np.ones(len(xy))] @ np.linalg.inv(H).T
    return (p[:, :2] / p[:, 2:3]) * NORM


def loo(uv, xy):
    out = []
    for i in range(len(uv)):
        idx = [j for j in range(len(uv)) if j != i]
        out.append(1000.0 * float(np.linalg.norm(mapH(fitH(uv[idx], xy[idx]), uv[i:i + 1])[0] - xy[i])))
    return out


def associate(pred_fn, max_px):
    sel = []
    for r in rows:
        ep = str(r["ep"])
        pu, pv = pred_fn(r)
        cands = sorted(ids[ep], key=lambda b: math.hypot(b["u"] - pu, b["v"] - pv))
        ds = [round(math.hypot(b["u"] - pu, b["v"] - pv), 1) for b in cands[:3]]
        if not cands or math.hypot(cands[0]["u"] - pu, cands[0]["v"] - pv) > max_px:
            sel.append({"ep": r["ep"], "chosen": None, "pred_px": [round(pu, 1), round(pv, 1)], "top3_dist_px": ds})
        else:
            sel.append({"ep": r["ep"], "chosen": [cands[0]["u"], cands[0]["v"]], "pred_px": [round(pu, 1), round(pv, 1)],
                        "top3_dist_px": ds, "blob": cands[0]})
    return sel


rounds = []
sel = associate(lambda r: prior_px(r["grasp_xyz_m"][0], r["grasp_xyz_m"][1]), 80.0)
for rnd in range(3):
    ok = [s for s in sel if s["chosen"] is not None]
    uv = np.array([s["chosen"] for s in ok], float)
    xy = np.array([[next(r for r in rows if r["ep"] == s["ep"])["grasp_xyz_m"][0],
                    next(r for r in rows if r["ep"] == s["ep"])["grasp_xyz_m"][1]] for s in ok])
    H = fitH(uv, xy)
    e = loo(uv, xy)
    rounds.append({"round": rnd, "n": len(ok), "loo_rms_mm": round(float(np.sqrt(np.mean(np.square(e)))), 2),
                   "loo_median_mm": round(float(np.median(e)), 2), "loo_max_mm": round(float(np.max(e)), 2),
                   "H": H.round(8).tolist()})
    Hc = H
    sel = associate(lambda r: tuple(mapH(Hc, np.zeros((0, 2)))) if False else tuple(invH(Hc, [r["grasp_xyz_m"][:2]])[0]), 40.0)

# final association used for the report
ok = [s for s in sel if s["chosen"] is not None]
uv = np.array([s["chosen"] for s in ok], float)
xy = np.array([[next(r for r in rows if r["ep"] == s["ep"])["grasp_xyz_m"][0],
                next(r for r in rows if r["ep"] == s["ep"])["grasp_xyz_m"][1]] for s in ok])
H = fitH(uv, xy)
e = loo(uv, xy)
res_in = 1000.0 * np.linalg.norm(mapH(H, uv) - xy, axis=1)
per = []
for s, ee, ri in zip(ok, e, res_in):
    r = next(q for q in rows if q["ep"] == s["ep"])
    per.append({"ep": s["ep"], "pixel_uv": [round(s["chosen"][0], 1), round(s["chosen"][1], 1)],
                "tcp_xy_m": [r["grasp_xyz_m"][0], r["grasp_xyz_m"][1]], "loo_err_mm": round(ee, 2),
                "in_sample_err_mm": round(float(ri), 2), "pred_px_before": s["pred_px"], "top3_dist_px": s["top3_dist_px"]})
unlabelled = [{"ep": s["ep"], "pred_px": s["pred_px"], "top3_dist_px": s["top3_dist_px"]} for s in sel if s["chosen"] is None]
json.dump({"method": "plate-prior association (nearest start blob), 3 rounds of refit and re-association",
           "prior": {"u": "336.7 + 1043.7*y", "v": "-43.5 + 1043.7*x", "source": "two plate correspondences (blue, green)"},
           "rounds": rounds, "final_H_normalised_pixels_to_xy_m": H.round(8).tolist(),
           "final_pairs": per, "unlabelled": unlabelled,
           "loo_rms_mm": round(float(np.sqrt(np.mean(np.square(e)))), 2),
           "loo_median_mm": round(float(np.median(e)), 2), "loo_max_mm": round(float(np.max(e)), 2)},
          open(os.path.join(HERE, "fit_plate_prior.json"), "w"), indent=1)

# overlay sheet for visual check of the chosen labels
cells = []
for s in sel:
    img = cv2.imread(os.path.join(HERE, eps[str(s["ep"])]["frame_start"]))
    for b in ids[str(s["ep"])]:
        cv2.rectangle(img, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), (0, 255, 255), 1)
    cv2.drawMarker(img, (int(s["pred_px"][0]), int(s["pred_px"][1])), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
    if s["chosen"] is not None:
        cv2.circle(img, (int(s["chosen"][0]), int(s["chosen"][1])), 7, (255, 0, 255), -1)
    cv2.putText(img, "ep%d d=%s" % (s["ep"], s["top3_dist_px"][0] if s["top3_dist_px"] else "-"), (6, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cells.append(cv2.resize(img, (320, 240)))
while len(cells) % 4:
    cells.append(np.zeros((240, 320, 3), np.uint8))
sheet = np.vstack([np.hstack(cells[i:i + 4]) for i in range(0, len(cells), 4)])
cv2.imwrite(os.path.join(HERE, "fit_plate_prior_overlay.png"), sheet)
print("rounds", [(r["round"], r["n"], r["loo_rms_mm"], r["loo_median_mm"], r["loo_max_mm"]) for r in rounds])
for p in per:
    print("ep", p["ep"], "px", p["pixel_uv"], "xy", p["tcp_xy_m"], "loo", p["loo_err_mm"], "pred", p["pred_px_before"], "top3", p["top3_dist_px"])
print("unlabelled", unlabelled)
print("FINAL loo rms", round(float(np.sqrt(np.mean(np.square(e)))), 2), "median", round(float(np.median(e)), 2), "max", round(float(np.max(e)), 2))
