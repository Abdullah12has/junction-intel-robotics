#!/usr/bin/env python3
"""P3-TWIN-DYN E1: builds g2_report.json, metrics.json and the PNG plots from the fold, full-fit and reference results.

Inputs (all written by fit_dynamics.py / baseline_kinematic.py): loeo_result.json, loeo/fold_*.json, full_result.json,
baseline_kinematic.json, and the logs. Outputs: g2_report.json, metrics.json, g2_per_joint_rmse.png,
g2_heldout_overlay.png, g2_gripper_readings.png (all in work/P3-TWIN-DYN/E1).
"""
import datetime
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import twin_lib as T  # noqa: E402
import plotlib as P  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

WORK = f"{T.R}/work/P3-TWIN-DYN/E1"
GATE_RMSE = 0.5
GATE_LAG = 1.0
FRAMES_PER_S = 30.0


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def bar_plot(path, labels, ident, ref, gate):
    img = Image.new("RGB", (900, 440), (255, 255, 255))
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = 70, 50, 860, 360
    top = max(max(ident), max(ref), gate) * 1.2
    d.rectangle([x0, y0, x1, y1], outline=(20, 20, 20))
    d.text((x0 + 4, 14), "Leave-one-episode-out RMSE per joint (normalized units), 17 folds", fill=(0, 0, 0),
           font=P.FONT)
    _, _, ticks = P.nice_ticks(0.0, top, 5)
    for v in ticks:
        if v < 0 or v > top:
            continue
        yy = y1 - v / top * (y1 - y0)
        d.line([(x0, yy), (x1, yy)], fill=P.COLORS["lgray"])
        d.text((x0 - 38, yy - 5), f"{v:g}", fill=(0, 0, 0), font=P.FONT)
    gy = y1 - gate / top * (y1 - y0)
    d.line([(x0, gy), (x1, gy)], fill=P.COLORS["r"], width=2)
    d.text((x1 - 170, gy - 14), "G2 gate 0.5 units", fill=P.COLORS["r"], font=P.FONT)
    slot = (x1 - x0) / len(labels)
    for k, lab in enumerate(labels):
        for s, (vals, color) in enumerate(((ident, P.COLORS["b"]), (ref, P.COLORS["gray"]))):
            bx0 = x0 + k * slot + slot * (0.12 + 0.4 * s)
            bx1 = bx0 + slot * 0.36
            by = y1 - vals[k] / top * (y1 - y0)
            d.rectangle([bx0, by, bx1, y1], fill=color)
            d.text((bx0, by - 12), f"{vals[k]:.2f}", fill=(0, 0, 0), font=P.FONT)
        d.text((x0 + k * slot + slot * 0.25, y1 + 6), lab, fill=(0, 0, 0), font=P.FONT)
    d.rectangle([x0 + 10, y1 + 30, x0 + 24, y1 + 44], fill=P.COLORS["b"])
    d.text((x0 + 30, y1 + 31), "identified dynamics (this stage)", fill=(0, 0, 0), font=P.FONT)
    d.rectangle([x0 + 300, y1 + 30, x0 + 314, y1 + 44], fill=P.COLORS["gray"])
    d.text((x0 + 320, y1 + 31), "reference: no dynamics (goal delayed by a fitted lag)", fill=(0, 0, 0), font=P.FONT)
    d.text((x0 + 10, y1 + 52), "* gripper: free-motion rows only (rows with pos5 < 3 and goal5 < 3 masked); "
           "per-joint lag and per-episode values in g2_report.json", fill=(0, 0, 0), font=P.FONT)
    img.save(path)


def simulate_one(ep, p, ro):
    st = T.simulate([ep], p, ro)[0]
    k = T.calib_k()
    sim_norm = np.zeros((len(ep["t"]), 6))
    for j in range(6):
        q = np.interp(ep["t"], st[:, 0], st[:, 1 + j])
        sim_norm[:, j] = q / k[j]
    return sim_norm


def overlay(path, ep, sim_norm, title):
    img, draw = P.new_canvas(1000, 1200)
    draw.text((20, 8), title, fill=(0, 0, 0), font=P.FONT)
    t = ep["t"]
    for j in range(6):
        box = (90, 40 + j * 190, 900, 40 + j * 190 + 150)
        vals = np.concatenate([ep["goal"][:, j], ep["pos"][:, j], sim_norm[:, j]])
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        pad = 0.05 * (hi - lo + 1e-6)
        pn = P.Panel(draw, box, (0, t[-1]), (lo - pad, hi + pad), title=T.JOINTS[j], xlabel="time s")
        pn.line(t, ep["goal"][:, j], "o", 1)
        pn.line(t, ep["pos"][:, j], "b", 1)
        pn.line(t, sim_norm[:, j], "r", 1)
    img.save(path)


def gripper_plot(path, ep, sim_norm, title):
    img, draw = P.new_canvas(1000, 560)
    draw.text((20, 8), title, fill=(0, 0, 0), font=P.FONT)
    t = ep["t"]
    lo, hi = -2.0, 30.0
    pn = P.Panel(draw, (90, 40, 940, 250), (0, t[-1]), (lo, hi), title="gripper, full episode", xlabel="time s")
    pn.line(t, ep["goal"][:, 5], "o", 1)
    pn.line(t, ep["pos"][:, 5], "b", 1)
    pn.line(t, sim_norm[:, 5], "r", 1)
    pn.legend([("logged goal", "o", True), ("logged present", "b", False), ("sim present", "r", False)], 700, 50)
    contact = (ep["pos"][:, 5] < T.GRIP_CONTACT_POS) & (ep["goal"][:, 5] < T.GRIP_CONTACT_POS)
    idx = np.flatnonzero(contact)
    if idx.size:
        longest = max(np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1), key=len)
        t0, t1 = max(t[longest[0]] - 2.0, 0.0), min(t[longest[-1]] + 2.0, t[-1])
    else:
        t0, t1 = 0.0, t[-1]
    pz = P.Panel(draw, (90, 300, 940, 510), (t0, t1), (-0.5, 4.0), title="gripper zoom on holding plateau",
                 xlabel="time s")
    w = (t >= t0 - 0.5) & (t <= t1 + 0.5)  # Panel.line does not clip, so pass only the window samples
    pz.line(t[w], ep["goal"][w, 5], "o", 1)
    pz.line(t[w], ep["pos"][w, 5], "b", 1)
    pz.line(t[w], sim_norm[w, 5], "r", 1)
    img.save(path)


def main():
    eps = T.load_episodes()
    ro = T.mjro.Rollout(nthread=T.NTHREAD)
    loeo = json.load(open(f"{WORK}/loeo_result.json"))
    base = json.load(open(f"{WORK}/baseline_kinematic.json"))
    full = json.load(open(f"{WORK}/full_result.json"))
    fold_files = sorted(glob.glob(f"{WORK}/loeo/fold_*.json"))
    folds = {json.load(open(f))["fold"]: json.load(open(f)) for f in fold_files}

    p_full = np.array(full["params_6x6"])
    free_names = ["kp", "kv", "frictionloss", "armature", "delay"]
    bound_hits = []
    z = np.log(p_full[:, T.FREE])
    lo, hi = np.log(T.LO_MAT), np.log(T.HI_MAT)
    for j in range(6):
        for c, nm in enumerate(free_names):
            if z[j, c] - lo[j, c] < 1e-3:
                bound_hits.append(f"{T.JOINTS[j]}.{nm} at lower bound")
            elif hi[j, c] - z[j, c] < 1e-3:
                bound_hits.append(f"{T.JOINTS[j]}.{nm} at upper bound")

    loeo_rmse = np.array(loeo["loeo_rmse_norm"])
    loeo_lag = np.array(loeo["loeo_lag_frames"])
    ins = np.array(full["insample_rmse_norm"])
    ref_rmse = np.array(base["loeo_rmse_norm"])
    ref_lag = np.array(base["loeo_lag_frames"])
    per_joint = {}
    for j, name in enumerate(T.JOINTS):
        per_joint[name] = {
            "loeo_rmse_norm": round(float(loeo_rmse[j]), 4),
            "loeo_rmse_pass_0p5": bool(loeo_rmse[j] <= GATE_RMSE),
            "loeo_lag_frames": round(float(loeo_lag[j]), 3),
            "loeo_lag_pass_1frame": bool(loeo_lag[j] <= GATE_LAG),
            "insample_rmse_norm_full_fit": round(float(ins[j]), 4),
            "reference_no_dynamics_loeo_rmse_norm": round(float(ref_rmse[j]), 4),
            "reference_no_dynamics_loeo_lag_frames": round(float(ref_lag[j]), 3),
            "loeo_rows_scored": int(loeo["loeo_rows_scored"][j]),
        }
    failing_rmse = [T.JOINTS[j] for j in range(6) if loeo_rmse[j] > GATE_RMSE]
    failing_lag = [T.JOINTS[j] for j in range(6) if loeo_lag[j] > GATE_LAG]
    verdict = ("G2 NOT MET for the arm (RMSE above 0.5 on: " + ", ".join(failing_rmse) + "); "
               "lag gate fails on: " + (", ".join(failing_lag) if failing_lag else "none") +
               "; gripper: free-motion rows only, brick contact not modelled (PARTIAL)")
    total_rows = int(sum(len(e["t"]) for e in eps))
    masked_rows = int(sum(np.sum((e["pos"][:, 5] < T.GRIP_CONTACT_POS) & (e["goal"][:, 5] < T.GRIP_CONTACT_POS))
                          for e in eps))
    closed_by_ep = {}
    for e in eps:
        n = int(np.sum((e["pos"][:, 5] >= 0.5) & (e["pos"][:, 5] <= 0.7) & (e["goal"][:, 5] <= 1.0)))
        if n:
            closed_by_ep[e["name"]] = n
    gr = {"contact_rows_masked": masked_rows, "total_rows": total_rows,
          "contact_rows_fraction": round(masked_rows / total_rows, 4),
          "closed_readings_0p5_to_0p7_with_goal_le_1_by_episode": closed_by_ep,
          "holding_readings_1p4_to_2p5_with_goal_le_1_rows": int(sum(
              np.sum((e["pos"][:, 5] >= 1.4) & (e["pos"][:, 5] <= 2.5) & (e["goal"][:, 5] <= 1.0)) for e in eps))}
    report = {
        "stage": "twin-dyn-v1", "utc": now(), "gate": "G2", "verdict": verdict,
        "gate_definition": "per-joint leave-one-episode-out RMSE (normalized units, pooled over held-out rows) <= 0.5 "
                           "and held-out lag error <= 1 frame (33.3 ms); lag = best shift of the simulated present "
                           "against the logged present, pooled over held-out rows",
        "per_joint": per_joint,
        "gripper_readings": gr,
        "per_episode_loeo": loeo["per_episode"],
        "episodes_scored": len(loeo["per_episode"]),
        "folds_present": len(fold_files),
        "full_fit": {"nfev": full["nfev"], "njev": full["njev"], "status": full["status"],
                     "status_meaning": "0 = max_nfev reached (not converged)",
                     "cost": full["cost"], "bound_hits": bound_hits,
                     "parameters": {T.JOINTS[j]: {nm: float(p_full[j, c]) for c, nm in enumerate(
                         ["kp", "kv", "frictionloss", "armature", "forcerange", "delay"])}
                         for j in range(6)}},
        "reference_no_dynamics": {"description": "present = logged goal delayed by a per-joint lag fitted on the 16 "
                                                 "training episodes; no sag, friction or speed model",
                                  "loeo_rmse_norm": ref_rmse.tolist(), "loeo_lag_frames": ref_lag.tolist(),
                                  "g2_pass": base["g2_pass"]},
        "data": {"files_used": 17, "rows": int(sum(len(e["t"]) for e in eps)),
                 "missing_joints_csv": ["any_1731/ep_0003", "any_1731/ep_0004", "any_1740/ep_0005",
                                        "any_1740/ep_0008", "any_1740/ep_0018"],
                 "idle_episodes_no_motion": ["any_1740/ep_0002", "any_1740/ep_0003"],
                 "sample_period_median_s": 0.035},
        "method": {"simulator": "MuJoCo 4.0.1 (laptop sim-env), URDF so101.urdf without geometry (inertials kept), "
                                "no joint limits, <position> actuator per joint, implicitfast, DT 0.004 s",
                   "command_input": "logged goal (the setpoint the motion server actually sent: plan_ramp smoothstep "
                                    "at 30 Hz plus compensation offset), zero-order hold, delayed by fitted delay",
                   "rad_per_unit": "width/200 (body) or /100 (gripper) * 2pi/4095 from project_robots.json",
                   "forcerange": "fixed 10 N.m (URDF effort limit), not identified",
                   "fit": "scipy least_squares trf, log-parameters, 2-point jacobian, max_nfev 20 per fit",
                   "start": "fixed INIT per fold (no held-out information in the start point)",
                   "gripper": "rows with pos5 < 3 and goal5 < 3 (brick holding plateau and its closing edge) masked",
                   "lag_grid": "shifts in 5 ms steps (0.15 frame at 30 Hz), pooled over held-out rows"},
        "not_done": ["gripper brick contact (31.7 mm brick compliance) not modelled: the logs show holding at "
                     "1.5 to 2.4 units with goal near 0, the sim closes to goal",
                     "closed readings 0.5 to 0.7 with goal <= 1 are in the logs (any_1740/ep_0006, any_1740/ep_0013); "
                     "these rows are masked from the gripper RMSE, so the closed-stop behaviour is not fitted",
                     "fits not converged (max_nfev 20); more iterations would be needed for a converged claim",
                     "R/sim/twin/robot/so101_twin.xml exists (P3-TWIN-ROBOT) but was not used: these fits use the "
                     "URDF-derived no-geometry model, so the two models were not compared",
                     "no comparison with R/twin (deleted by Hassan 19:48Z, D-102); nothing copied from it"],
        "files": {"params_json": "R/sim/twin/dynamics/twin_dynamics_v1.json", "model_xml":
                  "R/sim/twin/dynamics/so101_dyn_v1.xml", "fit_scripts": ["R/sim/twin/dynamics/twin_lib.py",
                  "R/sim/twin/dynamics/fit_dynamics.py"], "tools": "work/P3-TWIN-DYN/E1/tools/",
                  "plots": ["g2_per_joint_rmse.png", "g2_heldout_overlay.png", "g2_gripper_readings.png"]},
    }
    with open(f"{WORK}/g2_report.json", "w") as fh:
        json.dump(report, fh, indent=1)
    metrics = {"stage": "twin-dyn-v1", "utc": now(),
               "metrics": {"loeo_rmse_norm": loeo_rmse.round(4).tolist(), "loeo_lag_frames": loeo_lag.round(3).tolist(),
                           "insample_rmse_norm_full_fit": ins.round(4).tolist(),
                           "reference_no_dynamics_loeo_rmse_norm": ref_rmse.round(4).tolist(),
                           "reference_no_dynamics_loeo_lag_frames": ref_lag.round(3).tolist(),
                           "g2_rmse_pass": bool(loeo["g2_rmse_pass"]), "g2_lag_pass": bool(loeo["g2_lag_pass"]),
                           "g2_pass": bool(loeo["g2_pass"]), "episodes_scored": len(loeo["per_episode"]),
                           "rows_scored_per_joint": loeo["loeo_rows_scored"]},
               "previous": None, "delta": None}
    with open(f"{WORK}/metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=1)

    bar_plot(f"{WORK}/g2_per_joint_rmse.png", ["pan", "lift", "elbow", "wrist_flex", "roll", "gripper*"],
             loeo_rmse.tolist(), ref_rmse.tolist(), GATE_RMSE)
    fold_ep = folds[2]
    ep2 = [e for e in eps if e["name"] == fold_ep["episode"]][0]
    sim2 = simulate_one(ep2, np.array(fold_ep["p_6x6"]), ro)
    overlay(f"{WORK}/g2_heldout_overlay.png", ep2, sim2,
            f"held-out fold 2 ({ep2['name']}): logged goal (orange), logged present (blue), sim present (red), "
            "normalized units")
    ep13 = [e for e in eps if e["name"] == "any_1740/ep_0013"][0]
    sim13 = simulate_one(ep13, p_full, ro)
    gripper_plot(f"{WORK}/g2_gripper_readings.png", ep13, sim13,
                 "any_1740/ep_0013 gripper, full-fit parameters (in-sample): logged goal, logged present, sim present")
    print("verdict:", verdict)
    print("bound hits:", bound_hits)
    print("wrote g2_report.json, metrics.json, g2_per_joint_rmse.png, g2_heldout_overlay.png, g2_gripper_readings.png")


if __name__ == "__main__":
    main()
