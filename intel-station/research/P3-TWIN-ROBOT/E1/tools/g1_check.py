#!/usr/bin/env python3
"""E1 G1 gate: TCP pose from MuJoCo against kin.fk over the recorded joint states (17 episodes, all rows), plus the
200-sample fixed set. Candidate: twin_ours (R/sim/twin/robot/so101_twin.xml). Baseline: official_urdf (converter output
from the same URDF without meshes, the previous stage). Gate: position error < 0.5 mm. Writes g1_report.json and
metrics.json in E1. R/twin was deleted at 19:48Z (D-102) and is not used.
Run: nice -n 19 $SIM_PY -I -B g1_check.py"""
import datetime
import json
import os
import sys
import time

import numpy as np

R = "/home/intel/intel-junction-2026"
E1 = f"{R}/work/P3-TWIN-ROBOT/E1"
sys.path.insert(0, f"{R}/sim/twin/robot")
import tcp  # noqa: E402

THRESH_MM = 0.5
OURS = f"{R}/sim/twin/robot/so101_twin.xml"
OFFICIAL = f"{E1}/official_urdf_twin.xml"
KIN_NPZ = f"{E1}/kin_reference.npz"
FIXED = f"{R}/sim/twin/tests/data/g1_fixed_sample.json"
PATHS = {"twin_ours": OURS, "official_urdf": OFFICIAL}


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stats(x):
    x = np.asarray(x, dtype=float)
    return {"n": int(x.size), "max": float(x.max()), "median": float(np.median(x)),
            "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)), "mean": float(x.mean())}


def summarize(dp, da, ep, ep_names):
    per = []
    for e in np.unique(ep):
        m = ep == e
        label = ep_names[int(e)] if int(e) < len(ep_names) else str(int(e))
        per.append({"episode": label, "rows": int(m.sum()), "pos_mm_max": float(dp[m].max()),
                    "pos_mm_median": float(np.median(dp[m])), "ori_deg_max": float(da[m].max()),
                    "ori_deg_median": float(np.median(da[m]))})
    return {"pos_mm": stats(dp), "ori_deg": stats(da), "rows_over_threshold": int(np.sum(dp >= THRESH_MM)),
            "pass_G1": bool(dp.max() < THRESH_MM), "per_episode": per}


def flat(c):
    return {"pos_mm_max": c["pos_mm"]["max"], "pos_mm_median": c["pos_mm"]["median"],
            "pos_mm_p99": c["pos_mm"]["p99"], "ori_deg_max": c["ori_deg"]["max"],
            "ori_deg_median": c["ori_deg"]["median"], "rows_over_threshold": c["rows_over_threshold"],
            "fixed_pos_mm_max": c["fixed_sample"]["pos_mm_max"], "fixed_ori_deg_max": c["fixed_sample"]["ori_deg_max"]}


def main():
    t0 = time.time()
    ref = np.load(KIN_NPZ, allow_pickle=False)
    joints = np.asarray(ref["joints"], dtype=float)[:, :5]
    kin_p = np.asarray(ref["kin_tcp_mm"], dtype=float)
    kin_r = np.asarray(ref["kin_R"], dtype=float).reshape(-1, 3, 3)
    ep = np.asarray(ref["episode_index"]).astype(int)
    ep_names = [str(x) for x in ref["episode_names"]]
    n = len(joints)
    assert kin_p.shape == (n, 3) and kin_r.shape == (n, 3, 3) and ep.shape == (n,), "kin reference shape mismatch"

    with open(FIXED, encoding="utf-8") as fh:
        fixed = json.load(fh)
    if isinstance(fixed, dict):
        fixed = fixed.get("samples", [])
    fixed_j = np.array([s["joints_norm"] for s in fixed], dtype=float)[:, :5]
    fixed_p = np.array([s["kin_tcp_mm"] for s in fixed], dtype=float)
    fixed_r = np.array([s["kin_R"] for s in fixed], dtype=float).reshape(-1, 3, 3)

    cands, tcps = {}, {}
    for name, path in PATHS.items():
        arm = tcp.ArmModel(path)
        pos, rot = tcp.tcp_batch(arm, joints)
        tcps[name] = (pos, rot)
        dp, da = tcp.pose_errors(pos, rot, kin_p, kin_r)
        res = summarize(dp, da, ep, ep_names)
        fpos, frot = tcp.tcp_batch(arm, fixed_j)
        fdp, fda = tcp.pose_errors(fpos, frot, fixed_p, fixed_r)
        res["fixed_sample"] = {"n": int(len(fixed_j)), "pos_mm_max": float(fdp.max()),
                               "ori_deg_max": float(fda.max()), "pass_G1": bool(fdp.max() < THRESH_MM)}
        m = arm.model
        res["model"] = {"path": path, "sha256": tcp.sha256(path), "nbody": int(m.nbody), "njnt": int(m.njnt),
                        "nsite": int(m.nsite), "ngeom": int(m.ngeom)}
        cands[name] = res
        print(f"{name}: rows={n} pos_max_mm={res['pos_mm']['max']:.3e} ori_max_deg={res['ori_deg']['max']:.3e} "
              f"pass={res['pass_G1']} fixed_pos_max_mm={res['fixed_sample']['pos_mm_max']:.3e} "
              f"ngeom={res['model']['ngeom']}")

    cdp, cda = tcp.pose_errors(tcps["twin_ours"][0], tcps["twin_ours"][1],
                               tcps["official_urdf"][0], tcps["official_urdf"][1])
    cross = {"pos_mm_max": float(cdp.max()), "ori_deg_max": float(cda.max())}
    ours = cands["twin_ours"]
    decision = {
        "candidate": "twin_ours", "gate": "G1", "threshold_mm": THRESH_MM, "pass_G1": ours["pass_G1"],
        "pos_mm_max": ours["pos_mm"]["max"], "ori_deg_max": ours["ori_deg"]["max"],
        "fixed_sample_pos_mm_max": ours["fixed_sample"]["pos_mm_max"],
        "baseline": "official_urdf", "twin_ours_vs_official_urdf_tcp": cross,
        "note": "R/twin was deleted at 19:48Z (D-102); no comparison with it is made.",
    }
    report = {
        "gate": "G1", "stage": "twin-robot-v1", "utc": utc_now(), "threshold_mm": THRESH_MM,
        "kin_reference": {"path": KIN_NPZ, "sha256": tcp.sha256(KIN_NPZ), "rows": n, "episodes": len(ep_names),
                          "joints_used": "pos0..pos4 normalized; gripper excluded"},
        "fixed_sample": {"path": FIXED, "sha256": tcp.sha256(FIXED), "n": int(len(fixed_j))},
        "candidates": cands,
        "decision": decision,
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(E1, "g1_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)

    m_ours = flat(cands["twin_ours"])
    m_prev = flat(cands["official_urdf"])
    doc = {"stage": "twin-robot-v1", "utc": report["utc"], "gate": "G1", "candidate": "twin_ours", "rows": n,
           "pass_G1": cands["twin_ours"]["pass_G1"], "metrics": m_ours,
           "previous": {"name": "official_urdf (converter output, no meshes)", **m_prev},
           "delta": {k: m_ours[k] - m_prev[k] for k in m_ours}}
    with open(os.path.join(E1, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print("twin_ours vs official_urdf TCP:", cross)


if __name__ == "__main__":
    main()
