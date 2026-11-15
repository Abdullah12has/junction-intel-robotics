#!/usr/bin/env python3
"""V1 independent check of P3-TWIN-ROBOT (verifier code; not the builder's scripts).
Routes: A own URDF parser + numpy FK on the edge-developer-kit so101.urdf (the file kin.py loads);
B kin.fk called live from app/tools/kin.py (read only); C MuJoCo 4.0.1 on sim/twin/robot/so101_twin.xml with
radians from my own K (project_robots.json); D MuJoCo on the builder's E1/official_urdf_twin.xml (read only);
E own FK on the plugin so101_new_calib.urdf (read only). Gripper figures and the jaw-tip radius come from the raw
joints.csv files and the copied moving-jaw STL. Writes only V1/outputs/indep_results.json.
Run: nice -n 19 env OMP_NUM_THREADS=4 MUJOCO_GL=egl SIM_PY -I -B indep_check.py"""
import csv
import glob
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np

R = "/home/intel/intel-junction-2026"
V1 = f"{R}/work/P3-TWIN-ROBOT/V1"
E1 = f"{R}/work/P3-TWIN-ROBOT/E1"
URDF_KIN = (f"{R}/repos/edge-developer-kit-reference-scripts/samples/ai/edge-ai-demo-studio/workers/services/"
            "robotics-ai/platforms/so101/urdf/so101.urdf")
URDF_PLUG = f"{R}/repos/physicalai/packages/physicalai-mujoco-plugin/urdf/so101/so101_new_calib.urdf"
MJCF_TWIN = f"{R}/sim/twin/robot/so101_twin.xml"
MJCF_OFF = f"{E1}/official_urdf_twin.xml"
PROJECT = f"{R}/app/motion/project_robots.json"
FIXED = f"{R}/sim/twin/tests/data/g1_fixed_sample.json"
KREF = f"{R}/sim/twin/tests/data/kin_calibration_reference.json"
G1_REPORT = f"{E1}/g1_report.json"
JAW_STL = f"{R}/sim/twin/robot/meshes/moving_jaw_so101_v1.stl"
JAW_BODY_Z = 0.0189  # MJCF geom pos of moving_jaw_so101_v1 inside its body frame
BODY = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
TIP = "gripper_frame_link"
TICK = 2.0 * math.pi / 4095.0
HDR = ["t"] + [f"goal{i}" for i in range(6)] + [f"pos{i}" for i in range(6)]


def own_calibration():
    with open(PROJECT, encoding="utf-8") as fh:
        entries = json.load(fh)
    ent = [e for e in entries if e.get("type") == "SO101_Follower"]
    assert len(ent) == 1, len(ent)
    cal = ent[0]["payload"]["calibration"]
    rng, K = {}, {}
    for j in BODY + ["gripper"]:
        lo, hi = int(cal[j]["range_min"]), int(cal[j]["range_max"])
        rng[j] = (lo, hi)
        K[j] = (hi - lo) / (100.0 if j == "gripper" else 200.0) * TICK
    return rng, K


def rot_rpy(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def load_chain(path, tip=TIP, base="base_link"):
    root = ET.parse(path).getroot()
    by_child = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = [float(v) for v in o.get("xyz", "0 0 0").split()] if o is not None else [0.0, 0.0, 0.0]
        rpy = [float(v) for v in o.get("rpy", "0 0 0").split()] if o is not None else [0.0, 0.0, 0.0]
        ax = j.find("axis")
        axis = [float(v) for v in ax.get("xyz", "0 0 1").split()] if ax is not None else [0.0, 0.0, 1.0]
        T = np.eye(4)
        T[:3, :3] = rot_rpy(*rpy)
        T[:3, 3] = xyz
        child = j.find("child").get("link")
        by_child[child] = {"name": j.get("name"), "type": j.get("type"), "parent": j.find("parent").get("link"),
                           "T": T, "axis": np.array(axis)}
    chain, link = [], tip
    while link != base:
        if link not in by_child:
            raise ValueError(f"no joint leads to {link} in {path}")
        chain.append(by_child[link])
        link = by_child[link]["parent"]
    return list(reversed(chain))


def rodrigues(axis, th):
    a = axis / np.linalg.norm(axis)
    kx = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    s = np.sin(th)[:, None, None]
    c = np.cos(th)[:, None, None]
    return np.eye(3)[None] + s * kx[None] + (1.0 - c) * (kx @ kx)[None]


def fk_chain(chain, q):
    n = len(next(iter(q.values())))
    T = np.tile(np.eye(4), (n, 1, 1))
    for j in chain:
        T = T @ j["T"][None]
        if j["type"] == "revolute":
            M = np.tile(np.eye(4), (n, 1, 1))
            M[:, :3, :3] = rodrigues(j["axis"], q[j["name"]])
            T = T @ M
    return T


def ang_err_deg(Ra, Rb):
    M = np.einsum("nji,njk->nik", Ra, Rb)
    tr = np.trace(M, axis1=1, axis2=2)
    v = np.stack([M[:, 2, 1] - M[:, 1, 2], M[:, 0, 2] - M[:, 2, 0], M[:, 1, 0] - M[:, 0, 1]], axis=1) / 2.0
    return np.degrees(np.arctan2(np.linalg.norm(v, axis=1), (tr - 1.0) / 2.0))


def cmp_pair(pa, Ra, pb, Rb):
    dp = np.linalg.norm(pa - pb, axis=1)
    da = ang_err_deg(Ra, Rb)
    return {"n": int(len(dp)), "pos_mm_max": float(dp.max()), "pos_mm_median": float(np.median(dp)),
            "pos_mm_p99": float(np.percentile(dp, 99)), "ori_deg_max": float(da.max()),
            "rows_ge_0p5mm": int(np.sum(dp >= 0.5))}


def mj_tcp(path, q):
    import mujoco
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    adr = {}
    for j in BODY:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        assert jid >= 0, j
        adr[j] = int(m.jnt_qposadr[jid])
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, TIP)
    assert site >= 0, TIP
    n = len(q["shoulder_pan"])
    p = np.empty((n, 3))
    Rm = np.empty((n, 3, 3))
    for i in range(n):
        for j in BODY:
            d.qpos[adr[j]] = q[j][i]
        mujoco.mj_forward(m, d)
        p[i] = d.site_xpos[site]
        Rm[i] = d.site_xmat[site].reshape(3, 3)
    info = {"nbody": int(m.nbody), "njnt": int(m.njnt), "nsite": int(m.nsite), "ngeom": int(m.ngeom),
            "nmesh": int(m.nmesh)}
    return p * 1000.0, Rm, info


def kin_live(n5):
    sys.path.insert(0, f"{R}/app/tools")
    import kin
    n = len(n5)
    p = np.empty((n, 3))
    Rm = np.empty((n, 3, 3))
    for i in range(n):
        pm, rot = kin.fk([float(v) for v in n5[i]])
        p[i] = np.asarray(pm, dtype=float)
        Rm[i] = np.asarray(rot, dtype=float)
    return p, Rm, str(kin.URDF), [float(v) for v in kin.CAL.k]


def load_all():
    files = sorted(glob.glob(f"{R}/logs/sort/any_17*/ep_*/joints.csv"))
    TT, G, P, EP, names, counts = [], [], [], [], [], {}
    for ei, f in enumerate(files):
        rows = 0
        with open(f, newline="", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            assert rd.fieldnames == HDR, rd.fieldnames
            for rec in rd:
                vals = [float(rec[h]) for h in HDR]
                TT.append(vals[0])
                G.append(vals[1:7])
                P.append(vals[7:13])
                EP.append(ei)
                rows += 1
        rel = os.path.relpath(os.path.dirname(f), f"{R}/logs/sort")
        names.append(rel)
        counts[rel] = rows
    return files, names, np.array(TT), np.array(G), np.array(P), np.array(EP), counts


def gstats(p5, mask):
    x = p5[mask]
    if x.size == 0:
        return None
    mode = Counter(np.round(x, 2).tolist()).most_common(1)[0][0]
    return {"n": int(x.size), "median": float(np.median(x)), "mean": float(x.mean()), "min": float(x.min()),
            "max": float(x.max()), "mode_2dp": float(mode)}


def read_binary_stl(path):
    with open(path, "rb") as fh:
        data = fh.read()
    n = int(np.frombuffer(data[80:84], dtype="<u4")[0])
    if len(data) != 84 + 50 * n:
        raise ValueError(f"{path} is not a binary STL with {n} triangles")
    dt = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec = np.frombuffer(data, dtype=dt, count=n, offset=84)
    return rec["v"].reshape(-1, 3).astype(float), n


def main():
    t0 = time.time()
    rng, K = own_calibration()
    files, names, TT, G, P, EP, counts = load_all()
    res = {"rows": int(len(P)), "episodes": len(files), "rows_per_episode": counts,
           "calibration_ticks": {j: list(rng[j]) for j in rng}, "k_own": K}
    with open(G1_REPORT, encoding="utf-8") as fh:
        rep = json.load(fh)
    b_counts = {e["episode"]: e["rows"] for e in rep["candidates"]["twin_ours"]["per_episode"]}
    res["rows_match_builder_per_episode"] = bool(b_counts == counts)
    with open(KREF, encoding="utf-8") as fh:
        kref = json.load(fh)
    res["k_own_minus_kin_reference_max_abs"] = float(max(abs(K[j] - kref["k_rad_per_unit"][i])
                                                         for i, j in enumerate(BODY)))

    n5 = P[:, :5]
    q = {j: K[j] * n5[:, b] for b, j in enumerate(BODY)}
    chain = load_chain(URDF_KIN)
    res["kin_urdf_chain"] = [[c["name"], c["type"], [round(float(v), 6) for v in c["axis"]]] for c in chain]
    T_own = fk_chain(chain, q)
    p_own, R_own = T_own[:, :3, 3] * 1000.0, T_own[:, :3, :3]
    p_kin, R_kin, kin_urdf, kin_k = kin_live(n5)
    p_tw, R_tw, info_tw = mj_tcp(MJCF_TWIN, q)
    p_of, R_of, info_of = mj_tcp(MJCF_OFF, q)
    res["kin_urdf_used_by_kin_py"] = kin_urdf
    res["kin_k_used_by_kin_py"] = kin_k
    res["mjcf_twin_counts"] = info_tw
    res["mjcf_official_counts"] = info_of
    full = {}
    full["mujoco_twin vs kin_live"] = cmp_pair(p_tw, R_tw, p_kin, R_kin)
    full["own_fk vs kin_live"] = cmp_pair(p_own, R_own, p_kin, R_kin)
    full["mujoco_twin vs own_fk"] = cmp_pair(p_tw, R_tw, p_own, R_own)
    full["mujoco_official vs kin_live"] = cmp_pair(p_of, R_of, p_kin, R_kin)
    full["mujoco_twin vs mujoco_official"] = cmp_pair(p_tw, R_tw, p_of, R_of)
    try:
        chain_pl = load_chain(URDF_PLUG)
        T_pl = fk_chain(chain_pl, q)
        res["plugin_urdf_chain"] = [[c["name"], c["type"], [round(float(v), 6) for v in c["axis"]]]
                                    for c in chain_pl]
        full["own_fk(plugin urdf) vs own_fk(kin urdf)"] = cmp_pair(T_pl[:, :3, 3] * 1000.0, T_pl[:, :3, :3],
                                                                   p_own, R_own)
    except Exception as exc:  # report, do not hide
        res["plugin_urdf_error"] = f"{type(exc).__name__}: {exc}"
    res["full_rows"] = full

    with open(FIXED, encoding="utf-8") as fh:
        fx = json.load(fh)["samples"]
    fj = np.array([s["joints_norm"] for s in fx], dtype=float)[:, :5]
    s_p = np.array([s["kin_tcp_mm"] for s in fx], dtype=float)
    s_R = np.array([s["kin_R"] for s in fx], dtype=float).reshape(-1, 3, 3)
    fq = {j: K[j] * fj[:, b] for b, j in enumerate(BODY)}
    fT = fk_chain(chain, fq)
    f_own_p, f_own_R = fT[:, :3, 3] * 1000.0, fT[:, :3, :3]
    f_kin_p, f_kin_R, _, _ = kin_live(fj)
    f_tw_p, f_tw_R, _ = mj_tcp(MJCF_TWIN, fq)
    res["fixed200"] = {
        "n": len(fx),
        "stored_vs_live_kin": cmp_pair(s_p, s_R, f_kin_p, f_kin_R),
        "mujoco_twin_vs_stored": cmp_pair(f_tw_p, f_tw_R, s_p, s_R),
        "mujoco_twin_vs_own_fk": cmp_pair(f_tw_p, f_tw_R, f_own_p, f_own_R),
        "mujoco_twin_vs_live_kin": cmp_pair(f_tw_p, f_tw_R, f_kin_p, f_kin_R),
    }

    sens = {}
    for j in BODY:
        qq = dict(q)
        qq[j] = q[j] + TICK
        d = np.linalg.norm(fk_chain(chain, qq)[:, :3, 3] * 1000.0 - p_own, axis=1)
        sens["plus_one_encoder_tick_" + j] = float(d.max())
    qq = {j: q[j] * 4095.0 / 4096.0 for j in BODY}
    d = np.linalg.norm(fk_chain(chain, qq)[:, :3, 3] * 1000.0 - p_own, axis=1)
    sens["ticks_per_turn_4096_instead_of_4095"] = float(d.max())
    res["sensitivity_tcp_shift_max_mm"] = sens

    z = p_own[:, 2]
    res["tcp_z_mm_own"] = {"min": float(z.min()), "p0_1": float(np.percentile(z, 0.1)),
                           "median": float(np.median(z)), "rows_below_minus32p3": int(np.sum(z < -32.3))}

    g5, p5 = G[:, 5], P[:, 5]
    res["goal5_top_values"] = [[float(v), int(c)] for v, c in Counter(np.round(g5, 2).tolist()).most_common(6)]
    res["goal5_distinct_2dp"] = int(len(np.unique(np.round(g5, 2))))
    closed = g5 <= 1.0
    empty = closed & (p5 <= 1.0)
    held = closed & (p5 >= 1.57) & (p5 <= 2.42)
    mid = closed & (p5 > 1.0) & (p5 < 1.57)
    high = closed & (p5 > 2.42)
    opn = np.isclose(g5, 25.0, atol=1e-9)
    sel_open = np.abs(p5 - 25.0) <= 0.5
    res["gripper"] = {
        "closed_cmd_rows_goal5_le_1": int(closed.sum()),
        "empty_close_closed_cmd_p5_le_1": gstats(p5, empty),
        "closed_cmd_p5_between_1_and_1p57": gstats(p5, mid),
        "held_closed_cmd_p5_1p57_to_2p42": gstats(p5, held),
        "closed_cmd_p5_gt_2p42": gstats(p5, high),
        "open_goal5_eq_25": gstats(p5, opn),
        "open_selected_by_reading_within_0p5_of_25": gstats(p5, sel_open),
        "top_closed_cmd_p5_values_2dp": [[float(v), int(c)] for v, c in
                                         Counter(np.round(p5[closed], 2).tolist()).most_common(12)],
    }
    res["held_rows_per_episode"] = {names[i]: int(np.sum(held & (EP == i)))
                                    for i in range(len(names)) if np.sum(held & (EP == i)) > 0}
    res["held_tcp_z_median_mm"] = float(np.median(z[held])) if held.any() else None

    V, nt = read_binary_stl(JAW_STL)
    V = V + np.array([0.0, 0.0, JAW_BODY_Z])
    rad = np.hypot(V[:, 0], V[:, 1])
    tip_m = float(rad.max())
    res["jaw"] = {"stl_triangles": int(nt), "radial_max_raw_units": tip_m,
                  "radial_max_mm_if_stl_in_metres": tip_m * 1000.0,
                  "mm_tip_travel_per_gripper_unit_if_metres": tip_m * 1000.0 * K["gripper"],
                  "units_note": "STL units assumed metres (MuJoCo mesh convention); check the raw value"}

    os.makedirs(f"{V1}/outputs", exist_ok=True)
    with open(f"{V1}/outputs/indep_results.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("rows", res["rows"], "episodes", res["episodes"], "rows_match_builder", res["rows_match_builder_per_episode"])
    print("k_own_minus_kin_reference_max_abs", res["k_own_minus_kin_reference_max_abs"])
    for name, v in full.items():
        print("FULL", name, "pos_max_mm", v["pos_mm_max"], "ori_max_deg", v["ori_deg_max"], "ge_0.5", v["rows_ge_0p5mm"])
    for name, v in res["fixed200"].items():
        if isinstance(v, dict):
            print("FIXED200", name, "pos_max_mm", v["pos_mm_max"], "ori_max_deg", v["ori_deg_max"])
    print("SENS", json.dumps(sens))
    print("GRIPPER empty", res["gripper"]["empty_close_closed_cmd_p5_le_1"])
    print("GRIPPER held", res["gripper"]["held_closed_cmd_p5_1p57_to_2p42"])
    print("GRIPPER open25", res["gripper"]["open_goal5_eq_25"])
    print("GRIPPER open_sel", res["gripper"]["open_selected_by_reading_within_0p5_of_25"])
    print("JAW", json.dumps(res["jaw"]))
    print("TCPZ", json.dumps(res["tcp_z_mm_own"]))
    print("seconds", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
