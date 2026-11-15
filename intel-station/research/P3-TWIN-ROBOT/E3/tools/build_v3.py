#!/usr/bin/env python3
"""E3: build so101_twin_v3.xml from v2. The jaw hulls (gripper_link and moving jaw, group 3) are replaced by box pads
fitted to the jaw vertices within RPAD of the TCP (in each body frame). Gripper actuator gain and force limit are set
here. Then: gap table between the pads over the gripper range, G1 on v3, and the grasp test (nominal and 10 seeds).
Run: sim-env/bin/python -I -B build_v3.py [rpad_mm] [kp] [forcerange]"""
import json
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

R = "/home/intel/intel-junction-2026"
ROBOT = f"{R}/sim/twin/robot"
V2 = f"{ROBOT}/so101_twin_v2.xml"
V3 = f"{ROBOT}/so101_twin_v3.xml"
OUT = f"{R}/work/P3-TWIN-ROBOT/E3/outputs/build_v3.json"
sys.path.insert(0, f"{R}/sim/twin/tests")
import test_grasp as tg  # noqa: E402

PADS = [("gripper_link", "wrist_roll_follower_so101_v1", "pad_fixed"),
        ("moving_jaw_so101_v1_link", "moving_jaw_so101_v1", "pad_moving")]
KP_KV_RATIO = 0.00154278912 / 0.5


def quat_mat(q):
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, np.asarray(q, dtype=float))
    return r.reshape(3, 3)


def pad_boxes(m, d, rpad):
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_frame_link")
    tcp_w = d.site_xpos[site].copy()
    out = {}
    for body, mesh, name in PADS:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
        g = [i for i in range(m.ngeom) if m.geom_bodyid[i] == bid and m.geom_group[i] == 3
             and m.geom_dataid[i] >= 0 and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, int(m.geom_dataid[i])) == mesh]
        g = g[0]
        mid = int(m.geom_dataid[g])
        v0, nv = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        verts = m.mesh_vert[v0:v0 + nv]
        pb = verts @ quat_mat(m.geom_quat[g]).T + m.geom_pos[g]           # mesh frame -> body frame
        tcp_b = d.xmat[bid].reshape(3, 3).T @ (tcp_w - d.xpos[bid])        # TCP in body frame
        sel = pb[np.linalg.norm(pb - tcp_b, axis=1) < rpad]
        lo, hi = sel.min(axis=0), sel.max(axis=0)
        out[name] = {"body": body, "mesh": mesh, "n_vertices": int(len(sel)),
                     "center_m": ((lo + hi) / 2).tolist(), "half_m": (np.maximum((hi - lo) / 2, 0.001)).tolist()}
    return out


def write_v3(pads, kp, forcerange):
    tree = ET.parse(V2)
    root = tree.getroot()
    for body in root.iter("body"):
        names = [p[2] for p in PADS if p[0] == body.get("name")]
        if not names:
            continue
        for g in list(body.findall("geom")):
            if g.get("group") == "3":
                body.remove(g)
        for name in names:
            p = pads[name]
            ET.SubElement(body, "geom", name=name, type="box",
                          size=" ".join(f"{x:.6f}" for x in p["half_m"]),
                          pos=" ".join(f"{x:.6f}" for x in p["center_m"]),
                          contype="1", conaffinity="1", group="3", condim="4",
                          friction="0.35 0.005 0.0001", rgba="0.2 0.6 1.0 0.3")
    for a in root.iter("position"):
        if a.get("name") == "act_gripper":
            a.set("kp", repr(kp))
            a.set("kv", repr(kp * KP_KV_RATIO))
            a.set("forcerange", f"{-forcerange} {forcerange}")
    ET.indent(root, space="  ")
    text = ET.tostring(root, encoding="unicode")
    header = ("<!-- so101_twin_v3.xml (E3, P3-TWIN-ROBOT). From so101_twin_v2.xml by work/P3-TWIN-ROBOT/E3/tools/"
              "build_v3.py. Jaw hulls replaced by box pads fitted to jaw vertices within RPAD of the TCP. Gripper "
              "position gain and force limit set for the stall behaviour (see README). -->\n")
    with open(V3, "w") as fh:
        fh.write(header + text + "\n")


def gap_table(readings):
    m = mujoco.MjModel.from_xml_path(V3)
    d = mujoco.MjData(m)
    gf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pad_fixed")
    gm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pad_moving")
    jq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    rows = []
    for n in readings:
        d.qpos[m.jnt_qposadr[jq]] = tg.units.gripper_norm_to_rad(n, tg.LAW)
        mujoco.mj_forward(m, d)
        dist = mujoco.mj_geomDistance(m, d, gf, gm, 0.5, np.zeros(6))
        rows.append({"reading": n, "gap_mm": round(dist * 1000, 2)})
    return rows


def main():
    rpad = float(sys.argv[1]) / 1000 if len(sys.argv) > 1 else 0.03
    kp = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    fr = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    m = mujoco.MjModel.from_xml_path(V2)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    pads = pad_boxes(m, d, rpad)
    write_v3(pads, kp, fr)
    res = {"rpad_mm": rpad * 1000, "kp": kp, "forcerange": fr, "pads": pads}
    res["gap_mm_by_reading"] = gap_table([0.52, 1.5, 1.57, 2.29, 5.0, 10.0, 15.0, 20.0, 24.6])
    tg.SRC = V3
    res["g1_v3"] = dict(zip(("max_pos_mm", "max_ori_deg", "n"), tg._g1_errors(V3)))
    nom = tg.run_sequence()
    emp = tg.run_sequence(brick=False)
    seeds = [tg.run_sequence(*tg.seed_case(s)) for s in range(10)]
    res["nominal"] = {k: nom.get(k) for k in ("held", "brick_lift_mm", "reading_after_close", "reading_after_lift",
                                              "tcp_brick_xy_mm", "lift_contacts", "closed_contacts")}
    res["empty"] = {"reading_after_close": emp["reading_after_close"]}
    res["seeds_held"] = sum(1 for s in seeds if s["held"])
    res["seed_readings_after_lift"] = [s["reading_after_lift"] for s in seeds]
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("rpad_mm", "kp", "forcerange", "gap_mm_by_reading", "g1_v3", "nominal",
                                          "empty", "seeds_held")}, indent=1, default=str))


if __name__ == "__main__":
    main()
