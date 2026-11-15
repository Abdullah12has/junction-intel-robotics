#!/usr/bin/env python3
"""V1 independent check of scene_twin_v1.xml: loads the builder's file read-only, counts bodies and geoms,
reads brick mass and friction, and simulates 5 s of free settling, tracking XY drift, z change and tilt."""
import json
import math

import mujoco
import numpy as np

XML = "/home/intel/intel-junction-2026/sim/twin/scene/scene_twin_v1.xml"
OUT = "/home/intel/intel-junction-2026/work/P3-TWIN-SCENE/V1/outputs/mjcf_check.json"
BRICKS = ["brick_blue", "brick_orange", "brick_yellow", "brick_white"]
SIM_S = 5.0


def main():
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    res = {"mujoco_version": mujoco.__version__, "xml": XML, "timestep_s": float(m.opt.timestep),
           "nbody_incl_world": int(m.nbody), "ngeom": int(m.ngeom), "njnt": int(m.njnt), "nlight": int(m.nlight)}
    res["bodies"] = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    per_body = {}
    for g in range(m.ngeom):
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
        per_body[bn] = per_body.get(bn, 0) + 1
    res["geoms_per_body"] = per_body
    bid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in BRICKS}
    gid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n + "_geom") for n in BRICKS}
    res["brick_mass_g"] = {n: round(float(m.body_mass[bid[n]]) * 1000.0, 4) for n in BRICKS}
    res["brick_sliding_friction"] = {n: float(m.geom_friction[gid[n], 0]) for n in BRICKS}
    res["brick_half_size_mm"] = {n: [round(float(x) * 1000.0, 3) for x in m.geom_size[gid[n]]] for n in BRICKS}
    mujoco.mj_forward(m, d)
    start = {n: d.xpos[bid[n]].copy() for n in BRICKS}
    res["start_z_mm"] = {n: round(float(start[n][2]) * 1000.0, 3) for n in BRICKS}
    worst = {n: {"max_xy_drift_mm": 0.0, "max_abs_dz_mm": 0.0, "max_tilt_deg": 0.0} for n in BRICKS}
    R = np.zeros(9)
    max_ncon, min_dist = 0, 1e9
    for _ in range(int(round(SIM_S / m.opt.timestep))):
        mujoco.mj_step(m, d)
        max_ncon = max(max_ncon, int(d.ncon))
        for i in range(int(d.ncon)):
            min_dist = min(min_dist, float(d.contact[i].dist) * 1000.0)
        for n in BRICKS:
            p = d.xpos[bid[n]]
            p0 = start[n]
            mujoco.mju_quat2Mat(R, d.xquat[bid[n]])
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(R[8])))))
            w = worst[n]
            w["max_xy_drift_mm"] = max(w["max_xy_drift_mm"], 1000.0 * math.hypot(p[0] - p0[0], p[1] - p0[1]))
            w["max_abs_dz_mm"] = max(w["max_abs_dz_mm"], 1000.0 * abs(p[2] - p0[2]))
            w["max_tilt_deg"] = max(w["max_tilt_deg"], tilt)
    res["sim_seconds"] = round(float(d.time), 4)
    res["final_z_mm"] = {n: round(float(d.xpos[bid[n]][2]) * 1000.0, 3) for n in BRICKS}
    res["worst_over_sim"] = {n: {k: round(v, 4) for k, v in worst[n].items()} for n in BRICKS}
    res["max_simultaneous_contacts"] = max_ncon
    res["min_contact_dist_mm"] = round(min_dist, 4)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
