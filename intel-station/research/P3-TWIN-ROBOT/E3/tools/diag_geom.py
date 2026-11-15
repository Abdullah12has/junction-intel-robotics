#!/usr/bin/env python3
"""E3 diagnostic: where are the jaws relative to the TCP site and the brick at the descended hover pose, and what is the
hull gap at open and closed. Output: outputs/diag_geom.json. Run with: sim-env/bin/python -I -B diag_geom.py"""
import json
import sys

import mujoco
import numpy as np

R = "/home/intel/intel-junction-2026"
sys.path.insert(0, f"{R}/sim/twin/tests")
import test_grasp as tg  # noqa: E402
tg.SRC = f"{R}/sim/twin/robot/so101_twin_v2.xml"

OUT = f"{R}/work/P3-TWIN-ROBOT/E3/outputs/diag_geom.json"


def mm(v):
    return [round(float(x) * 1000, 2) for x in v]


def main():
    m = tg.build_model(tg.BRICK_XY, 0.0)
    rig = tg.Rig(m)
    d = rig.d
    gf = tg.geom_by_mesh(m, tg.JAW_FIXED_MESH)
    gm = tg.geom_by_mesh(m, tg.JAW_MOVING_MESH)
    g_open, g_closed = tg.gripper_ctrl(25.0), tg.gripper_ctrl(0.0)
    out = {}
    q = np.zeros(5)
    tg.settle(rig, q, g_open, 1.0)
    q, s1 = tg.goto(rig, [tg.BRICK_XY[0], tg.BRICK_XY[1], tg.MAT_Z + tg.TIP_FINAL], q, g_open)
    out["descended_8mm"] = {"stage": s1, "tcp_mm": mm(d.site_xpos[rig.site]),
                            "brick_centre_mm": mm(d.xpos[rig.brick]),
                            "fixed_jaw_centre_mm": mm(d.geom_xpos[gf]), "moving_jaw_centre_mm": mm(d.geom_xpos[gm]),
                            "mat_top_mm": round(tg.MAT_Z * 1000, 2)}
    out["open_gap_mm"] = round(mujoco.mj_geomDistance(m, d, gf, gm, 0.5, np.zeros(6)) * 1000, 2)
    tg.settle(rig, q, g_closed, 1.0)
    out["closed_gap_mm"] = round(mujoco.mj_geomDistance(m, d, gf, gm, 0.5, np.zeros(6)) * 1000, 2)
    out["closed_fixed_jaw_centre_mm"] = mm(d.geom_xpos[gf])
    out["closed_moving_jaw_centre_mm"] = mm(d.geom_xpos[gm])
    out["closed_reading"] = round(tg.gripper_reading(rig), 4)
    # jaw geometry in the gripper_link frame at the zero pose: AABB of each mesh geom (world, open)
    for name, g in (("fixed", gf), ("moving", gm)):
        mid = m.geom_dataid[g]
        v0, nv = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        verts = m.mesh_vert[v0:v0 + nv]
        Rg = d.geom_xmat[g].reshape(3, 3)
        wv = verts @ Rg.T + d.geom_xpos[g]
        out[f"{name}_jaw_aabb_mm"] = {"min": mm(wv.min(axis=0)), "max": mm(wv.max(axis=0))}
    json.dump(out, open(OUT, "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
