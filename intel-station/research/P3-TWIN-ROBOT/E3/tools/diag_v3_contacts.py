#!/usr/bin/env python3
"""E3: stage-by-stage brick position and brick contact partners (body pairs) for the grasp sequence on v3.
Output: outputs/diag_v3_contacts.json. Run with: sim-env/bin/python -I -B diag_v3_contacts.py"""
import json
import sys

import mujoco
import numpy as np

R = "/home/intel/intel-junction-2026"
sys.path.insert(0, f"{R}/sim/twin/tests")
import test_grasp as tg  # noqa: E402

tg.SRC = f"{R}/sim/twin/robot/so101_twin_v3.xml"
OUT = f"{R}/work/P3-TWIN-ROBOT/E3/outputs/diag_v3_contacts.json"


def partners(rig):
    m, d = rig.m, rig.d
    res = []
    for i in range(d.ncon):
        c = d.contact[i]
        b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[c.geom1]))
        b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[c.geom2]))
        if "brick" in (b1, b2):
            res.append([b1, b2, round(float(c.dist) * 1000, 2)])
    return res


def main():
    m = tg.build_model(tg.BRICK_XY, 0.0)
    rig = tg.Rig(m)
    d = rig.d
    g_open, g_closed = tg.gripper_ctrl(25.0), tg.gripper_ctrl(0.0)
    log = []
    q = np.zeros(5)
    tg.settle(rig, q, g_open, 1.0)
    log.append(("start_open", d.xpos[rig.brick].round(4).tolist(), partners(rig)))
    stages = [("hover", tg.MAT_Z + tg.HOVER_DZ, g_open), ("d26", tg.MAT_Z + tg.TIP_TARGET, g_open),
              ("d8", tg.MAT_Z + tg.TIP_FINAL, g_open)]
    for name, z, g in stages:
        q, s = tg.goto(rig, [tg.BRICK_XY[0], tg.BRICK_XY[1], z], q, g)
        log.append((name, d.xpos[rig.brick].round(4).tolist(), partners(rig), s))
    tg.settle(rig, q, g_closed, 1.0)
    log.append(("closed", d.xpos[rig.brick].round(4).tolist(), partners(rig),
                {"reading": round(tg.gripper_reading(rig), 4)}))
    json.dump([list(x) for x in log], open(OUT, "w"), indent=1, default=str)
    for row in log:
        print(json.dumps(row, default=str))


if __name__ == "__main__":
    main()
