#!/usr/bin/env python3
"""E2 check: compile so101_twin_v2.xml, count geoms and actuators, and re-run G1 (fixed 200-sample set, 0.5 mm gate)
on v2 with the E1 tcp.py helpers. Run with: python -I -B check_v2.py"""
import json
import sys

import numpy as np
import mujoco

R = "/home/intel/intel-junction-2026"
ROBOT = f"{R}/sim/twin/robot"
V2 = f"{ROBOT}/so101_twin_v2.xml"
V1 = f"{ROBOT}/so101_twin.xml"
G1 = f"{R}/sim/twin/tests/data/g1_fixed_sample.json"
sys.path.insert(0, ROBOT)
import tcp  # noqa: E402


def main():
    out = {}
    for name, path in (("v1", V1), ("v2", V2)):
        m = mujoco.MjModel.from_xml_path(path)
        out[name] = {"bodies": m.nbody, "joints": m.njnt, "geoms": m.ngeom,
                     "collision_geoms_group3": int(np.sum(m.geom_group == 3)),
                     "actuators": m.nu, "excludes": m.nexclude}
        print(name, out[name])
    s = json.load(open(G1))
    samples = s["samples"]
    print("sample keys", sorted(samples[0].keys()))
    jn = np.array([x["joints_norm"] for x in samples], dtype=float)
    kp = np.array([x["kin_tcp_mm"] for x in samples], dtype=float)
    kr = np.array([x["kin_R"] for x in samples], dtype=float).reshape(-1, 3, 3)
    arm = tcp.ArmModel(V2)
    p, rot = tcp.tcp_batch(arm, jn)
    pos, ori = tcp.pose_errors(p, rot, kp, kr)
    out["g1_v2"] = {"n": int(len(jn)), "max_pos_mm": float(np.max(pos)), "max_ori_deg": float(np.max(ori)),
                    "gate_mm": 0.5, "pass": bool(np.max(pos) < 0.5)}
    print("G1 v2", out["g1_v2"])
    json.dump(out, open(f"{R}/work/P3-TWIN-ROBOT/E2/outputs/check_v2.json", "w"), indent=2)


if __name__ == "__main__":
    main()
