#!/usr/bin/env python3
"""V1: TCP and gripper effect of the plugin's normalized convention (repos/physicalai/.../profiles/so101.py,
SO101_JOINT_RANGES, pinned to URDF limits) versus our calibration convention (kin.py / project_robots.json, used by
the twin). Same 21693 logged rows, same URDF chain (indep_check's own FK). Writes outputs/plugin_units_check.json.
Run: nice -n 19 env OMP_NUM_THREADS=4 SIM_PY -I -B plugin_units_check.py"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/intel/intel-junction-2026/work/P3-TWIN-ROBOT/V1/tools")
import indep_check as ic  # noqa: E402  (module guard: only definitions are imported)

V1 = "/home/intel/intel-junction-2026/work/P3-TWIN-ROBOT/V1"
PLUG = {  # profiles/so101.py SO101_JOINT_RANGES, radians
    "shoulder_pan": (-1.9198621771937616, 1.9198621771937634),
    "shoulder_lift": (-1.7453292519943224, 1.7453292519943366),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.6580628494556928, 1.6580627293335335),
    "wrist_roll": (-2.7438472969992493, 2.841206309382605),
    "gripper": (-0.17453297762778586, 1.7453291995659765),
}


def main():
    rng, K = ic.own_calibration()
    files, names, TT, G, P, EP, counts = ic.load_all()
    n5 = P[:, :5]
    chain = ic.load_chain(ic.URDF_KIN)
    q_cal = {j: K[j] * n5[:, b] for b, j in enumerate(ic.BODY)}
    q_pl = {}
    scale = {}
    for b, j in enumerate(ic.BODY):
        lo, hi = PLUG[j]
        q_pl[j] = lo + (n5[:, b] + 100.0) / 200.0 * (hi - lo)
        scale[j] = {"plugin_rad_per_unit": (hi - lo) / 200.0, "kin_rad_per_unit": K[j],
                    "ratio_plugin_over_kin": ((hi - lo) / 200.0) / K[j]}
    glo, ghi = PLUG["gripper"]
    scale["gripper"] = {"plugin_rad_per_unit": (ghi - glo) / 100.0, "kin_rad_per_unit": K["gripper"],
                        "ratio_plugin_over_kin": ((ghi - glo) / 100.0) / K["gripper"]}
    p_cal = ic.fk_chain(chain, q_cal)[:, :3, 3] * 1000.0
    p_pl = ic.fk_chain(chain, q_pl)[:, :3, 3] * 1000.0
    d = np.linalg.norm(p_pl - p_cal, axis=1)
    # gripper jaw angle at a few readings: plugin law (closed stop at n=0, plugin scale) vs kin closed_at_0 law
    stop = -0.17453292519943295  # URDF gripper lower limit, rad
    grip = {}
    for n in (0.52, 1.57, 2.42, 24.62):
        q_p = glo + n / 100.0 * (ghi - glo)
        q_k = stop + K["gripper"] * n
        grip[str(n)] = {"plugin_deg": float(np.degrees(q_p)), "kin_closed_at_0_deg": float(np.degrees(q_k)),
                        "difference_deg": float(np.degrees(q_k - q_p))}
    res = {
        "rows": int(len(d)),
        "scale_per_joint": scale,
        "tcp_shift_plugin_vs_calibration_mm": {"max": float(d.max()), "median": float(np.median(d)),
                                               "p99": float(np.percentile(d, 99))},
        "gripper_jaw_angle_at_readings": grip,
        "note": ("Plugin span = URDF limits; calibration span = project_robots.json ticks x 2*pi/4095. "
                 "Both are linear maps; the twin uses the calibration convention, which the logs use."),
    }
    os.makedirs(f"{V1}/outputs", exist_ok=True)
    with open(f"{V1}/outputs/plugin_units_check.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    for j in scale:
        print(j, "plugin", round(scale[j]["plugin_rad_per_unit"], 8), "kin", round(scale[j]["kin_rad_per_unit"], 8),
              "ratio", round(scale[j]["ratio_plugin_over_kin"], 5))
    print("TCP shift plugin vs calibration (mm): max", round(float(d.max()), 2), "median", round(float(np.median(d)), 2))
    print("GRIPPER", json.dumps(grip))


if __name__ == "__main__":
    main()
