#!/usr/bin/env python3
"""E1 stage A (data-env, numpy only): kinematics reference from app/tools/kin.py for every recorded joint state.

Reads R/logs/sort/any_17*/ep_*/joints.csv (read only). For each row, kin.fk(pos0..pos4) gives the TCP position (mm,
base frame) and the 3x3 TCP rotation of gripper_frame_link. Writes:
  E1/kin_reference.npz                         all rows (episode, row, t, joints pos0..5, kin TCP mm, kin R)
  E1/kin_reference_summary.json                per-episode counts, file hashes, skipped rows, calibration used
  R/sim/twin/tests/data/g1_fixed_sample.json   200 evenly spaced rows (fixed sample for the pytest G1 test)
  R/sim/twin/tests/data/kin_calibration_reference.json  kin.CAL values for the unit-mapping test
Run: nice -n 19 $DATA_PY -I kin_reference.py"""
import csv
import glob
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/intel/intel-junction-2026/app/tools")
import kin  # noqa: E402  (kin.fk(n5) -> (TCP mm, 3x3); kin.CAL from project_robots.json SO101_Follower)

R = "/home/intel/intel-junction-2026"
E1 = f"{R}/work/P3-TWIN-ROBOT/E1"
TESTS_DATA = f"{R}/sim/twin/tests/data"
HDR = ["t"] + [f"goal{i}" for i in range(6)] + [f"pos{i}" for i in range(6)]
FIXED_N = 200


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_joints(path):
    rows, bad = [], 0
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        if header != HDR:
            raise ValueError(f"unexpected header in {path}: {header}")
        for k, rec in enumerate(rd):
            try:
                vals = [float(v) for v in rec]
            except ValueError:
                bad += 1
                continue
            if len(vals) != 13 or not all(np.isfinite(vals)):
                bad += 1
                continue
            rows.append((k, vals))
    return rows, bad


def main():
    files = sorted(glob.glob(f"{R}/logs/sort/any_17*/ep_*/joints.csv"))
    if not files:
        raise SystemExit("no joints.csv found")
    ep_id, row_id, t_all, joints, kin_p, kin_r = [], [], [], [], [], []
    per_episode, names = [], []
    for ei, path in enumerate(files):
        rows, bad = read_joints(path)
        rel = os.path.relpath(os.path.dirname(path), f"{R}/logs/sort")
        names.append(rel)
        for k, vals in rows:
            n5 = vals[7:12]
            p_mm, rot = kin.fk(n5)
            ep_id.append(ei)
            row_id.append(k)
            t_all.append(vals[0])
            joints.append(vals[7:13])
            kin_p.append(np.asarray(p_mm, dtype=float))
            kin_r.append(np.asarray(rot, dtype=float))
        per_episode.append({"episode": rel, "file": path, "sha256": sha256(path), "rows_used": len(rows),
                            "rows_skipped_bad": bad})
    ep_id = np.asarray(ep_id)
    row_id = np.asarray(row_id)
    t_all = np.asarray(t_all)
    joints = np.asarray(joints)
    kin_p = np.asarray(kin_p)
    kin_r = np.asarray(kin_r)
    np.savez(f"{E1}/kin_reference.npz", episode_index=ep_id, row_index=row_id, t=t_all, joints=joints,
             kin_tcp_mm=kin_p, kin_R=kin_r, episode_names=np.asarray(names))
    cal = kin.CAL
    calib = {j: {"range_min": c.range_min, "range_max": c.range_max, "homing_offset": c.homing_offset}
             for j, c in cal.joints.items()}
    summary = {"episodes": per_episode, "n_rows": int(len(ep_id)), "kin_source": kin.URDF,
               "kin_tcp": "gripper_frame_link (Chain default tip)", "calibration_json": kin.R + "/app/motion/project_robots.json",
               "calibration": calib, "k_rad_per_unit": [float(v) for v in cal.k]}
    with open(f"{E1}/kin_reference_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    # fixed sample for the pytest G1 test
    idx = np.linspace(0, len(ep_id) - 1, FIXED_N).round().astype(int)
    if len(np.unique(idx)) != FIXED_N:
        raise ValueError("fixed sample indices are not unique")
    sample = []
    for i in idx:
        sample.append({"episode": names[ep_id[i]], "row": int(row_id[i]),
                       "joints_norm": [float(v) for v in joints[i, :5]],
                       "kin_tcp_mm": [float(v) for v in kin_p[i]],
                       "kin_R": [float(v) for v in kin_r[i].reshape(9)]})
    os.makedirs(TESTS_DATA, exist_ok=True)
    with open(f"{TESTS_DATA}/g1_fixed_sample.json", "w", encoding="utf-8") as fh:
        json.dump({"source": "R/logs/sort/any_17*/ep_*/joints.csv rows, evenly spaced (np.linspace over all rows)",
                   "reference": "app/tools/kin.py kin.fk (data-env)", "n": FIXED_N, "samples": sample}, fh)
    # kin calibration reference for the unit mapping test
    rng = np.random.default_rng(20261114)
    pts = [[-100.0] * 5, [100.0] * 5, [0.0] * 5, [0.59, 25.0, -40.0, 62.0, 3.5]]
    pts += [list(rng.uniform(-100, 100, 5)) for _ in range(12)]
    ref = [{"norm": [float(v) for v in p], "rad": [float(v) for v in cal.rad_from_norm(p)]} for p in pts]
    with open(f"{TESTS_DATA}/kin_calibration_reference.json", "w", encoding="utf-8") as fh:
        json.dump({"source": "app/tools/kin.py CAL.rad_from_norm (pick_v2/kinematics.py FollowerCalibration)",
                   "k_rad_per_unit": [float(v) for v in cal.k], "points": ref}, fh)
    print("episodes:", len(files), "rows:", len(ep_id), "k:", [round(float(v), 9) for v in cal.k])


if __name__ == "__main__":
    main()
