#!/usr/bin/env python3
"""Tests for the SO-101 twin robot model (P3-TWIN-ROBOT, E1).

G1 (test_g1_*): TCP pose from MuJoCo against kin.fk on the 200-sample fixed set (data/g1_fixed_sample.json),
position threshold 0.5 mm; orientation error printed, not gated.
Units: units.py against kin's own k values and norm-to-rad points (data/kin_calibration_reference.json), against
app/motion/project_robots.json, against the URDF joint limits, and the gripper law identities.
Model files: every mesh that so101_twin.xml references is present and is not a Git LFS pointer.
Run without pytest: nice -n 19 $SIM_PY -I -B test_kinematics.py   (exit 0 when every test passes)."""
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT = os.path.normpath(os.path.join(HERE, "..", "robot"))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, ROBOT)
import tcp  # noqa: E402
import units  # noqa: E402

R = "/home/intel/intel-junction-2026"
THRESH_MM = 0.5
URDF = (f"{R}/repos/edge-developer-kit-reference-scripts/samples/ai/edge-ai-demo-studio/workers/services/"
        "robotics-ai/platforms/so101/urdf/so101.urdf")
PROJECT_JSON = f"{R}/app/motion/project_robots.json"
OURS = os.path.join(ROBOT, "so101_twin.xml")
MESH_DIR = os.path.join(ROBOT, "meshes")
LFS_HEAD = b"version https://git-lfs.github.com/spec/v1"


def _load_data(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        return json.load(fh)


def _fixed_sample():
    fixed = _load_data("g1_fixed_sample.json")
    if isinstance(fixed, dict):
        fixed = fixed.get("samples", [])
    j = np.array([s["joints_norm"] for s in fixed], dtype=float)[:, :5]
    p = np.array([s["kin_tcp_mm"] for s in fixed], dtype=float)
    rot = np.array([s["kin_R"] for s in fixed], dtype=float).reshape(-1, 3, 3)
    return j, p, rot


def test_k_matches_kin_reference():
    ref = _load_data("kin_calibration_reference.json")
    for joint, k_ref in zip(units.BODY_JOINTS, ref["k_rad_per_unit"]):
        assert abs(units.K[joint] - k_ref) <= 1e-12 * abs(k_ref), (joint, units.K[joint], k_ref)


def test_norm_to_rad_matches_kin_points():
    ref = _load_data("kin_calibration_reference.json")
    for pt in ref["points"]:
        for g, r in zip(units.norm_to_rad(pt["norm"]), pt["rad"]):
            assert abs(g - r) <= 1e-12, (pt["norm"], g, r)
        for b, n in zip(units.rad_to_norm(pt["rad"]), pt["norm"]):
            assert abs(b - n) <= 1e-9, (pt["norm"], b, n)


def test_calibration_matches_project_json():
    with open(PROJECT_JSON, encoding="utf-8") as fh:
        data = json.load(fh)
    entry = next(e for e in data if e.get("type") == "SO101_Follower")
    cal = entry["payload"]["calibration"]
    for joint in units.JOINTS:
        assert (int(cal[joint]["range_min"]), int(cal[joint]["range_max"])) == units.CALIBRATION[joint], joint
        assert int(cal[joint]["drive_mode"]) == 0, joint


def test_urdf_limits_match():
    root = ET.parse(URDF).getroot()
    elements = {el.get("name"): el for el in root.findall("joint")}
    for joint in units.JOINTS:
        lim = elements[joint].find("limit")
        lo, hi = float(lim.get("lower")), float(lim.get("upper"))
        assert abs(lo - units.URDF_LIMITS[joint][0]) <= 1e-6, (joint, "lower", lo)
        assert abs(hi - units.URDF_LIMITS[joint][1]) <= 1e-6, (joint, "upper", hi)


def test_gripper_laws():
    lo, hi = units.CALIBRATION["gripper"]
    assert abs(units.K["gripper"] - (hi - lo) / 100.0 * 2.0 * math.pi / 4095.0) <= 1e-15
    assert abs(units.closed_at_0_rad_from_tick(lo) - units.GRIPPER_CLOSED_STOP_RAD) <= 1e-12
    for n in np.linspace(0.0, 100.0, 41):
        a = units.closed_at_0_rad_from_tick(units.gripper_tick_from_norm(n))
        b = units.gripper_norm_to_rad(n, "closed_at_0")
        assert abs(a - b) <= 1e-12, (n, a, b)
        for law in units.GRIPPER_LAWS:
            q = units.gripper_norm_to_rad(n, law)
            assert abs(units.gripper_rad_to_norm(q, law) - n) <= 1e-9, (law, n)


def test_meshes_present_and_not_lfs_pointers():
    root = ET.parse(OURS).getroot()
    files = sorted({el.get("file") for el in root.iter("mesh") if el.get("file")})
    assert files, "so101_twin.xml declares no meshes"
    for name in files:
        path = os.path.join(MESH_DIR, name)
        assert os.path.isfile(path), path
        with open(path, "rb") as fh:
            head = fh.read(len(LFS_HEAD))
        assert not head.startswith(LFS_HEAD), f"{name} is a Git LFS pointer"
        assert os.path.getsize(path) > 84, f"{name} is too small to be a binary STL"


def test_g1_fixed_sample_twin_ours():
    j, p, rot = _fixed_sample()
    arm = tcp.ArmModel(OURS)
    pos, rm = tcp.tcp_batch(arm, j)
    dp, da = tcp.pose_errors(pos, rm, p, rot)
    print(f"twin_ours G1 fixed sample n={len(j)} pos_max_mm={dp.max():.3e} ori_max_deg={da.max():.3e}")
    assert dp.max() < THRESH_MM, float(dp.max())


if __name__ == "__main__":
    failures = 0
    for name in sorted(k for k in list(globals()) if k.startswith("test_")):
        try:
            globals()[name]()
            print("PASS", name)
        except Exception as exc:  # report every failing test, then exit nonzero
            failures += 1
            print("FAIL", name, type(exc).__name__, exc)
    print("failures:", failures)
    sys.exit(1 if failures else 0)
