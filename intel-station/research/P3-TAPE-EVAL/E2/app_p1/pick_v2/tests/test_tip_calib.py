"""Tests for the tip-visible calibration routine (fake bus and synthetic camera only; no port, no camera)."""

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import pytest

from pick_v2 import fake_bus as fb
from pick_v2 import tip_calib as tc
from pick_v2.kinematics import Chain, FollowerCalibration, tcp_xyz
from pick_v2.params import HOME_NORM
from pick_v2.table_calib import apply_homography

URDF = Path(os.environ.get(
    "SO101_URDF",
    "/home/claude/kb/repos/edge-developer-kit-reference-scripts/samples/ai/edge-ai-demo-studio/"
    "workers/services/robotics-ai/platforms/so101/urdf/so101.urdf"))
ROBOTS = Path(os.environ.get("SO101_ROBOTS", "/home/claude/kb/shared/studio-api/project_robots.json"))
MOTION = Path(os.environ.get("SO101_MOTION_DIR", "/home/claude/kb/candidates/code/motion"))


@pytest.fixture(scope="module")
def chain_cal():
    return Chain(URDF), FollowerCalibration.from_studio_json(ROBOTS, "SO101_Follower")


def _args(tmp_path, n_poses):
    return argparse.Namespace(urdf=str(URDF), robots=str(ROBOTS), robot_name=tc.ROBOT_NAME,
                              motion_dir=str(MOTION), out=str(tmp_path), speed=15.0, n_poses=n_poses,
                              plate=[], anchors=None)


def test_speed_and_tolerance_guards():
    with pytest.raises(ValueError):
        tc.GridConfig(speed_units_s=25.0)
    with pytest.raises(ValueError):
        tc.GridConfig(speed_units_s=10.0)
    with pytest.raises(ValueError):
        tc.GridConfig(tolerance_units=3.0)
    with pytest.raises(ValueError):
        tc.GridConfig(loo_gate_mm=12.0)
    assert tc.GridConfig(speed_units_s=15.0).speed_units_s == 15.0
    assert tc.GridConfig(speed_units_s=20.0).speed_units_s == 20.0


def test_poses_one_height_and_keepouts(chain_cal):
    chain, cal = chain_cal
    cfg = tc.GridConfig()
    poses = tc.plan_poses(chain, cal, cfg, plates=fb.DRY_PLATES)
    assert 20 <= len(poses) <= cfg.n_poses
    z_target = tc.TABLE_Z_BASE_M + cfg.z_above_table_m
    for p in poses:
        assert abs(tcp_xyz(chain, cal, np.array(p.n_body))[2] - z_target) < 0.002
        assert math.hypot(p.xy[0] - tc.PAN_AXIS_XY[0], p.xy[1] - tc.PAN_AXIS_XY[1]) >= tc.BASE_KEEPOUT_M
        for px, py, pr in fb.DRY_PLATES:
            assert math.hypot(p.xy[0] - px, p.xy[1] - py) >= pr + cfg.plate_margin_m


def test_detect_tip_single_blob_and_ambiguous_pair():
    img = np.full((120, 160, 3), 70, np.uint8)
    img[40:50, 60:70] = (230, 25, 25)
    u, v = tc.detect_tip(img)
    assert abs(u - 64.5) < 0.6 and abs(v - 44.5) < 0.6
    img[90:96, 120:126] = (230, 25, 25)  # second blob at 0.36 of the first: ambiguous
    assert tc.detect_tip(img) is None


def test_gate_passes_clean_fit_and_fails_on_outlier(chain_cal):
    chain, cal = chain_cal
    cfg = tc.GridConfig(n_poses=24)
    poses = tc.plan_poses(chain, cal, cfg, plates=fb.DRY_PLATES)
    rng = np.random.default_rng(1)
    samples = []
    for p in poses:
        n = np.array(p.n_body)
        xy = tc._tip_xy(chain, cal, n[None], fb.B_TRUE)[0]
        uv = apply_homography(fb.H_TRUE, xy[None])[0] + rng.normal(0.0, 0.5, 2)
        samples.append(tc.Sample(p.index, list(p.xy), list(p.n_body), n.tolist(),
                                 [float(uv[0]), float(uv[1])], 0.0, "ok"))
    clean = tc.calibrate(chain, cal, samples, cfg=cfg)
    assert clean["status"] == "PASS"
    assert clean["loo_rms_mm"] < 5.0
    samples[len(samples) // 2].uv = [samples[len(samples) // 2].uv[0] + 80.0, samples[len(samples) // 2].uv[1]]
    assert tc.calibrate(chain, cal, samples, cfg=cfg)["status"] == "FAIL"


def test_dry_run_gate_and_anchor_identifiability(tmp_path, monkeypatch, chain_cal):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    out = tc.dry_run(_args(tmp_path, 24), tc.GridConfig(n_poses=24))
    res = out["results"]
    assert res["tip_only"]["status"] == "PASS"
    assert res["plate_anchors"]["status"] == "PASS"
    assert abs(res["plate_anchors"]["offsets_normalized_units"]["shoulder_pan"] - fb.B_TRUE[0]) < 0.6
    assert res["plate_anchors"]["identifiability"]["shoulder_pan"]["identifiable"]
    assert not res["tip_only"]["identifiability"]["shoulder_pan"]["identifiable"]
    assert Path(out["path"]).exists()


def test_dry_run_never_opens_the_follower(tmp_path, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    motion = tc.import_motion_layer(MOTION)

    def _forbidden(*args, **kwargs):
        raise AssertionError("dry run must not open the follower port")

    monkeypatch.setattr(motion, "open_follower_held", _forbidden)
    out = tc.dry_run(_args(tmp_path, 20), tc.GridConfig(n_poses=20))
    assert out["results"]["tip_only"]["usable"] >= 20


def test_biased_read_back_drops_every_pose_and_fails(monkeypatch, chain_cal):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    chain, cal = chain_cal
    motion = tc.import_motion_layer(MOTION)
    cfg = tc.GridConfig(n_poses=20)
    follower = fb.FakeFollower(np.concatenate([np.asarray(HOME_NORM, dtype=float), [30.0]]),
                               read_bias=[8.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    cam = fb.FakeTopCamera(follower, chain, cal, fb.H_TRUE, fb.B_TRUE)
    player = motion.MotionPlayer(follower, max_speed=cfg.speed_units_s, tolerance=tc.TOLERANCE_UNITS)
    _, samples = tc.run_session(mode="test", chain=chain, cal=cal, player=player, capture_rgb=cam,
                                plates=fb.DRY_PLATES, cfg=cfg, log=lambda _msg: None)
    assert samples and all(s.status == "not_arrived" for s in samples)
    assert tc.calibrate(chain, cal, samples, cfg=cfg)["status"] == "FAILED"


def test_live_refuses_without_confirm_flag(tmp_path):
    with pytest.raises(SystemExit):
        tc.main(["live", "--urdf", str(URDF), "--out", str(tmp_path), "--plate", "0.10,0.14,0.06"])


def _live_args(tmp_path):
    args = _args(tmp_path, 20)
    args.camera_serial = tc.TOP_CAMERA_SERIAL
    return args


def _needs_follower_api(motion):
    if not hasattr(motion, "follower_serial_port"):
        pytest.skip("motion layer snapshot has no follower_serial_port; the laptop copy runs this test")


def test_live_uses_by_id_port_and_stops_before_motion(tmp_path, monkeypatch):
    motion = tc.import_motion_layer(MOTION)
    _needs_follower_api(motion)
    seen = {"camera_stopped": False}
    monkeypatch.setattr(motion, "port_serial_number", lambda port: tc.PORT_SERIAL)  # host-independent, no device I/O
    monkeypatch.setattr(tc, "realsense_top_capture",
                        lambda serial: (lambda: None, lambda: seen.__setitem__("camera_stopped", True)))
    monkeypatch.setattr("builtins.input", lambda prompt="": "YES")

    def _record_open(port, calibration, **kwargs):
        seen["port"] = port
        raise RuntimeError("test stop before motion")

    monkeypatch.setattr(motion, "open_follower_held", _record_open)
    with pytest.raises(RuntimeError):
        tc.live(_live_args(tmp_path), tc.GridConfig(n_poses=20), plates=fb.DRY_PLATES, anchors=([], []))
    assert seen["port"] == motion.FOLLOWER_PORT_BY_ID
    assert seen["camera_stopped"]


def test_live_without_yes_opens_nothing(tmp_path, monkeypatch):
    motion = tc.import_motion_layer(MOTION)
    _needs_follower_api(motion)
    opened = []
    monkeypatch.setattr(motion, "port_serial_number", lambda port: tc.PORT_SERIAL)
    monkeypatch.setattr(tc, "realsense_top_capture", lambda serial: (lambda: None, lambda: None))
    monkeypatch.setattr("builtins.input", lambda prompt="": "no")
    monkeypatch.setattr(motion, "open_follower_held", lambda *a, **k: opened.append(a))
    with pytest.raises(SystemExit):
        tc.live(_live_args(tmp_path), tc.GridConfig(n_poses=20), plates=fb.DRY_PLATES, anchors=([], []))
    assert opened == []
