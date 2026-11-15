"""Tests for the sim2 geometry priors: loading, keep-out discs and the prior check (synthetic geometry file)."""

import json

import numpy as np

from pick_v2 import fake_bus as fb
from pick_v2 import tip_calib as tc
from pick_v2.table_calib import apply_homography


def _write_geometry(path, green_xy, green_uv):
    g = {
        "frame": "test frame (MJCF base)",
        "plates_in_robot_base_frame_m": {
            "blue": {"centre_xy_m": [0.1857, -0.1099], "n": 3, "sd_xy_m": [0.0044, 0.0057, 0.008]},
            "green": {"centre_xy_m": list(green_xy), "n": 4, "sd_xy_m": [0.0157, 0.0138, 0.0214]},
        },
        "uncertainty_horizontal_m": {"budget_1sigma_m": 0.032},
        "top_view_pixels_640x480": {"green_plate_real_top_camera": list(green_uv)},
    }
    path.write_text(json.dumps(g), encoding="utf-8")
    return path


def test_priors_load_and_keepout_discs(tmp_path):
    pri = tc.load_priors(str(_write_geometry(tmp_path / "g.json", (0.2095, 0.2134), (578.6, 181.1))))
    assert pri["budget_m"] == 0.032
    assert pri["plates"]["blue"]["xy_m"] == [0.1857, -0.1099]
    discs = tc.plates_from_priors(pri, 0.06)
    assert len(discs) == 2
    assert abs(discs[0][2] - 0.092) < 1e-9


def test_prior_check_zero_when_prior_matches_homography(tmp_path):
    uv = np.array([[578.6, 181.1]])
    xy_true = apply_homography(np.linalg.inv(fb.H_TRUE), uv)[0]
    pri = tc.load_priors(str(_write_geometry(tmp_path / "match.json", tuple(xy_true), (578.6, 181.1))))
    chk = tc.prior_check(fb.H_TRUE, pri)
    assert chk["difference_mm"] < 1e-6
    assert chk["within_budget"]


def test_prior_check_flags_a_prior_five_centimetres_off(tmp_path):
    uv = np.array([[578.6, 181.1]])
    xy_true = apply_homography(np.linalg.inv(fb.H_TRUE), uv)[0]
    far = (xy_true[0] + 0.05, xy_true[1])
    pri = tc.load_priors(str(_write_geometry(tmp_path / "far.json", far, (578.6, 181.1))))
    chk = tc.prior_check(fb.H_TRUE, pri)
    assert abs(chk["difference_mm"] - 50.0) < 1e-3
    assert not chk["within_budget"]
