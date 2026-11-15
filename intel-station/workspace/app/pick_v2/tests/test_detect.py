"""The numpy red mask must agree with OpenCV's HSV thresholds (cross-check; skipped where cv2 is absent)."""

import numpy as np
import pytest

from pick_v2 import tip_calib as tc


def test_numpy_red_mask_matches_opencv_hsv():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    # reddish patch so that the hue, saturation and value edges are exercised
    img[:60, :80, 0] = rng.integers(150, 256, size=(60, 80))
    img[:60, :80, 1] = rng.integers(0, 90, size=(60, 80))
    img[:60, :80, 2] = rng.integers(0, 90, size=(60, 80))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = hsv[..., 0]
    ref = ((h <= 8) | (h >= 170)) & (hsv[..., 1] >= 120) & (hsv[..., 2] >= 80)
    mine = tc._red_mask(img)
    disagree = float(np.mean(ref != mine))
    assert disagree < 0.005, f"{disagree:.4%} of pixels disagree with OpenCV"
