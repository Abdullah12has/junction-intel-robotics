"""Tests of the live adapters that need no device: the DET2 scan wrapper and the deep-check crops. The DET2 module is a fake
here, so nothing opens a camera, a port or a Studio session."""

from __future__ import annotations

import unittest

import numpy as np

from loop.policy_live import DetScanner, InspectorDeep


class HandsPresent(RuntimeError):
    """The fake of the DET2 module's HandsPresent exception."""


class _Module:
    """A fake DET2 module. scan(top_bgr, close_bgr) returns the brick records of both views, bbox as [x, y, w, h]."""

    HandsPresent = HandsPresent

    def __init__(self, *, hands: bool = False, broken: bool = False) -> None:
        self.hands = hands
        self.broken = broken
        self.seen: list = []

    def scan(self, top, close=None):  # noqa: ANN001 - fake signature
        self.seen.append(top)
        if self.broken:
            raise ValueError("bad frame")
        if self.hands:
            raise HandsPresent("hands_present: 1 skin blob(s)")
        return [
            {"id": "b00_top", "view": "top", "colour": "red", "bbox": [10, 20, 30, 40], "centre_px": [25.0, 40.0]},
            {"id": "b01_close", "view": "close", "colour": "red", "bbox": [1, 2, 3, 4], "centre_px": [2.0, 3.0]},
        ]


class _Inspector:
    """A fake DET2 Inspector: records each crop it is given and returns it, as the module does (BGR)."""

    def __init__(self) -> None:
        self.calls: list = []

    def inspect_rgb(self, crop, camera_key=None):  # noqa: ANN001 - fake signature
        self.calls.append((crop.shape, camera_key, crop[0, 0].tolist()))
        return {"label": "GOOD", "score": 0.1, "threshold": 0.5, "reasons": [], "device": "CPU", "crop": crop}


def _red_rgb(height: int, width: int) -> np.ndarray:
    image = np.zeros((height, width, 3), np.uint8)
    image[..., 0] = 255  # red in RGB
    return image


class DetScannerTests(unittest.TestCase):
    def test_normal_return_clears_the_top_view_and_leaves_the_close_hand_unknown(self) -> None:
        report = DetScanner(_Module()).scan(_red_rgb(4, 4), _red_rgb(4, 4))
        self.assertIs(report.hand_top, False)
        self.assertIsNone(report.hand_close)  # the module has no close-view hand check, so motion is refused
        self.assertEqual([brick["id"] for brick in report.bricks], ["b00_top", "b01_close"])

    def test_bbox_becomes_xyxy_in_the_view_it_belongs_to(self) -> None:
        top, close = DetScanner(_Module()).scan(_red_rgb(4, 4), _red_rgb(4, 4)).bricks
        self.assertEqual(top["top_bbox"], [10, 20, 40, 60])
        self.assertIsNone(top["close_bbox"])
        self.assertEqual(close["close_bbox"], [1, 2, 4, 6])
        self.assertIsNone(close["top_bbox"])

    def test_frames_reach_the_module_in_bgr_order(self) -> None:
        module = _Module()
        DetScanner(module).scan(_red_rgb(2, 2), _red_rgb(2, 2))
        self.assertEqual(module.seen[0][0, 0].tolist(), [0, 0, 255])  # red is the last channel in BGR

    def test_hands_in_the_top_view_refuse_with_no_bricks(self) -> None:
        report = DetScanner(_Module(hands=True)).scan(_red_rgb(2, 2), _red_rgb(2, 2))
        self.assertIs(report.hand_top, True)
        self.assertIsNone(report.hand_close)
        self.assertEqual(report.bricks, ())

    def test_other_module_errors_are_not_swallowed(self) -> None:
        with self.assertRaises(ValueError):
            DetScanner(_Module(broken=True)).scan(_red_rgb(2, 2), _red_rgb(2, 2))


class InspectorDeepTests(unittest.TestCase):
    def test_crops_come_from_the_matched_boxes_in_bgr_and_return_as_rgb(self) -> None:
        inspector = _Inspector()
        deep = InspectorDeep(inspector, "realsense 243622061421", "realsense 243722064316")
        target = {"top_bbox": [10, 20, 30, 40], "close_bbox": None}
        result = deep.check(_red_rgb(100, 100), _red_rgb(100, 100), target)
        self.assertEqual(len(inspector.calls), 1)  # the close view has no box, so the inspector is not called for it
        shape, camera_key, first_pixel = inspector.calls[0]
        self.assertEqual(shape, (20, 20, 3))
        self.assertEqual(camera_key, "realsense 243622061421")
        self.assertEqual(first_pixel, [0, 0, 255])
        self.assertEqual(result["top"]["label"], "GOOD")
        self.assertEqual(result["top"]["crop"][0, 0].tolist(), [255, 0, 0])  # back to RGB for the saved crop
        self.assertEqual(result["close"]["label"], "UNCERTAIN")
        self.assertEqual(result["close"]["reasons"], ["no_brick"])


if __name__ == "__main__":
    unittest.main()
