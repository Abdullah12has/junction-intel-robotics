"""Dry-run scenarios and unit checks for the spot cell. Fakes only: no camera, serial port, Studio or DET2 model."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from loop import spot_cell as C
from loop import triage as T
from loop import run_spot as R


class DryScenarios(unittest.TestCase):
    def test_every_scenario_meets_its_expectation(self) -> None:
        for name in R.SCENARIOS:
            with self.subTest(scenario=name), tempfile.TemporaryDirectory() as tmp:
                summary = R.run_dry(name, str(Path(tmp) / "logs"), write_images=False)
                self.assertTrue(summary["pass"], summary)


class Pairing(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = C.Config(top_roi=C.Circle(425.0, 230.0, 45.0), close_roi=C.Circle(360.0, 250.0, 60.0))

    def test_bricks_outside_the_spot_roi_are_ignored(self) -> None:
        bricks = [{"id": "plate", "centre_px": [150, 300]}, {"id": "cluster", "centre_px": [471.8, 204.0]}]
        self.assertIsNone(C.spot_brick(bricks, self.cfg.top_roi))

    def test_confident_defect_from_either_view_wins(self) -> None:
        top = [{"id": "t", "centre_px": [425, 232], "verdict": "GOOD", "reasons": []}]
        close = [{"id": "c", "centre_px": [360, 250], "verdict": "CONFIDENT_DEFECT", "reasons": ["tape"]}]
        target = C.choose_target(top, close, self.cfg)
        self.assertEqual(target["verdict"], "CONFIDENT_DEFECT")
        self.assertEqual(T.triage(target)[0], T.GREEN_DIRECT)

    def test_missing_close_view_is_uncertain_and_inspected(self) -> None:
        top = [{"id": "t", "centre_px": [425, 232], "verdict": "GOOD", "reasons": []}]
        target = C.choose_target(top, [], self.cfg)
        self.assertEqual(target["verdict"], T.UNCERTAIN)
        self.assertEqual(T.triage(target)[0], T.INSPECT)


class Vote(unittest.TestCase):
    def test_majority_needs_two_matching_pairs(self) -> None:
        self.assertEqual(C.majority([T.GOOD, T.GOOD, T.DEFECTIVE]), T.GOOD)
        self.assertEqual(C.majority([T.GOOD, T.DEFECTIVE, T.UNCERTAIN]), T.UNCERTAIN)
        self.assertEqual(C.majority([T.UNCERTAIN, T.UNCERTAIN, T.GOOD]), T.UNCERTAIN)


class SubprocessBoundary(unittest.TestCase):
    def test_reads_last_json_line_and_exit_code(self) -> None:
        from loop.spot_motion import SubprocessArm

        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.py"
            good.write_text('import json\nprint("noise")\nprint(json.dumps({"ok": True, "latency_ms": 12.5, "gripper": 2.8}))\n',
                            encoding="utf-8")
            bad = Path(tmp) / "bad.py"
            bad.write_text('import sys\nprint("stalled", file=sys.stderr)\nsys.exit(2)\n', encoding="utf-8")
            seg = SubprocessArm(sys.executable, good, "p", "e", "m").run("spot_pick_from_home")
            self.assertTrue(seg.ok)
            self.assertEqual(seg.gripper, 2.8)
            failed = SubprocessArm(sys.executable, bad, "p", "e", "m").teardown()
            self.assertFalse(failed.ok)
            self.assertEqual(failed.exit_code, 2)


if __name__ == "__main__":
    unittest.main()
