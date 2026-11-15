"""Tests of the loop policy mode (LOOP-CONTRACT section 8): triage and routing, the ACT and HOME monitors, the run events,
the operator control, the config checks, the live gate and every dry scenario.

Every external part is a fake (loop.policy_fakes). Nothing opens a robot, a camera, a serial port or a Studio session.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from loop import triage as T
from loop.events import EVENT_TYPES, ControlFile, RunDir, json_safe
from loop.grip_monitor import ActMonitor, HomeStill
from loop.policy_fakes import HOME_REFERENCE, HOME_STATE, SCENARIOS, TRAJECTORIES, run_dry, trajectory_state
from loop.policy_run import main as policy_main

PKG_ROOT = Path(__file__).resolve().parents[1]  # the code copy root; the package is app/loop
LOOP_DIR = PKG_ROOT / "app" / "loop"


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-strict JSON constant {token}")


class TriageTests(unittest.TestCase):
    def test_confident_defect_goes_straight_to_green(self) -> None:
        route, reasons = T.triage({"verdict": "CONFIDENT_DEFECT", "reasons": ["shape:not_2x2"]})
        self.assertEqual(route, T.GREEN_DIRECT)
        self.assertIn("confident_defect", reasons)

    def test_every_other_verdict_is_inspected(self) -> None:
        for verdict in ("GOOD", "UNCERTAIN", "DEFECTIVE", ""):
            with self.subTest(verdict=verdict):
                self.assertEqual(T.triage({"verdict": verdict})[0], T.INSPECT)

    def test_find_target_prefers_red_then_most_isolated(self) -> None:
        red_a = {"id": "r1", "colour": "red", "centre_px": [100, 100]}
        red_b = {"id": "r2", "colour": "red", "centre_px": [500, 400]}
        blue = {"id": "b1", "colour": "blue", "centre_px": [520, 380]}
        brick, how = T.find_target([red_a, red_b, blue])
        self.assertEqual((brick["id"], how), ("r1", "red"))
        green = {"id": "g1", "colour": "green", "centre_px": [10, 10]}
        brick, how = T.find_target([blue, green])
        self.assertEqual((brick["id"], how), ("b1", "most_isolated"))
        self.assertEqual(T.find_target([])[1], "none")

    def test_only_inspected_good_reaches_blue(self) -> None:
        cases = [
            (T.GREEN_DIRECT, None, T.GREEN, "direct"),
            (T.INSPECT, T.GOOD, T.BLUE, "inspected"),
            (T.INSPECT, T.DEFECTIVE, T.GREEN, "inspected"),
            (T.INSPECT, T.UNCERTAIN, T.GREEN, "inspected"),
            (T.INSPECT, None, T.GREEN, "fail_safe"),
        ]
        for route, label, plate, via in cases:
            with self.subTest(route=route, label=label):
                decision = T.decide("b1", route, label)
                self.assertEqual((decision["plate"], decision["via"], decision["rule"]), (plate, via, "OD-10"))

    def test_fuse_is_fail_safe(self) -> None:
        good = {"label": "GOOD", "score": 0.1, "threshold": 0.5, "reasons": []}
        defect = {"label": "DEFECTIVE", "score": 0.9, "threshold": 0.5, "reasons": ["tape/appearance"]}
        unsure = {"label": "UNCERTAIN", "score": 0.5, "threshold": 0.5, "reasons": []}
        near = {"label": "GOOD", "score": 0.47, "threshold": 0.5, "reasons": []}
        no_brick = {"label": "GOOD", "score": 0.1, "threshold": 0.5, "reasons": ["no_brick"]}
        self.assertEqual(T.fuse({"top": good, "close": good})[0], T.GOOD)
        self.assertEqual(T.fuse({"top": good, "close": defect})[0], T.DEFECTIVE)
        self.assertEqual(T.fuse({"top": good, "close": unsure})[0], T.UNCERTAIN)
        self.assertEqual(T.fuse({"top": near, "close": good})[0], T.UNCERTAIN)
        self.assertEqual(T.fuse({"top": good, "close": no_brick})[0], T.UNCERTAIN)
        self.assertEqual(T.fuse({})[0], T.UNCERTAIN)

    def test_release_side_reads_the_pan(self) -> None:
        self.assertEqual(T.release_side(32.1), T.BLUE)
        self.assertEqual(T.release_side(-54.7), T.GREEN)
        self.assertEqual(T.release_side(-10.0), T.NONE)
        self.assertEqual(T.release_side(None), T.NONE)
        self.assertEqual(T.release_side(float("nan")), T.NONE)
        verify = T.verify_result("b1", T.GREEN, 32.1, "frames/x.jpg")
        self.assertFalse(verify["ok"])
        self.assertEqual(verify["release_side"], T.BLUE)

    def test_hands_and_missing_reports_refuse_motion(self) -> None:
        self.assertEqual(T.hands_in_view(False, False), [])
        self.assertEqual(T.hands_in_view(True, False), ["top"])
        self.assertEqual(T.hands_in_view(False, None), ["close"])

    def test_scripted_skip_only_for_the_picked_scanned_direct_defect(self) -> None:
        self.assertTrue(T.may_skip_inspection(T.GREEN_DIRECT, (300.0, 200.0), (305.0, 198.0)))
        self.assertFalse(T.may_skip_inspection(T.GREEN_DIRECT, (300.0, 200.0), (400.0, 200.0)))
        self.assertFalse(T.may_skip_inspection(T.INSPECT, (300.0, 200.0), (300.0, 200.0)))

    def test_match_target_scales_between_scans(self) -> None:
        still = [{"id": "s1", "centre_px": [600, 400]}, {"id": "s2", "centre_px": [100, 100]}]
        self.assertEqual(T.match_target(still, {"id": "b1", "centre_px": [300, 200]}, scale=2.0)["id"], "s1")
        self.assertIsNone(T.match_target(still, {"id": "b1", "centre_px": [300, 50]}, scale=2.0))


class MonitorTests(unittest.TestCase):
    def _feed(self, monitor: ActMonitor, trajectory: list, seconds: float) -> list[tuple[str, dict]]:
        events: list[tuple[str, dict]] = []
        t = 0.0
        while t <= seconds and not (monitor.done or monitor.timed_out):
            events.extend(monitor.feed(t, trajectory_state(trajectory, t)))
            t += 0.05
        return events

    def test_blue_trace_grasps_carries_releases_returns_and_finishes(self) -> None:
        monitor = ActMonitor(HOME_REFERENCE)
        events = self._feed(monitor, TRAJECTORIES["blue"], 12.0)
        self.assertEqual([name for name, _ in events], ["grasp", "carry", "release", "return", "done"])
        self.assertEqual(T.release_side(monitor.release_pan), T.BLUE)

    def test_green_trace_releases_on_the_green_side(self) -> None:
        monitor = ActMonitor(HOME_REFERENCE)
        self._feed(monitor, TRAJECTORIES["green"], 12.0)
        self.assertEqual(T.release_side(monitor.release_pan), T.GREEN)

    def test_stuck_policy_times_out_after_45_seconds(self) -> None:
        monitor = ActMonitor(HOME_REFERENCE, timeout_s=45.0)
        events = self._feed(monitor, TRAJECTORIES["stuck"], 60.0)
        names = [name for name, _ in events]
        self.assertTrue(monitor.timed_out)
        self.assertIn("timeout", names)
        self.assertNotIn("done", names)

    def test_home_still_needs_one_second_at_home(self) -> None:
        still = HomeStill(HOME_REFERENCE)
        results = [still.feed(i * 0.05, HOME_STATE) for i in range(30)]
        self.assertFalse(results[5])
        self.assertTrue(results[-1])


class EventTests(unittest.TestCase):
    def test_json_safe_turns_nan_and_infinity_into_null(self) -> None:
        self.assertEqual(
            json_safe({"a": float("nan"), "b": [float("inf"), 1.5, (2, 3)]}),
            {"a": None, "b": [None, 1.5, [2, 3]]},
        )

    def test_unknown_event_type_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = RunDir(Path(tmp))
            try:
                with self.assertRaises(ValueError):
                    run.emit("bogus")
            finally:
                run.close()

    def test_event_types_cover_sections_5_and_8(self) -> None:
        for name in ("run_start", "state", "skill_start", "skill_end", "inspection", "stats", "error", "heartbeat",
                     "run_end", "scan", "triage", "decision", "policy_run", "verify"):
            self.assertIn(name, EVENT_TYPES)

    def test_control_file_returns_each_id_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "control.json"
            control = ControlFile(path)
            path.write_text(json.dumps({"cmd": "pause", "id": 1}), encoding="utf-8")
            self.assertEqual(control.poll(), [{"cmd": "pause", "id": 1}])
            self.assertEqual(control.poll(), [])
            path.write_text(json.dumps({"cmd": "resume", "id": 1}), encoding="utf-8")
            self.assertEqual(control.poll(), [])
            path.write_text(json.dumps({"cmd": "resume", "id": 2}), encoding="utf-8")
            self.assertEqual(control.poll(), [{"cmd": "resume", "id": 2}])


class CliTests(unittest.TestCase):
    def test_scenarios_lists_every_dry_scenario(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = policy_main(["scenarios"])
        self.assertEqual(code, 0)
        self.assertEqual(len(out.getvalue().splitlines()), len(SCENARIOS))

    def test_live_is_refused_without_typing_go(self) -> None:
        out = io.StringIO()
        with mock.patch("builtins.input", return_value="no"), contextlib.redirect_stdout(out):
            code = policy_main(["live", str(LOOP_DIR / "policy_rules.yaml")])
        self.assertEqual(code, 2)
        self.assertIn("GO was not typed", out.getvalue())


class ScenarioTests(unittest.TestCase):
    @staticmethod
    def run_case(name: str) -> tuple[dict, list[dict]]:
        with tempfile.TemporaryDirectory() as tmp:
            summary = asyncio.run(run_dry(name, Path(tmp)))
            text = (Path(summary["run_dir"]) / "events.jsonl").read_text(encoding="utf-8")
            events = [json.loads(line, parse_constant=_reject_constant) for line in text.splitlines()]
        return summary, events

    @staticmethod
    def of_type(events: list[dict], event_type: str) -> list[dict]:
        return [event for event in events if event["type"] == event_type]

    def test_every_scenario_ends_with_its_expected_reason_and_strict_events(self) -> None:
        for name, spec in SCENARIOS.items():
            with self.subTest(scenario=name):
                summary, events = self.run_case(name)
                self.assertEqual(summary["reason"], spec.expect)
                self.assertEqual(events[-1]["type"], "run_end")
                for event in events:
                    self.assertIn(event["type"], EVENT_TYPES)
                    for key in ("t", "mono", "type"):
                        self.assertIn(key, event)

    def test_good_2x2_goes_blue_after_still_inspection(self) -> None:
        summary, events = self.run_case("good_2x2_blue")
        self.assertEqual(summary["reason"], "max_cycles")
        decision = self.of_type(events, "decision")[0]
        self.assertEqual((decision["plate"], decision["via"]), ("BLUE", "inspected"))
        self.assertEqual(self.of_type(events, "inspection")[0]["result"], "GOOD")
        self.assertEqual(summary["policy_starts"][0]["model_id"], "DRYRUN-PLATE-BLUE")
        self.assertIn("unload_environment", summary["calls"])
        verify = self.of_type(events, "verify")[0]
        self.assertTrue(verify["ok"])
        self.assertEqual(verify["release_side"], "BLUE")
        self.assertEqual(self.of_type(events, "stats")[-1]["good"], 1)

    def test_taped_2x2_goes_green_after_inspection(self) -> None:
        summary, events = self.run_case("taped_2x2_green")
        decision = self.of_type(events, "decision")[0]
        self.assertEqual((decision["plate"], decision["via"]), ("GREEN", "inspected"))
        self.assertEqual(self.of_type(events, "inspection")[0]["result"], "DEFECTIVE")
        self.assertEqual(summary["policy_starts"][0]["model_id"], "DRYRUN-PLATE-GREEN")
        self.assertTrue(self.of_type(events, "verify")[0]["ok"])

    def test_uncertain_2x2_goes_green(self) -> None:
        _, events = self.run_case("uncertain_2x2_green")
        decision = self.of_type(events, "decision")[0]
        self.assertEqual((decision["plate"], decision["via"]), ("GREEN", "inspected"))
        self.assertEqual(self.of_type(events, "inspection")[0]["result"], "UNCERTAIN")

    def test_2x4_goes_green_with_no_inspection(self) -> None:
        summary, events = self.run_case("2x4_green_direct")
        decision = self.of_type(events, "decision")[0]
        self.assertEqual((decision["plate"], decision["via"]), ("GREEN", "direct"))
        self.assertEqual(self.of_type(events, "triage")[0]["route"], "GREEN_DIRECT")
        self.assertEqual(self.of_type(events, "inspection"), [])
        self.assertEqual(summary["calls"].count("unload_environment"), 1)  # the hand-back only
        self.assertEqual(summary["calls"][-1], "unload_environment")
        self.assertEqual(summary["still_captures"], 0)

    def test_policy_timeout_stops_the_policy_and_keeps_the_arm_held(self) -> None:
        summary, events = self.run_case("policy_timeout")
        self.assertEqual(summary["reason"], "policy_timeout")
        phases = [event["phase"] for event in self.of_type(events, "policy_run")]
        self.assertIn("timeout", phases)
        self.assertEqual(self.of_type(events, "verify"), [])
        self.assertIn("stop", summary["calls"])
        self.assertEqual(summary["calls"].count("unload_environment"), 1)  # the inspection unload only, no hand-back
        self.assertEqual(summary["calls"][-1], "stop")

    def test_wrong_plate_release_is_caught_by_verify(self) -> None:
        summary, events = self.run_case("wrong_plate_release")
        self.assertEqual(summary["reason"], "verify_failed")
        verify = self.of_type(events, "verify")[0]
        self.assertFalse(verify["ok"])
        self.assertEqual((verify["plate_intended"], verify["release_side"]), ("BLUE", "GREEN"))

    def test_hand_in_view_refuses_before_any_motion(self) -> None:
        summary, events = self.run_case("hand_in_scene")
        self.assertEqual(summary["reason"], "hand_in_view")
        self.assertNotIn("start_policy", summary["calls"])
        self.assertNotIn("unload_environment", summary["calls"])
        scan = self.of_type(events, "scan")[0]
        self.assertTrue(scan["hands"]["close"])
        self.assertEqual(self.of_type(events, "error")[0]["where"], "SCAN")

    def test_runtime_error_mid_act_holds_and_ends(self) -> None:
        summary, events = self.run_case("runtime_error")
        self.assertEqual(summary["reason"], "error")
        self.assertIn("ACT", [event["where"] for event in self.of_type(events, "error")])
        self.assertIn("error", [event["phase"] for event in self.of_type(events, "policy_run")])
        self.assertEqual(summary["calls"].count("unload_environment"), 1)  # the inspection unload only, no hand-back
        self.assertEqual(summary["calls"][-1], "stop")

    def test_operator_stop_during_act_holds_then_hands_the_arm_back(self) -> None:
        summary, events = self.run_case("operator_stop")
        self.assertEqual(summary["reason"], "stopped")
        stop = [event for event in self.of_type(events, "control") if event["cmd"] == "stop"][0]
        self.assertTrue(stop["accepted"])
        self.assertIn("unload_environment", summary["calls"])


class ConfigTests(unittest.TestCase):
    def test_home_gripper_is_the_open_value_not_zero(self) -> None:
        text = (LOOP_DIR / "poses.yaml").read_text(encoding="utf-8")
        match = re.search(r"^\s*home:\s*\[([^\]]*)\]", text, re.M)
        self.assertIsNotNone(match)
        gripper = float(match.group(1).split(",")[-1])
        self.assertGreater(gripper, 20.0)
        self.assertLess(gripper, 30.0)

    def test_no_sandbox_paths_in_the_laptop_configs(self) -> None:
        for name in ("policy_rules.yaml", "poses.yaml"):
            with self.subTest(name=name):
                self.assertNotIn("/home/claude", (LOOP_DIR / name).read_text(encoding="utf-8"))

    def test_motion_layer_comes_from_the_laptop_app_motion(self) -> None:
        text = (LOOP_DIR / "policy_rules.yaml").read_text(encoding="utf-8")
        self.assertIn("motion_dir: /home/intel/intel-junction-2026/app/motion", text)

    def test_live_refuses_while_placeholders_and_uncalibrated_inputs_remain(self) -> None:
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML is not installed")
        from loop.policy_live import missing_inputs

        rules = yaml.safe_load((LOOP_DIR / "policy_rules.yaml").read_text(encoding="utf-8"))
        missing = " ".join(missing_inputs(rules))
        self.assertIn("placeholder", missing)
        self.assertNotIn("det2", missing)  # the DET2 E2 module is bound (det2.module_dir is set); the other gates remain
        self.assertIn("exposure", missing)


if __name__ == "__main__":
    unittest.main()
