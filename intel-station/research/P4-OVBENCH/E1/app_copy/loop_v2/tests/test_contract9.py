"""LOOP-CONTRACT section 9 and the D-080 rules on the policy events: every event carries cycle and t, run_start.config has the
dashboard shape with the GPU named, policy_run and skill_start carry backend and device, inspect_cameras lists both cameras,
policy_run and inspection carry latency_ms, and a policy timeout holds and returns home without unloading. Dry runs only."""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
import unittest
from pathlib import Path

from loop.policy_cycle import PlateSkill
from loop.policy_fakes import SCENARIOS, run_dry
from loop.policy_live import config_from_rules

ISO_MS_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")

RULES = {
    "paths": {"telemetry_dir": "/tmp/loop-v2-test/telemetry"},
    "studio": {"hz": 30, "top_key": "realsense 243622061421", "close_key": "realsense 243722064316"},
    "plates": {
        "BLUE": {"skill": "S2", "model_id": "MODEL-BLUE", "task": "move the red brick to the blue plate",
                 "backend": "openvino", "device": "GPU", "mode": "sync", "request_threshold": 0.5, "grace_s": 3},
        "GREEN": {"skill": "S3", "model_id": "MODEL-GREEN", "task": "move the red brick to the green plate",
                  "backend": "openvino", "device": "GPU", "mode": "sync", "request_threshold": 0.5, "grace_s": 3},
    },
    "home_reference": {"shoulder_lift.pos": -98.5, "elbow_flex.pos": 96.0, "wrist_flex.pos": 62.0},
    "timing": {"home_tol": 10.0, "still_s": 1.0, "motion_eps": 0.5, "grasp_below": 4.0, "grasp_hold_s": 0.3,
               "release_at": 6.0, "act_timeout_s": 45.0, "home_wait_s": 5.0, "scan_retries": 2, "scan_wait_s": 1.0,
               "verify_settle_s": 0.5, "frames_ready_s": 10.0, "max_cycles": 10},
    "triage": {"uncertain_margin": 0.1},
    "detector": {"backend": "det2_e2", "device": "CPU", "model_dir": None},
}


def _run(name: str) -> tuple[dict, list[dict]]:
    """Runs one dry scenario in a temporary folder; returns its summary and its parsed events."""
    with tempfile.TemporaryDirectory() as tmp:
        summary = asyncio.run(run_dry(name, Path(tmp)))
        text = (Path(summary["run_dir"]) / "events.jsonl").read_text(encoding="utf-8")
    return summary, [json.loads(line) for line in text.splitlines()]


class Section9Tests(unittest.TestCase):
    def test_every_event_of_every_scenario_carries_cycle_and_utc_t(self) -> None:
        for name in sorted(SCENARIOS):
            with self.subTest(scenario=name):
                _, events = _run(name)
                self.assertEqual(events[0]["type"], "run_start")
                self.assertEqual(events[0]["cycle"], 0)
                cycles = [event["cycle"] for event in events]
                self.assertEqual(cycles, sorted(cycles))  # the cycle never goes back
                for event in events:
                    self.assertIsInstance(event["cycle"], int)
                    self.assertRegex(event["t"], ISO_MS_Z)

    def test_run_start_config_names_the_gpu_for_every_skill(self) -> None:
        _, events = _run("good_2x2_blue")
        config = events[0]["config"]
        self.assertEqual(set(config), {"mode", "live_source", "detector", "skills", "telemetry"})
        self.assertEqual(config["mode"], "policy")
        self.assertEqual(config["live_source"], "fake")
        self.assertEqual(config["detector"], {"backend": "fake", "device": "none"})
        self.assertEqual(set(config["skills"]), {"S2", "S3"})
        for skill in config["skills"].values():
            self.assertEqual((skill["backend"], skill["device"]), ("openvino", "GPU"))
        self.assertEqual(config["telemetry"], {"path": None})

    def test_policy_run_and_skill_start_name_backend_and_device(self) -> None:
        _, events = _run("good_2x2_blue")
        marked = [event for event in events if event["type"] in ("policy_run", "skill_start")]
        self.assertTrue(marked)
        for event in marked:
            self.assertEqual((event["backend"], event["device"]), ("openvino", "GPU"))

    def test_a_load_that_is_not_openvino_on_the_gpu_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            PlateSkill(skill="S2", plate="BLUE", model_id="M", task="t", device="auto")
        with self.assertRaises(ValueError):
            PlateSkill(skill="S2", plate="BLUE", model_id="M", task="t", backend="cpu")
        self.assertEqual(PlateSkill(skill="S2", plate="BLUE", model_id="M", task="t").device, "GPU")

    def test_live_rules_map_to_the_section_9_config(self) -> None:
        config = config_from_rules(RULES)
        self.assertEqual(config.live_source, "studio")
        self.assertEqual((config.detector_backend, config.detector_device), ("det2_e2", "CPU"))
        self.assertRegex(config.telemetry_path, r"^/tmp/loop-v2-test/telemetry/\d{8}\.jsonl$")
        self.assertEqual({skill.device for skill in config.plates.values()}, {"GPU"})

    def test_inspect_cameras_lists_both_cameras_with_the_section_9_fields(self) -> None:
        _, events = _run("good_2x2_blue")
        inspect = [event for event in events if event["type"] == "inspect_cameras"]
        self.assertEqual(len(inspect), 1)
        self.assertEqual(inspect[0]["cycle"], 1)
        cameras = inspect[0]["cameras"]
        self.assertEqual([camera["camera"] for camera in cameras], ["top", "close"])
        for camera in cameras:
            self.assertEqual(set(camera), {"camera", "label", "score", "threshold", "reasons", "frame", "crop"})
            self.assertEqual(camera["label"], "GOOD")
            self.assertIsNotNone(camera["frame"])
            self.assertIsNone(camera["crop"])  # the dry runs keep no crop images

    def test_policy_run_and_inspection_carry_latency_ms(self) -> None:
        _, events = _run("taped_2x2_green")
        inspection = [event for event in events if event["type"] == "inspection"]
        self.assertEqual(len(inspection), 1)
        self.assertNotIn("cameras", inspection[0])  # the per-camera results live in inspect_cameras
        self.assertIsInstance(inspection[0]["latency_ms"], float)
        phases = [event for event in events if event["type"] == "policy_run"]
        self.assertEqual(phases[0]["phase"], "start")
        for phase in phases:
            self.assertIsInstance(phase["latency_ms"], float)
            self.assertGreaterEqual(phase["latency_ms"], 0.0)

    def test_policy_timeout_holds_then_returns_home_without_unloading(self) -> None:
        summary, events = _run("policy_timeout")
        self.assertEqual(summary["reason"], "policy_timeout")
        calls = summary["calls"]
        after_start = calls[calls.index("start_policy") + 1:]
        self.assertNotIn("unload_environment", after_start)  # torque is kept: no environment unload after the timeout
        self.assertEqual(calls[-1], "stop")  # the policy is stopped and the arm held
        states = [event["state"] for event in events if event["type"] == "state"]
        self.assertEqual(states[-3:], ["RETURN_HOME", "HOME", "STOPPED"])


if __name__ == "__main__":
    unittest.main()
