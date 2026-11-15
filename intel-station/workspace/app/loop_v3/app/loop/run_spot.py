"""Spot cell entry point (loop v3): dry runs with fakes, and the gated live run.

  python -B -m loop.run_spot scenarios
  python -B -m loop.run_spot dry SCENARIO LOGS_ROOT [--no-images]
  python -B -m loop.run_spot live RULES.yaml [--live]

Live refuses unless --live is given, every rule is set (no null ROI), the presence flag exists, no stop file is
present, and GO is typed. Dry runs use fakes only: no camera, no serial port, no Studio, no DET2 model.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from loop import triage as T
from loop.events import ControlFile, RunDir
from loop.spot_cell import Circle, Config, HandsDetected, SegmentResult, SpotCell

SEQ_BLUE = ["spot_pick_from_home", "home_to_blue_carry", "release_blue", "blue_to_home", "teardown"]
SEQ_GREEN = ["spot_pick_from_home", "home_to_green_carry", "release_green", "green_to_home", "teardown"]
SEQ_RETRY = ["spot_pick_from_home", "spot_to_home_open"]
DRY_ROI = {"top": Circle(425.0, 230.0, 45.0), "close": Circle(360.0, 250.0, 60.0),
           "blue": Circle(150.0, 300.0, 90.0), "green": Circle(520.0, 320.0, 90.0)}
PLATE_CENTRE = {"blue": (150.0, 300.0), "green": (520.0, 320.0)}
LIVE_REASONS_OK = ("max_cycles", "stopped", "quit", "empty_spot_timeout", "stop_file")


def brick(name: str, view: str, verdict: str, reasons: tuple[str, ...] = (), centre: tuple[float, float] = (425.0, 232.0),
          colour: str = "white") -> dict[str, Any]:
    return {"id": name, "view": view, "colour": colour, "centre_px": list(centre), "verdict": verdict,
            "reasons": list(reasons), "confident": verdict.startswith("CONFIDENT")}


def scenario(verdict: str | None, expect: dict[str, Any], reasons: tuple[str, ...] = (), grips: tuple[float, ...] = (2.8,),
             **extra: Any) -> dict[str, Any]:
    if verdict is None:
        top: list[dict[str, Any]] = []
        close: list[dict[str, Any]] = []
    else:
        top = [brick("b01_top", "top", verdict, reasons)]
        close = [brick("b01_close", "close", verdict, reasons, (360.0, 250.0))]
    return {"top": top, "close": close, "grips": list(grips), "expect": expect, **extra}


SCENARIOS: dict[str, dict[str, Any]] = {
    "good_2x2_blue": scenario("GOOD", {"reason": "max_cycles", "plate": "BLUE", "via": "inspected", "sequences": SEQ_BLUE}),
    "taped_2x2_green": scenario("CONFIDENT_DEFECT", {"reason": "max_cycles", "plate": "GREEN", "via": "direct",
                                                   "sequences": SEQ_GREEN}, reasons=("tape",)),
    "2x4_green_direct": scenario("CONFIDENT_DEFECT", {"reason": "max_cycles", "plate": "GREEN", "via": "direct",
                                                     "sequences": SEQ_GREEN}, reasons=("shape_2x4",)),
    "uncertain_2x2_green": scenario("UNCERTAIN", {"reason": "max_cycles", "plate": "GREEN", "via": "inspected",
                                                 "sequences": SEQ_GREEN, "inspections": 3}),
    "missed_then_retry": scenario("GOOD", {"reason": "max_cycles", "plate": "BLUE", "via": "inspected",
                                          "sequences": SEQ_RETRY + SEQ_BLUE}, grips=(0.4, 2.8)),
    "missed_twice": scenario("GOOD", {"reason": "grasp_failed", "plate": "BLUE", "via": "inspected",
                                     "sequences": SEQ_RETRY + SEQ_RETRY + ["teardown"]}, grips=(0.4, 0.3)),
    "hand_in_scene": scenario("GOOD", {"reason": "hand_in_view", "sequences": ["teardown"]}, hands=True),
    "stop_mid_cycle": scenario("GOOD", {"reason": "stopped", "held": True,
                                       "sequences": ["spot_pick_from_home", "home_to_blue_carry"]},
                               stop_after="home_to_blue_carry"),
    "empty_spot_timeout": scenario(None, {"reason": "empty_spot_timeout", "sequences": ["teardown"]},
                                   wait_timeout_s=3.0, grips=()),
}


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


class FakeControl:
    def __init__(self, commands: tuple[str, ...] = ("start",)) -> None:
        self.queue: list[dict[str, Any]] = []
        self.next_id = 1
        for name in commands:
            self.push(name)

    def push(self, name: str) -> None:
        self.queue.append({"cmd": name, "id": self.next_id})
        self.next_id += 1

    def poll(self) -> list[dict[str, Any]]:
        queued, self.queue = self.queue, []
        return queued


class FakeWorld:
    """What the fake cameras see. A release moves the spot brick onto the plate it was sorted to."""

    def __init__(self, top: list[dict[str, Any]], close: list[dict[str, Any]]) -> None:
        self.top = list(top)
        self.close = list(close)
        self.plates: list[dict[str, Any]] = []


class FakeCamera:
    def __init__(self, world: FakeWorld) -> None:
        self.world = world

    def top_frame(self) -> np.ndarray:
        return np.zeros((480, 640, 3), np.uint8)

    def stills(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((480, 640, 3), np.uint8), np.zeros((480, 640, 3), np.uint8)


class FakeDetector:
    def __init__(self, world: FakeWorld, hands: bool = False) -> None:
        self.world = world
        self.hands = hands

    def _check(self) -> None:
        if self.hands:
            raise HandsDetected("hands_present: 1 skin blob(s)")

    def scan_top(self, frame: Any) -> list[dict[str, Any]]:
        self._check()
        return list(self.world.top) + list(self.world.plates)

    def scan(self, top: Any, close: Any) -> dict[str, list[dict[str, Any]]]:
        self._check()
        return {"top": list(self.world.top), "close": list(self.world.close)}


class FakeArm:
    def __init__(self, world: FakeWorld, grips: list[float], control: FakeControl, stop_after: str | None) -> None:
        self.world = world
        self.grips = list(grips)
        self.control = control
        self.stop_after = stop_after
        self.calls: list[str] = []

    def run(self, name: str) -> SegmentResult:
        self.calls.append(name)
        if name == self.stop_after:
            self.control.push("stop")
        gripper = self.grips.pop(0) if name == "spot_pick_from_home" and self.grips else None
        if name.startswith("release_"):
            colour = name.split("_", 1)[1]
            self.world.top, self.world.close = [], []
            self.world.plates = [brick(f"b01_{colour}", "top", "GOOD", (), PLATE_CENTRE[colour], colour)]
        return SegmentResult(True, 0, 0.0, gripper)

    def teardown(self) -> SegmentResult:
        self.calls.append("teardown")
        return SegmentResult(True, 0, 0.0, None)


def dry_config(spec: dict[str, Any], write_images: bool) -> Config:
    return Config(top_roi=DRY_ROI["top"], close_roi=DRY_ROI["close"], blue_roi=DRY_ROI["blue"],
                  green_roi=DRY_ROI["green"], max_cycles=int(spec.get("max_cycles", 1)),
                  wait_timeout_s=float(spec.get("wait_timeout_s", 300.0)), write_images=write_images)


def run_dry(name: str, logs_root: str, write_images: bool = True) -> dict[str, Any]:
    spec = SCENARIOS[name]
    clock = FakeClock()
    control = FakeControl(("start",))
    world = FakeWorld(spec["top"], spec["close"])
    arm = FakeArm(world, spec["grips"], control, spec.get("stop_after"))
    run = RunDir(Path(logs_root))
    cell = SpotCell(dry_config(spec, write_images), run, FakeCamera(world), FakeDetector(world, spec.get("hands", False)),
                    arm, control, clock)
    reason = cell.run_loop({"mode": "dry", "scenario": name, "fakes": "camera, detector, arm, control, clock"})
    run.close()
    events = [json.loads(line) for line in (run.path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    summary: dict[str, Any] = {
        "scenario": name, "reason": reason, "held": cell.held, "location": cell.location, "cycles": cell.cycles,
        "plate": cell.plate, "via": cell.via, "sequences": arm.calls, "inspections": run.emitted.get("inspection", 0),
        "errors": [event["message"] for event in events if event["type"] == "error"],
        "emitted": dict(run.emitted), "run_dir": str(run.path), "expect": spec["expect"],
    }
    summary["pass"] = all(summary[key] == value for key, value in spec["expect"].items())
    return summary


class RealClock:
    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class LiveCamera:
    """Stills via ffmpeg from the by-path RealSense colour nodes in the rules. BGR arrays, 640x480 (B3, B9)."""

    def __init__(self, nodes: dict[str, str], scratch: Path) -> None:
        self.nodes = nodes
        self.scratch = scratch
        scratch.mkdir(parents=True, exist_ok=True)

    def grab(self, view: str) -> np.ndarray:
        import cv2

        target = self.scratch / f"{view}.png"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "v4l2", "-video_size", "640x480", "-i",
                        self.nodes[view], "-frames:v", "1", str(target)], check=True, timeout=15)
        frame = cv2.imread(str(target))
        if frame is None:
            raise RuntimeError(f"no frame from {view}")
        return frame

    def top_frame(self) -> np.ndarray:
        return self.grab("top")

    def stills(self) -> tuple[np.ndarray, np.ndarray]:
        return self.grab("top"), self.grab("close")


class LiveDetector:
    """DET2 det2_v1 scan_full, in process, on BGR frames (B9). Its HandsPresent becomes HandsDetected."""

    def __init__(self, det2_dir: str) -> None:
        sys.path.insert(0, det2_dir)
        import inspector  # det2_v1/inspector.py

        self.inspector = inspector

    def scan(self, top: Any, close: Any) -> dict[str, list[dict[str, Any]]]:
        try:
            result = self.inspector.scan_full(top, close)
        except self.inspector.HandsPresent as error:
            raise HandsDetected(str(error)) from error
        bricks = list(result["bricks"])
        return {"top": [b for b in bricks if b.get("view") == "top"],
                "close": [b for b in bricks if b.get("view") == "close"]}

    def scan_top(self, frame: Any) -> list[dict[str, Any]]:
        return self.scan(frame, None)["top"]


def load_rules(path: str) -> dict[str, Any]:
    import yaml  # PyYAML: Opus adds it to data-env (B8)

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def circle(value: Any) -> Circle | None:
    return None if value is None else Circle(float(value["cx"]), float(value["cy"]), float(value["r"]))


def config_from_rules(rules: dict[str, Any]) -> Config:
    roi, grasp, timing = rules["roi"], rules["grasp"], rules["timing"]
    return Config(top_roi=circle(roi["top"]), close_roi=circle(roi["close"]), blue_roi=circle(roi["blue_plate"]),
                  green_roi=circle(roi["green_plate"]), hold_min=float(grasp["hold_min"]),
                  hold_max=float(grasp["hold_max"]), missed_below=float(grasp["missed_below"]),
                  grasp_retries=int(grasp["retries"]), stable_s=float(timing["stable_s"]),
                  wait_timeout_s=float(timing["wait_timeout_s"]), wait_poll_s=float(timing["wait_poll_s"]),
                  live_period_s=float(timing["live_period_s"]), inspect_pairs=int(timing["inspect_pairs"]),
                  inspect_gap_s=float(timing["inspect_gap_s"]), max_cycles=int(timing["max_cycles"]),
                  stop_file=rules["paths"]["stop_file"], write_images=True)


def gate_problems(rules: dict[str, Any], live_flag: bool) -> list[str]:
    problems = [] if live_flag else ["--live not given"]
    for key in ("close", "blue_plate", "green_plate"):
        if rules["roi"].get(key) is None:
            problems.append(f"roi.{key} is null (not measured yet)")
    if not Path(rules["paths"]["presence_flag"]).exists():
        problems.append("presence flag missing (Hassan not present)")
    if Path(rules["paths"]["stop_file"]).exists():
        problems.append("stop file present")
    return problems


def run_live(rules_path: str, live_flag: bool) -> int:
    from loop.spot_motion import SubprocessArm

    rules = load_rules(rules_path)
    problems = gate_problems(rules, live_flag)
    if problems:
        print(json.dumps({"reason": "refused", "message": problems}), flush=True)
        return 2
    print("type GO to start the live spot run:", flush=True)
    if sys.stdin.readline().strip() != "GO":
        print(json.dumps({"reason": "refused", "message": "GO was not typed"}), flush=True)
        return 2
    paths = rules["paths"]
    run = RunDir(Path(paths["logs_root"]))
    arm = SubprocessArm(paths["motion_python"], Path(__file__).with_name("spot_motion.py"), paths["poses"],
                        paths["studio_export"], paths["motion_dir"])
    cell = SpotCell(config_from_rules(rules), run, LiveCamera(rules["cameras"], run.path / "scratch"),
                    LiveDetector(paths["det2_dir"]), arm, ControlFile(run.path / "control.json"), RealClock())
    reason = cell.run_loop({"mode": "spot", "live_source": "realsense", "cameras": rules["cameras"],
                            "detector": {"backend": "det2_v1", "device": "CPU"}, "skills": {},
                            "telemetry": {"path": paths["telemetry_dir"]}})
    run.close()
    print(json.dumps({"reason": reason, "held": cell.held, "location": cell.location, "run_dir": str(run.path)}),
          flush=True)
    return 0 if reason in LIVE_REASONS_OK and not cell.held else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Spot cell loop v3")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("scenarios")
    dry = commands.add_parser("dry")
    dry.add_argument("scenario")
    dry.add_argument("logs_root")
    dry.add_argument("--no-images", action="store_true")
    live = commands.add_parser("live")
    live.add_argument("rules")
    live.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "scenarios":
        print("\n".join(SCENARIOS))
        return 0
    if args.command == "dry":
        summary = run_dry(args.scenario, args.logs_root, write_images=not args.no_images)
        print(json.dumps(summary, separators=(",", ":")))
        return 0 if summary["pass"] else 1
    return run_live(args.rules, args.live)


if __name__ == "__main__":
    raise SystemExit(main())
