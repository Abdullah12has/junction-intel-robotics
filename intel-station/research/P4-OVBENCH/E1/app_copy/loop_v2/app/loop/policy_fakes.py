"""Dry-run fakes for loop_v2: a virtual clock, a Studio runtime stand-in, the DET2 scanner and Inspector stand-ins, still
cameras, and the scripted arm trajectories of the dry scenarios. Nothing here touches a robot, a camera or a port."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from loop.policy_cycle import PlateSkill, PolicyConfig, PolicyCycle
from loop.events import RunDir, ScheduledControl
from loop.triage import BLUE, GREEN, ScanReport

TOP_KEY = "realsense 243622061421"  # top view (FACTS-LIVE 13:23Z)
CLOSE_KEY = "realsense 243722064316"  # close low view
HOME_REFERENCE = {"shoulder_lift.pos": -98.5, "elbow_flex.pos": 96.0, "wrist_flex.pos": 62.0}  # rules reference.home_pose
HOME_STATE = {
    "shoulder_pan.pos": -6.6,
    "shoulder_lift.pos": -98.5,
    "elbow_flex.pos": 96.0,
    "wrist_flex.pos": 62.0,
    "wrist_roll.pos": -53.1,
    "gripper.pos": 25.7,
}
ROLL = -53.1


def _pick_and_place(plate_pan: float) -> list[tuple[float, ...]]:
    """Keyframes (t, pan, lift, elbow, wrist_flex, gripper): reach, close at 2.5 s, carry, open at 5 s, return by 8 s."""
    return [
        (0.0, -6.6, -98.5, 96.0, 62.0, 25.7),
        (1.0, -20.0, -60.0, 60.0, 40.0, 25.7),
        (2.0, -20.0, -40.0, 40.0, 40.0, 25.7),
        (2.5, -20.0, -40.0, 40.0, 40.0, 2.5),
        (4.5, plate_pan, -40.0, 60.0, 40.0, 2.5),
        (4.8, plate_pan, -50.0, 40.0, 50.0, 2.5),
        (5.0, plate_pan, -50.0, 40.0, 50.0, 25.7),
        (8.0, -6.6, -98.5, 96.0, 62.0, 25.7),
    ]


TRAJECTORIES: dict[str, list[tuple[float, ...]]] = {
    "blue": _pick_and_place(32.0),  # release pan +32.1 as measured for BLUE (FACTS-LIVE 13:52Z)
    "green": _pick_and_place(-54.0),  # release pans -51.8 to -57.3 as measured for GREEN
    "stuck": [  # grasps, then never releases: the policy times out
        (0.0, -6.6, -98.5, 96.0, 62.0, 25.7),
        (1.0, -20.0, -60.0, 60.0, 40.0, 25.7),
        (2.0, -20.0, -40.0, 40.0, 40.0, 25.7),
        (2.5, -20.0, -40.0, 40.0, 40.0, 2.5),
        (600.0, 0.0, -40.0, 40.0, 40.0, 2.5),
    ],
}


def trajectory_state(keys: Sequence[tuple[float, ...]], t: float) -> dict[str, float]:
    if t <= keys[0][0]:
        values = keys[0][1:]
    elif t >= keys[-1][0]:
        values = keys[-1][1:]
    else:
        values = keys[-1][1:]
        for start, end in zip(keys, keys[1:]):
            if start[0] <= t <= end[0]:
                weight = (t - start[0]) / (end[0] - start[0]) if end[0] > start[0] else 1.0
                values = tuple(a + (b - a) * weight for a, b in zip(start[1:], end[1:]))
                break
    pan, lift, elbow, wrist_flex, gripper = values
    return {
        "shoulder_pan.pos": pan,
        "shoulder_lift.pos": lift,
        "elbow_flex.pos": elbow,
        "wrist_flex.pos": wrist_flex,
        "wrist_roll.pos": ROLL,
        "gripper.pos": gripper,
    }


class VirtualClock:
    """Time moves only when something sleeps, so the dry runs are deterministic and fast."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = float(start)

    def now(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.t += max(0.0, float(seconds))
        await asyncio.sleep(0)


@dataclass(frozen=True)
class FakeFrame:
    key: str
    timestamp: float
    jpeg: bytes

    def rgb(self) -> None:
        return None


@dataclass(frozen=True)
class FakeObservation:
    timestamp: float
    state: dict[str, float]
    frames: dict[str, Any] = field(default_factory=dict)


class FakeRuntime:
    """Stand-in for loop.studio_runtime_client.StudioRuntimeClient (same method names and keyword arguments)."""

    def __init__(self, clock: VirtualClock, *, trajectory: Sequence[tuple[float, ...]] | None = None,
                 fault_at_s: float | None = None, hz: float = 20.0) -> None:
        self.clock = clock
        self.trajectory = trajectory
        self.fault_at_s = fault_at_s
        self.hz = hz
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.policy_active = False
        self.policy_t0 = 0.0
        self.frames = {key: FakeFrame(key, clock.now(), f"JPEG {key}".encode()) for key in (TOP_KEY, CLOSE_KEY)}

    def _record(self, call_name: str, **kwargs: Any) -> None:
        self.calls.append((call_name, kwargs))

    async def load_environment(self, environment: Mapping[str, Any], **_: Any) -> None:
        self._record("load_environment", name=environment.get("name"))

    async def unload_environment(self) -> None:
        self._record("unload_environment")
        self.policy_active = False

    async def wait_for_frames(self, keys: Sequence[str], timeout_s: float = 10.0) -> None:
        self._record("wait_for_frames", keys=list(keys))

    def latest_frame(self, key: str) -> FakeFrame | None:
        return self.frames.get(key)

    async def start_policy(self, model_id: Any, task: str, *, backend: str, device: str = "auto", hz: float = 30.0,
                           mode: str = "sync", request_threshold: float = 0.5, grace_s: float = 3.0) -> None:
        self._record("start_policy", model_id=str(model_id), task=task, backend=backend, device=device, hz=hz,
                     mode=mode, request_threshold=request_threshold, grace_s=grace_s)
        self.policy_active = True
        self.policy_t0 = self.clock.now()

    async def stop(self, *, unload_model: bool = False) -> None:
        self._record("stop")
        self.policy_active = False

    async def observations(self, *, decode: bool = True):  # noqa: ANN201 - async generator
        while True:
            now = self.clock.now()
            state = dict(HOME_STATE)
            if self.policy_active and self.trajectory is not None:
                elapsed = now - self.policy_t0
                if self.fault_at_s is not None and elapsed >= self.fault_at_s:
                    raise RuntimeError("studio runtime session closed (scripted fault)")
                state = trajectory_state(self.trajectory, elapsed)
            yield FakeObservation(timestamp=now, state=state)
            await self.clock.sleep(1.0 / self.hz)


class FakeScanner:
    """Stand-in for the DET2 scan. Reports are consumed in order; the last one repeats."""

    def __init__(self, reports: Sequence[ScanReport]) -> None:
        self._reports = list(reports)
        self.calls = 0

    def scan(self, top: Any, close: Any) -> ScanReport:
        report = self._reports[min(self.calls, len(self._reports) - 1)]
        self.calls += 1
        return report


class FakeDeep:
    """Stand-in for the Inspector deep check on the matched brick of the still frames."""

    def __init__(self, per_camera: Mapping[str, Mapping[str, Any]] | None) -> None:
        self._per_camera = {name: dict(result) for name, result in (per_camera or {}).items()}
        self.checked: list[str] = []

    def check(self, top: Any, close: Any, target: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        self.checked.append(str(target.get("id")))
        return {name: dict(result) for name, result in self._per_camera.items()}


class FakeStills:
    """Stand-in for the 1280x720 locked-exposure still capture. The scale is 1280 / 640 (scan frames are 640x480)."""

    scale = 2.0

    def __init__(self, log: RunDir) -> None:
        self._log = log
        self.captures = 0

    async def capture(self) -> tuple[str, str, float]:
        self.captures += 1
        return "still-top", "still-close", self.scale

    def save(self, relative: str, image: Any) -> str:
        return self._log.save_bytes(relative, str(image).encode())


def brick(brick_id: str, verdict: str, *, shape: str = "2x2", colour: str = "red",
          centre: tuple[float, float] = (300.0, 200.0), reasons: Sequence[str] = ()) -> dict[str, Any]:
    """One DET2 brick dict with the LOOP-CONTRACT section 8 scan fields."""
    half = 32.0
    return {
        "id": brick_id,
        "colour": colour,
        "top_bbox": [centre[0] - half, centre[1] - half, centre[0] + half, centre[1] + half],
        "centre_px": [centre[0], centre[1]],
        "angle_deg": 0.0,
        "footprint_px": 2 * half,
        "shape": shape,
        "shape_conf": 0.9,
        "tape_score": 0.1,
        "tape_conf": 0.9,
        "verdict": verdict,
        "confident": verdict == "CONFIDENT_DEFECT",
        "reasons": list(reasons),
        "crop_top": None,
        "crop_close": None,
    }


def scaled(brick_dict: Mapping[str, Any], scale: float = FakeStills.scale) -> dict[str, Any]:
    """The same brick as seen in the 1280x720 still scan."""
    cx, cy = brick_dict["centre_px"]
    return {**brick_dict, "centre_px": [cx * scale, cy * scale]}


def report(*bricks: Mapping[str, Any], hand_top: bool | None = False, hand_close: bool | None = False) -> ScanReport:
    return ScanReport(bricks=tuple(bricks), hand_top=hand_top, hand_close=hand_close)


def camera(label: str, score: float, threshold: float = 0.5, reasons: Sequence[str] = ()) -> dict[str, Any]:
    return {"label": label, "score": score, "threshold": threshold, "reasons": list(reasons), "device": "dry-run",
            "latency_ms": 0.0}


GOOD_DEEP = {"top": camera("GOOD", 0.12), "close": camera("GOOD", 0.10)}
TAPED_DEEP = {"top": camera("GOOD", 0.20), "close": camera("DEFECTIVE", 0.91, reasons=["tape/appearance"])}
UNCERTAIN_DEEP = {"top": camera("UNCERTAIN", 0.50), "close": camera("GOOD", 0.10)}


@dataclass(frozen=True)
class Scenario:
    description: str
    scan: tuple[ScanReport, ...]  # the policy scan, then the still scan when the brick is inspected
    deep: Mapping[str, Any] | None
    trajectory: str  # key of TRAJECTORIES
    expect: str  # the run's reason
    fault_at_s: float | None = None
    schedule: Mapping[int, str] = field(default_factory=dict)


_GOOD = brick("b1", "GOOD")
_UNSURE = brick("b1", "UNCERTAIN")
_DIRECT = brick("b1", "CONFIDENT_DEFECT", shape="not_2x2", reasons=["shape:not_2x2"])

SCENARIOS: dict[str, Scenario] = {
    "good_2x2_blue": Scenario("clean 2x2, inspected GOOD, released BLUE", (report(_GOOD), report(scaled(_GOOD))),
                              GOOD_DEEP, "blue", "max_cycles"),
    "taped_2x2_green": Scenario("taped 2x2, inspected DEFECTIVE, released GREEN", (report(_UNSURE), report(scaled(_UNSURE))),
                                TAPED_DEEP, "green", "max_cycles"),
    "uncertain_2x2_green": Scenario("unsure brick, inspected UNCERTAIN, released GREEN",
                                    (report(_UNSURE), report(scaled(_UNSURE))), UNCERTAIN_DEEP, "green", "max_cycles"),
    "2x4_green_direct": Scenario("2x4, CONFIDENT_DEFECT, no inspection, released GREEN", (report(_DIRECT),), None,
                                 "green", "max_cycles"),
    "policy_timeout": Scenario("policy grasps and never releases", (report(_GOOD), report(scaled(_GOOD))), GOOD_DEEP,
                               "stuck", "policy_timeout"),
    "wrong_plate_release": Scenario("GOOD brick meant for BLUE, policy releases on GREEN",
                                    (report(_GOOD), report(scaled(_GOOD))), GOOD_DEEP, "green", "verify_failed"),
    "hand_in_scene": Scenario("hand seen in the close view at the scan", (report(_GOOD, hand_close=True),), None,
                              "blue", "hand_in_view"),
    "runtime_error": Scenario("Studio session drops 1 s into ACT", (report(_GOOD), report(scaled(_GOOD))), GOOD_DEEP,
                              "blue", "error", fault_at_s=1.0),
    "operator_stop": Scenario("operator stop arrives during ACT", (report(_GOOD), report(scaled(_GOOD))), GOOD_DEEP,
                              "blue", "stopped", schedule={60: "stop"}),
}


class FakeHome:
    """Stand-in for the scripted return home after a policy timeout. The dry run only records that it was asked."""

    def __init__(self) -> None:
        self.calls = 0

    async def return_home(self) -> bool:
        self.calls += 1
        return True


def dry_config() -> PolicyConfig:
    plates = {
        BLUE: PlateSkill(skill="S2", plate=BLUE, model_id="DRYRUN-PLATE-BLUE",
                         task="move the red brick to the blue plate"),
        GREEN: PlateSkill(skill="S3", plate=GREEN, model_id="DRYRUN-PLATE-GREEN",
                          task="move the red brick to the green plate"),
    }
    return PolicyConfig(plates=plates, top_key=TOP_KEY, close_key=CLOSE_KEY, home=HOME_REFERENCE, max_cycles=1)


def dry_environment() -> dict[str, Any]:
    return {"name": "SO101 Recording - two cameras (dry run)", "robots": [], "cameras": []}


async def run_dry(name: str, logs_root: Path) -> dict[str, Any]:
    """One dry-run policy cycle for a scenario. logs_root receives the run directory. Returns a summary."""
    spec = SCENARIOS[name]
    clock = VirtualClock()
    log = RunDir(Path(logs_root))
    runtime = FakeRuntime(clock, trajectory=TRAJECTORIES[spec.trajectory], fault_at_s=spec.fault_at_s)
    stills = FakeStills(log)
    cycle = PolicyCycle(
        dry_config(),
        runtime=runtime,
        scanner=FakeScanner(spec.scan),
        deep=FakeDeep(spec.deep),
        stills=stills,
        control=ScheduledControl(dict(spec.schedule)),
        log=log,
        clock=clock,
        env_draft=dry_environment(),
        port_free=lambda: True,
        autostart=True,
        homer=FakeHome(),
    )
    reason = await cycle.run()
    log.close()
    return {
        "scenario": name,
        "description": spec.description,
        "reason": reason,
        "run_dir": str(log.path),
        "emitted": dict(sorted(log.emitted.items())),
        "calls": [call for call, _ in runtime.calls],
        "policy_starts": [kwargs for call, kwargs in runtime.calls if call == "start_policy"],
        "still_captures": stills.captures,
    }
