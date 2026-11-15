"""Spot cell cycle for loop v3 (D-082 mode c).

WAIT -> SCAN -> TRIAGE -> INSPECT (uncertain only) -> DECISION -> PICK -> GRASP CHECK -> SORT -> VERIFY -> WAIT.
Routing comes from loop.triage (I-52, OD-10). Hardware sits behind duck-typed objects: camera (top_frame, stills),
detector (scan_top, scan), arm (run, teardown) and control (poll). loop.run_spot builds the live adapters and the fakes.
Operator commands and the stop file are read every POLL_S during waits and before every motion segment (B10 to B12).
"""

from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from loop import triage as T
from loop.events import RunDir, write_jpeg_atomic

POLL_S = 0.2  # operator and stop-file read period (B10 to B12)
SCAN_KEYS = ("id", "colour", "centre_px", "angle_deg", "shape", "shape_conf", "tape_score", "tape_conf", "verdict",
             "confident", "reasons")
LABEL_FOR_VERDICT = {"CONFIDENT_DEFECT": T.DEFECTIVE, "DEFECTIVE": T.DEFECTIVE, "GOOD": T.GOOD,
                     "CONFIDENT_GOOD": T.GOOD, "UNCERTAIN": T.UNCERTAIN}
PLATE_SLUG = {T.BLUE: "blue", T.GREEN: "green"}
HOME_SEQUENCE = {"blue": "blue_to_home", "green": "green_to_home"}  # recorded return paths (B5)


class Stop(Exception):
    """Ends the run with this reason. The arm is then put away, unless it holds a brick."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class HandsDetected(RuntimeError):
    """A hand is in the top view: refuse to start or continue (B1)."""


@dataclass(frozen=True)
class SegmentResult:
    ok: bool
    exit_code: int
    latency_ms: float
    gripper: float | None
    detail: str = ""


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float

    def holds(self, centre: tuple[float, float] | None) -> bool:
        return centre is not None and math.hypot(centre[0] - self.cx, centre[1] - self.cy) <= self.r


@dataclass
class Config:
    top_roi: Circle
    close_roi: Circle | None = None
    blue_roi: Circle | None = None
    green_roi: Circle | None = None
    hold_min: float = 1.6
    hold_max: float = 4.5
    missed_below: float = 1.5
    grasp_retries: int = 1
    stable_s: float = 1.0
    wait_timeout_s: float = 300.0
    wait_poll_s: float = 0.5
    live_period_s: float = 0.5
    inspect_pairs: int = 3
    inspect_gap_s: float = 0.3
    max_cycles: int = 10
    heartbeat_s: float = 1.0
    write_images: bool = True
    stop_file: str | None = None


def spot_brick(bricks: Sequence[Mapping[str, Any]], roi: Circle | None) -> Mapping[str, Any] | None:
    """The brick whose centre lies in roi and is nearest its centre. Bricks outside the ROI are ignored (B2)."""
    best: Mapping[str, Any] | None = None
    best_distance = math.inf
    for brick in bricks:
        centre = T.centre_of(brick)
        if roi is None or not roi.holds(centre):
            continue
        distance = math.hypot(centre[0] - roi.cx, centre[1] - roi.cy)
        if distance < best_distance:
            best, best_distance = brick, distance
    return best


def on_plate(bricks: Sequence[Mapping[str, Any]], roi: Circle | None) -> bool:
    return roi is not None and any(roi.holds(T.centre_of(brick)) for brick in bricks)


def choose_target(top: Sequence[Mapping[str, Any]], close: Sequence[Mapping[str, Any]], cfg: Config) -> dict[str, Any] | None:
    """Pairs the spot brick of both views (B2) and fuses them. A CONFIDENT_DEFECT from any view wins (I-52)."""
    views = {"top": spot_brick(top, cfg.top_roi), "close": spot_brick(close, cfg.close_roi)}
    if all(brick is None for brick in views.values()):
        return None
    confident = [name for name, brick in views.items() if brick is not None and brick.get("verdict") == "CONFIDENT_DEFECT"]
    if confident:
        verdict, reasons = "CONFIDENT_DEFECT", [f"{name}:confident_defect" for name in confident]
    else:
        per_camera: dict[str, dict[str, Any]] = {}
        for name, brick in views.items():
            if brick is None:
                per_camera[name] = {"label": T.UNCERTAIN, "reasons": ["no_brick"]}
            else:
                per_camera[name] = {"label": LABEL_FOR_VERDICT.get(str(brick.get("verdict")), T.UNCERTAIN),
                                    "reasons": list(brick.get("reasons") or [])}
        verdict, reasons = T.fuse(per_camera)
    reference = views["top"] if views["top"] is not None else views["close"]
    return {"target_id": reference.get("id"), "verdict": verdict, "reasons": reasons,
            "views": {name: (brick.get("id") if brick else None) for name, brick in views.items()}}


def vote_of(target: Mapping[str, Any] | None) -> str:
    if target is None:
        return T.UNCERTAIN
    return T.DEFECTIVE if target["verdict"] == "CONFIDENT_DEFECT" else str(target["verdict"])


def majority(votes: Sequence[str]) -> str:
    """Two of the three pairs must agree on GOOD or DEFECTIVE; otherwise UNCERTAIN (fail-safe to GREEN)."""
    label, count = Counter(votes).most_common(1)[0]
    return label if count >= 2 and label != T.UNCERTAIN else T.UNCERTAIN


class SpotCell:
    """One run of the spot cell. run_loop() returns the run_end reason."""

    def __init__(self, cfg: Config, run: RunDir, camera: Any, detector: Any, arm: Any, control: Any, clock: Any) -> None:
        self.cfg, self.run = cfg, run
        self.camera, self.detector, self.arm, self.control, self.clock = camera, detector, arm, control, clock
        self.state = "IDLE"
        self.started = False
        self.paused = False
        self.stop_reason: str | None = None
        self.next_pending = False
        self.held = False
        self.location = "home"
        self.plate: str | None = None
        self.via: str | None = None
        self.cycles = 0
        self.counts = {"good": 0, "defective": 0, "uncertain": 0}
        self.cycle_seconds = 0.0
        self._last_beat = -math.inf
        self._last_live = -math.inf

    # events and state ---------------------------------------------------------------------------------------------
    def emit(self, event_type: str, **fields: Any) -> None:
        self.run.emit(event_type, **fields)

    def set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.emit("state", state=state)

    def beat(self) -> None:
        now = self.clock.now()
        if now - self._last_beat >= self.cfg.heartbeat_s:
            self._last_beat = now
            self.emit("heartbeat")

    # operator control and the stop file (B10 to B12) --------------------------------------------------------------
    def apply(self, name: str) -> bool:
        if name == "start":
            accepted, self.started = not self.started, True
        elif name == "pause":
            accepted, self.paused = not self.paused, True
        elif name == "resume":
            accepted, self.paused = self.paused, False
        elif name == "next":
            accepted = self.state == "WAIT" and not self.next_pending
            self.next_pending = self.next_pending or accepted
        elif name in ("stop", "quit"):
            accepted = True
            self.stop_reason = self.stop_reason or ("stopped" if name == "stop" else "quit")
        else:
            accepted = False
        return accepted

    def read_controls(self) -> None:
        for command in self.control.poll():
            name = str(command.get("cmd", ""))
            self.emit("control", cmd=name, id=command.get("id"), accepted=self.apply(name))
        if self.cfg.stop_file and os.path.exists(self.cfg.stop_file):
            self.stop_reason = self.stop_reason or "stop_file"
        if self.stop_reason:
            raise Stop(self.stop_reason)

    def idle_sleep(self, seconds: float) -> None:
        end = self.clock.now() + seconds
        while True:
            left = end - self.clock.now()
            if left > 1e-9:
                self.clock.sleep(min(POLL_S, left))
            self.beat()
            self.read_controls()
            if self.clock.now() >= end - 1e-9:
                return

    def checkpoint(self) -> None:
        """Read controls before a decision or a motion segment; hold while paused."""
        self.read_controls()
        if self.paused:
            resume_state = self.state
            self.set_state("PAUSED")
            while self.paused:
                self.idle_sleep(POLL_S)
            self.set_state(resume_state)

    # frames -------------------------------------------------------------------------------------------------------
    def write_live(self, frame: Any) -> None:
        now = self.clock.now()
        if self.cfg.write_images and frame is not None and now - self._last_live >= self.cfg.live_period_s:
            self._last_live = now
            write_jpeg_atomic(self.run.path / "live.jpg", frame[:, :, ::-1].copy())  # BGR to RGB for the preview

    def save_frame(self, name: str, frame: Any) -> str | None:
        if frame is None or not self.cfg.write_images:
            return None
        relative = f"frames/c{self.run.cycle:03d}_{name}.jpg"
        write_jpeg_atomic(self.run.path / relative, frame[:, :, ::-1].copy())
        return relative

    # arm segments -------------------------------------------------------------------------------------------------
    def segment(self, name: str, phase: str, plate: str | None = None) -> SegmentResult:
        seg = self.arm.run(name)
        self.emit("policy_run", phase=phase, sequence=name, plate=plate, model_id=None, task_text=None,
                  latency_ms=round(seg.latency_ms, 1), ok=seg.ok, exit_code=seg.exit_code, gripper=seg.gripper)
        if not seg.ok:
            self.emit("error", where=phase, message=seg.detail or f"exit {seg.exit_code}")
        return seg

    def drive(self, name: str, phase: str, plate: str | None = None) -> SegmentResult:
        self.checkpoint()  # stop, pause and stop file are read before every segment (B10 to B12)
        return self.segment(name, phase, plate)

    # run ------------------------------------------------------------------------------------------------------------
    def run_loop(self, config: Mapping[str, Any]) -> str:
        self.emit("run_start", mode="spot", config=dict(config))
        reason = "error"
        try:
            self.wait_for_start()
            while self.cycles < self.cfg.max_cycles:
                self.wait_for_brick()
                self.cycle()
            reason = "max_cycles"
        except Stop as stop:
            reason = stop.reason
        except HandsDetected as hands:
            self.emit("error", where=self.state.lower(), message=f"hand_in_view: {hands}")
            reason = "hand_in_view"
        except Exception as error:  # any other fault ends the run the same safe way
            self.emit("error", where=self.state.lower(), message=repr(error))
        reason = self.finish(reason)
        self.emit("run_end", reason=reason, held=self.held, location=self.location, cycles=self.cycles)
        return reason

    def finish(self, reason: str) -> str:
        """Every ending (B4): arm home with torque off. A held brick or an unknown arm position keeps torque on."""
        if not self.started:
            return reason  # never started: the arm was not touched
        if self.held:
            self.emit("error", where="hold", message="arm holds a brick: torque stays on, no motion, operator must act")
            return reason
        try:
            if self.location in HOME_SEQUENCE:
                back = self.segment(HOME_SEQUENCE[self.location], "return")
                self.location = "home" if back.ok else "unknown"
            if self.location != "home":
                self.emit("error", where="teardown", message=f"arm not at home ({self.location}): torque left on")
                return reason
            put_away = self.arm.teardown()
            self.emit("policy_run", phase="done" if put_away.ok else "error", sequence="teardown", plate=None,
                      model_id=None, task_text=None, latency_ms=round(put_away.latency_ms, 1), ok=put_away.ok,
                      exit_code=put_away.exit_code, gripper=None)
        except Exception as error:
            self.emit("error", where="teardown", message=repr(error))
        return reason

    def wait_for_start(self) -> None:
        self.set_state("IDLE")
        while not self.started:
            self.idle_sleep(POLL_S)

    def wait_for_brick(self) -> None:
        """Top frame every wait_poll_s; a brick in the spot ROI for stable_s, or 'next', starts a cycle (hands refuse)."""
        self.set_state("WAIT")
        self.run.cycle = self.cycles
        started_at = self.clock.now()
        stable_since: float | None = None
        while True:
            self.checkpoint()
            frame = self.camera.top_frame()
            self.write_live(frame)
            bricks = self.detector.scan_top(frame)  # raises HandsDetected
            if self.next_pending:
                self.next_pending = False
                return
            if spot_brick(bricks, self.cfg.top_roi) is None:
                stable_since = None
            else:
                stable_since = self.clock.now() if stable_since is None else stable_since
                if self.clock.now() - stable_since >= self.cfg.stable_s:
                    return
            if self.clock.now() - started_at >= self.cfg.wait_timeout_s:
                raise Stop("empty_spot_timeout")
            self.idle_sleep(self.cfg.wait_poll_s)

    def scan(self) -> dict[str, Any] | None:
        self.set_state("SCAN")
        self.checkpoint()
        top, close = self.camera.stills()
        result = self.detector.scan(top, close)  # raises HandsDetected
        top_bricks, close_bricks = list(result.get("top", [])), list(result.get("close", []))
        target = choose_target(top_bricks, close_bricks, self.cfg)
        rows = [{"view": view, **{key: brick.get(key) for key in SCAN_KEYS}}
                for view, bricks in (("top", top_bricks), ("close", close_bricks)) for brick in bricks]
        self.emit("scan", bricks=rows, target_id=(target or {}).get("target_id"),
                  verdict=(target or {}).get("verdict"), views=(target or {}).get("views"),
                  frame_top=self.save_frame("top", top), frame_close=self.save_frame("close", close))
        return target

    def inspect_again(self) -> str:
        """Uncertain only: inspect_pairs more still pairs, inspect_gap_s apart, majority vote (I-52)."""
        self.set_state("INSPECT")
        votes: list[str] = []
        for pair in range(1, self.cfg.inspect_pairs + 1):
            self.idle_sleep(self.cfg.inspect_gap_s)
            self.checkpoint()
            top, close = self.camera.stills()
            began = self.clock.now()
            result = self.detector.scan(top, close)
            target = choose_target(list(result.get("top", [])), list(result.get("close", [])), self.cfg)
            votes.append(vote_of(target))
            self.emit("inspection", mode="still_hires", pair=pair, label=votes[-1], result=votes[-1], score=None,
                      threshold=None, reasons=(target or {}).get("reasons", ["no_brick"]), route=T.INSPECT,
                      latency_ms=round((self.clock.now() - began) * 1000.0, 1),
                      frame=self.save_frame(f"inspect{pair}_top", top))
        return majority(votes)

    def pick_with_grasp_check(self) -> None:
        """PICK, then the gripper band: 1.6 to 4.5 held; below 1.5 missed (home, rescan, one retry); else stop."""
        attempts = 0
        while True:
            self.set_state("PICK")
            self.checkpoint()
            self.held = True  # from here the gripper may close: a stop or an error now holds and reports
            pick = self.segment("spot_pick_from_home", "grasp", self.plate)
            if not pick.ok:
                raise Stop("pick_failed")
            self.location = "home"  # spot_pick_from_home ends at spot_wp9_closed, the home side of the spot path
            grip = pick.gripper
            if grip is None:
                raise Stop("grasp_unreadable")
            if self.cfg.hold_min <= grip <= self.cfg.hold_max:
                return
            if grip >= self.cfg.missed_below:
                raise Stop("grasp_unclear")
            attempts += 1
            self.emit("error", where="grasp", message="missed", gripper=grip, attempt=attempts)
            self.held = False
            back = self.drive("spot_to_home_open", "return", self.plate)
            if not back.ok:
                self.location = "unknown"
                raise Stop("return_failed")
            if attempts > self.cfg.grasp_retries:
                raise Stop("grasp_failed")
            if self.scan() is None:
                raise Stop("brick_gone")

    def sort(self) -> None:
        slug = PLATE_SLUG[self.plate or T.GREEN]
        self.set_state("SORT")
        began = self.clock.now()
        self.emit("sort_start", route=self.plate)
        if not self.drive(f"home_to_{slug}_carry", "carry", self.plate).ok:
            raise Stop("carry_failed")
        if not self.drive(f"release_{slug}", "release", self.plate).ok:
            raise Stop("release_failed")
        self.held = False
        self.location = slug
        if not self.drive(f"{slug}_to_home", "return", self.plate).ok:
            self.location = "unknown"
            raise Stop("return_failed")
        self.location = "home"
        self.emit("sort_end", ok=True, duration_s=round(self.clock.now() - began, 2))

    def verify(self, target: Mapping[str, Any]) -> None:
        self.set_state("VERIFY")
        self.checkpoint()
        frame = self.camera.top_frame()
        bricks = self.detector.scan_top(frame)  # raises HandsDetected
        side = next((name for name, roi in ((T.BLUE, self.cfg.blue_roi), (T.GREEN, self.cfg.green_roi))
                     if on_plate(bricks, roi)), T.NONE)
        spot_empty = spot_brick(bricks, self.cfg.top_roi) is None
        ok = spot_empty and side == self.plate
        self.emit("verify", target_id=target.get("target_id"), plate_intended=self.plate, release_side=side, ok=ok,
                  frame_top=self.save_frame("verify_top", frame))
        if not ok:
            raise Stop("verify_failed")

    def cycle(self) -> None:
        began = self.clock.now()
        self.run.cycle = self.cycles + 1
        target = self.scan()
        if target is None:
            self.emit("error", where="scan", message="no_brick_at_spot")
            return
        route, reasons = T.triage(target)
        self.emit("triage", target_id=target["target_id"], route=route, reasons=reasons)
        inspected = None
        if route == T.INSPECT:
            inspected = self.inspect_again() if target["verdict"] == T.UNCERTAIN else target["verdict"]
        decision = T.decide(target["target_id"], route, inspected)
        self.emit("decision", **decision)
        self.plate, self.via = decision["plate"], decision["via"]
        self.pick_with_grasp_check()
        self.sort()
        self.verify(target)
        self.cycles += 1
        key = {T.GOOD: "good", T.UNCERTAIN: "uncertain"}.get(str(target["verdict"]), "defective")
        self.counts[key] += 1
        self.cycle_seconds += self.clock.now() - began
        self.emit("stats", cycles=self.cycles, good=self.counts["good"], defective=self.counts["defective"],
                  uncertain=self.counts["uncertain"], mean_cycle_s=round(self.cycle_seconds / self.cycles, 2))
