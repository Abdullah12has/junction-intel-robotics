"""Policy mode of loop_v2 (LOOP-CONTRACT section 8, OD-12): HOME -> SCAN -> TRIAGE -> [INSPECT] -> ACT -> VERIFY -> HOME.

The loop drives no motor. The Studio policy moves the follower for the plate's skill. The loop watches the joint state,
ends the policy with set_control null, and routes every brick with loop.triage. Each external part is injected: the
Studio runtime client, the DET2 scanner, the Inspector deep check, the still cameras, operator control, the run log and
the clock. Dry runs inject loop.policy_fakes. Nothing here opens a device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from loop.grip_monitor import ActMonitor, HomeStill
from loop.triage import (
    DEFECTIVE,
    GOOD,
    INSPECT,
    PLATE_FOR_LABEL,
    SKILL_FOR_PLATE,
    UNCERTAIN,
    decide,
    finite_number,
    find_target,
    fuse,
    hands_in_view,
    match_target,
    triage,
    verify_result,
)


class _Operator(Exception):
    """An operator stop or quit arrived."""


class _Safety(Exception):
    """A refusal or a failure. The cycle holds the arm and the run ends with this reason."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


@dataclass(frozen=True)
class PlateSkill:
    skill: str
    plate: str
    model_id: str
    task: str
    backend: str = "openvino"
    device: str = "GPU"  # D-080: every load names the GPU; 'auto' resolves to CPU and is 3-4x slower
    hz: float = 30.0
    mode: str = "sync"
    request_threshold: float = 0.5
    grace_s: float = 3.0

    def __post_init__(self) -> None:
        # D-080 (Opus): every Studio load_model the loop sends carries backend 'openvino' and device 'GPU', explicitly.
        if self.backend != "openvino" or self.device != "GPU":
            raise ValueError(f"plate {self.plate} must load with backend 'openvino' and device 'GPU' (D-080), "
                             f"got backend {self.backend!r} and device {self.device!r}")


@dataclass(frozen=True)
class PolicyConfig:
    plates: Mapping[str, PlateSkill]
    top_key: str
    close_key: str
    home: Mapping[str, float]
    home_tol: float = 10.0
    still_s: float = 1.0
    eps: float = 0.5
    grasp_below: float = 4.0
    grasp_hold_s: float = 0.3
    release_at: float = 6.0
    act_timeout_s: float = 45.0
    home_wait_s: float = 5.0
    scan_retries: int = 2
    scan_wait_s: float = 1.0
    verify_settle_s: float = 0.5
    frames_ready_s: float = 10.0
    uncertain_margin: float = 0.1
    max_cycles: int = 1
    live_source: str = "fake"  # run_start.config (section 9): fake in the dry runs, studio in the live run
    detector_backend: str = "fake"
    detector_device: str = "none"
    telemetry_path: str | None = None  # the day file of R/logs/telemetry; null in the dry runs


class _NoHome:
    """No scripted return home is configured: a policy timeout then holds the arm with torque on and nothing moves."""

    async def return_home(self) -> bool:
        return False


def _worst(per_camera: Mapping[str, Mapping[str, Any]]) -> tuple[Any, Any]:
    """(score, threshold) of the camera with the highest score, for the inspection event."""
    best: tuple[Any, Any] = (None, None)
    for result in per_camera.values():
        score = result.get("score")
        if finite_number(score) and (best[0] is None or score > best[0]):
            best = (score, result.get("threshold"))
    return best


class PolicyCycle:
    """One policy run: start, then bricks one after another. A refusal, error, stop or verify failure ends the run."""

    def __init__(
        self,
        config: PolicyConfig,
        *,
        runtime: Any,
        scanner: Any,
        deep: Any,
        stills: Any,
        control: Any,
        log: Any,
        clock: Any,
        env_draft: Mapping[str, Any],
        port_free: Any = None,
        autostart: bool = False,
        homer: Any = None,
    ) -> None:
        self.c = config
        self.rt = runtime
        self.scanner = scanner
        self.deep = deep
        self.stills = stills
        self.control = control
        self.log = log
        self.clock = clock
        self.env_draft = dict(env_draft)
        self.port_free = port_free
        self.homer = homer if homer is not None else _NoHome()
        self.state = "IDLE"
        self.cycle = 0
        self.counts = {GOOD: 0, DEFECTIVE: 0, UNCERTAIN: 0}
        self.durations: list[float] = []
        self.started = autostart
        self.paused = False
        self.stop_requested = False
        self.quit_requested = False
        self._beat = clock.now()
        self._obs: Any = None

    # ----- events, state and operator control ----------------------------------------------------------------

    def _emit(self, event_type: str, **fields: Any) -> None:
        self.log.emit(event_type, cycle=self.cycle, **fields)

    def _state(self, name: str) -> None:
        self.state = name
        self.log.emit("state", state=name, cycle=self.cycle)

    async def _poll(self) -> None:
        for command in self.control.poll():
            name = command["cmd"]
            accepted = False
            if name == "start" and not self.started:
                self.started, accepted = True, True
            elif name == "pause" and not self.paused:
                self.paused, accepted = True, True
            elif name == "resume" and self.paused:
                self.paused, accepted = False, True
            elif name in ("stop", "quit"):
                self.stop_requested = True
                self.quit_requested = self.quit_requested or name == "quit"
                accepted = True
            self._emit("control", cmd=name, id=command["id"], accepted=accepted, state=self.state)
        now = self.clock.now()
        if now - self._beat >= 1.0:
            self._beat = now
            self._emit("heartbeat")

    async def _next_observation(self) -> Any:
        await self._poll()
        if self.stop_requested:
            raise _Operator("operator stop")
        return await self._obs.__anext__()

    async def _checkpoint(self) -> None:
        await self._poll()
        if self.stop_requested:
            raise _Operator("operator stop")

    async def _hold(self) -> None:
        """set_control null: the follower holds where it is. A failed hold is reported, never ignored."""
        try:
            await self.rt.stop()
        except Exception as error:
            self._emit("error", where="hold", message=f"{type(error).__name__}: {error}")

    async def _between_cycles(self) -> None:
        if not self.paused:
            return
        await self._hold()
        self._state("PAUSED")
        while self.paused and not self.stop_requested:
            await self.clock.sleep(0.1)
            await self._poll()
        if self.stop_requested:
            raise _Operator("operator stop")

    def summary(self) -> dict[str, Any]:
        """run_start.config, exactly the LOOP-CONTRACT section 9 shape."""
        return {
            "mode": "policy",
            "live_source": self.c.live_source,
            "detector": {"backend": self.c.detector_backend, "device": self.c.detector_device},
            "skills": {skill.skill: {"backend": skill.backend, "device": skill.device, "model_id": skill.model_id}
                       for skill in self.c.plates.values()},
            "telemetry": {"path": self.c.telemetry_path},
        }

    # ----- run -----------------------------------------------------------------------------------------------

    async def run(self) -> str:
        self.log.emit("run_start", run_id=self.log.run_id, mode="policy", config=self.summary())
        try:
            reason = await self._main()
        except Exception as error:
            self._emit("error", where=self.state, message=f"{type(error).__name__}: {error}")
            await self._hold()
            reason = "error"
        self._state("STOPPED")
        self.log.emit("run_end", reason=reason)
        return reason

    async def _main(self) -> str:
        self._state("IDLE")
        while not self.started:
            await self._poll()
            if self.stop_requested:
                return "quit" if self.quit_requested else "stopped"
            await self.clock.sleep(0.1)
        await self.rt.load_environment(self.env_draft)
        await self.rt.wait_for_frames([self.c.top_key, self.c.close_key], timeout_s=self.c.frames_ready_s)
        self._obs = self.rt.observations(decode=False)
        while True:
            try:
                await self._between_cycles()
            except _Operator:
                outcome = "quit" if self.quit_requested else "stopped"
                break
            outcome = await self._cycle()
            if outcome != "done":
                break
            if self.cycle >= self.c.max_cycles:
                outcome = "max_cycles"
                break
        if outcome in ("max_cycles", "stopped", "quit", "no_brick"):
            await self._handback()
        else:
            await self._hold()
            if outcome == "policy_timeout":
                await self._return_home()
        return outcome

    async def _return_home(self) -> None:
        """After a policy timeout the policy is stopped and the arm is held with torque on (D-080); the scripted return home
        follows. This path never unloads the environment, so torque is not dropped on the way."""
        self._state("RETURN_HOME")
        try:
            homed = bool(await self.homer.return_home())
            message = "the scripted return home did not finish; the arm stays held with torque on"
        except Exception as error:
            homed, message = False, f"{type(error).__name__}: {error}"
        if homed:
            self._state("HOME")
        else:
            self._emit("error", where="return_home", message=message)
            await self._hold()

    async def _handback(self) -> None:
        """Return the arm to the team: hold, then unload the environment (Studio disconnects the follower)."""
        await self._hold()
        try:
            await self.rt.unload_environment()
        except Exception as error:
            self._emit("error", where="handback", message=f"{type(error).__name__}: {error}")
            return
        if self.port_free is not None and not self.port_free():
            self._emit("error", where="handback", message="follower port still held after unload_environment")

    # ----- one brick -----------------------------------------------------------------------------------------

    async def _cycle(self) -> str:
        self.cycle += 1
        self.log.cycle = self.cycle  # every event of this brick carries it (section 9)
        started = self.clock.now()
        try:
            self._state("HOME")
            await self._await_home()
            self._state("SCAN")
            target, target_id = await self._scan()
            if target is None:
                return "no_brick"
            self._state("TRIAGE")
            route, reasons = triage(target)
            self._emit("triage", target_id=target_id, route=route, reasons=reasons)
            label = None
            if route == INSPECT:
                self._state("INSPECT")
                label = await self._inspect(target, target_id)
            decision = decide(target_id, route, label)
            self._emit("decision", **decision)
            self._state("ACT")
            release_pan = await self._act(target_id, decision["plate"])
            self._state("VERIFY")
            await self.clock.sleep(self.c.verify_settle_s)
            result = self._verify(target_id, decision["plate"], release_pan)
            if not result["ok"]:
                raise _Safety(
                    "verify_failed",
                    f"brick {target_id} was meant for {decision['plate']} but was released on {result['release_side']}",
                )
            self._count(label or DEFECTIVE, started)
            return "done"
        except _Operator:
            await self._hold()
            return "quit" if self.quit_requested else "stopped"
        except _Safety as stop:
            self._emit("error", where=self.state, message=stop.message)
            await self._hold()
            return stop.reason

    async def _await_home(self) -> None:
        monitor = HomeStill(self.c.home, tol=self.c.home_tol, still_s=self.c.still_s, eps=self.c.eps)
        first: float | None = None
        while True:
            observation = await self._next_observation()
            if first is None:
                first = observation.timestamp
            if monitor.feed(observation.timestamp, observation.state):
                return
            if observation.timestamp - first > self.c.home_wait_s:
                raise _Safety("not_home", f"the arm was not at home and still within {self.c.home_wait_s:g} s")

    async def _scan(self) -> tuple[Any, str | None]:
        """SCAN with refusal: a hand, or no report in either view, stops the motion (fail-safe)."""
        for attempt in range(self.c.scan_retries + 1):
            top = self.rt.latest_frame(self.c.top_key)
            close = self.rt.latest_frame(self.c.close_key)
            if top is None or close is None:
                raise _Safety("no_frame", "a RealSense frame is missing from the Studio runtime stream")
            report = self.scanner.scan(top, close)
            if not hands_in_view(report.hand_top, report.hand_close):
                break
            if attempt < self.c.scan_retries:
                await self.clock.sleep(self.c.scan_wait_s)
                await self._checkpoint()
        target, how = find_target(report.bricks)
        target_id = None if target is None else str(target.get("id"))
        frame_top = self.log.save_bytes(f"frames/c{self.cycle:03d}_scan_top.jpg", top.jpeg)
        frame_close = self.log.save_bytes(f"frames/c{self.cycle:03d}_scan_close.jpg", close.jpeg)
        hands = hands_in_view(report.hand_top, report.hand_close)
        self._emit(
            "scan",
            bricks=list(report.bricks),
            target_id=target_id,
            target_how=how,
            hands={"top": report.hand_top, "close": report.hand_close},
            frame_top=frame_top,
            frame_close=frame_close,
        )
        if hands:
            raise _Safety("hand_in_view", f"motion refused: a hand, or no report, in the {', '.join(hands)} view")
        return target, target_id

    async def _inspect(self, target: Mapping[str, Any], target_id: str | None) -> str:
        """INSPECT (still_hires): unload, capture 1280x720 stills, deep check, fuse, then reload with the same draft."""
        started = self.clock.now()
        await self._checkpoint()
        await self.rt.unload_environment()
        label, reasons, per_camera = UNCERTAIN, ["no_brick"], {}
        frame_top = frame_close = None
        try:
            if self.port_free is not None and not self.port_free():
                raise _Safety("port_busy", "the follower port is still held after unload_environment")
            top_img, close_img, scale = await self.stills.capture()
            still = self.scanner.scan(top_img, close_img)
            if hands_in_view(still.hand_top, still.hand_close):
                raise _Safety("hand_in_view", "a hand, or no report, in the still frames")
            match = match_target(still.bricks, target, scale=scale)
            if match is not None:
                per_camera = self.deep.check(top_img, close_img, match)
                label, reasons = fuse(per_camera, margin=self.c.uncertain_margin)
            frame_top = self.stills.save(f"frames/c{self.cycle:03d}_still_top.jpg", top_img)
            frame_close = self.stills.save(f"frames/c{self.cycle:03d}_still_close.jpg", close_img)
        finally:
            await self.rt.load_environment(self.env_draft)  # HANDOVER_IN: same draft, the store and the model are kept
            await self.rt.wait_for_frames([self.c.top_key, self.c.close_key], timeout_s=self.c.frames_ready_s)
        cameras = self._inspect_cameras(per_camera, {"top": frame_top, "close": frame_close})
        self._emit("inspect_cameras", cameras=cameras)
        score, threshold = _worst(per_camera)
        first = next(iter(per_camera.values()), {})
        crop = next((camera["crop"] for camera in cameras if camera["crop"] is not None), None)
        self._emit(
            "inspection",
            target_id=target_id,
            mode="still_hires",
            result=label,
            label=label,
            reasons=reasons,
            score=score,
            threshold=threshold,
            geometry=None,
            device=first.get("device"),
            latency_ms=round((self.clock.now() - started) * 1000.0, 1),
            frame=frame_top,
            crop=crop,
            route=SKILL_FOR_PLATE[PLATE_FOR_LABEL[label]],
        )
        return label

    def _inspect_cameras(self, per_camera: Mapping[str, Mapping[str, Any]], frames: Mapping[str, Any]) -> list[dict[str, Any]]:
        """inspect_cameras (section 9): one entry per camera; UNCERTAIN with no_brick when the deep check did not run."""
        cameras = []
        for name in ("top", "close"):
            result = per_camera.get(name) or {"label": UNCERTAIN, "score": None, "threshold": None, "reasons": ["no_brick"]}
            crop = result.get("crop")  # an RGB array from the live deep check; the dry runs have none
            cameras.append({
                "camera": name,
                "label": result.get("label"),
                "score": result.get("score"),
                "threshold": result.get("threshold"),
                "reasons": list(result.get("reasons") or []),
                "frame": frames.get(name),
                "crop": None if crop is None else self.stills.save(f"frames/c{self.cycle:03d}_crop_{name}.jpg", crop),
            })
        return cameras

    async def _act(self, target_id: str | None, plate: str) -> float | None:
        """ACT: the plate's skill runs as the Studio policy until grasp, release and return, or the timeout."""
        skill = self.c.plates[plate]
        monitor = ActMonitor(
            self.c.home,
            home_tol=self.c.home_tol,
            still_s=self.c.still_s,
            eps=self.c.eps,
            grasp_below=self.c.grasp_below,
            grasp_hold_s=self.c.grasp_hold_s,
            release_at=self.c.release_at,
            timeout_s=self.c.act_timeout_s,
        )

        marks = [self.clock.now()]  # policy_run latency_ms: the time since the previous phase, or since the ACT start

        def phase(name: str, **data: Any) -> None:
            now = self.clock.now()
            latency = round((now - marks[0]) * 1000.0, 1)
            marks[0] = now
            self._emit("policy_run", phase=name, model_id=skill.model_id, task_text=skill.task, plate=plate,
                       target_id=target_id, backend=skill.backend, device=skill.device, latency_ms=latency, **data)

        self._emit("skill_start", skill=skill.skill, model=skill.model_id, plate=plate, task_text=skill.task,
                   backend=skill.backend, device=skill.device)
        started = self.clock.now()
        await self._checkpoint()
        await self.rt.wait_for_frames([self.c.top_key, self.c.close_key], timeout_s=self.c.frames_ready_s)
        await self.rt.start_policy(
            skill.model_id,
            skill.task,
            backend=skill.backend,
            device=skill.device,
            hz=skill.hz,
            mode=skill.mode,
            request_threshold=skill.request_threshold,
            grace_s=skill.grace_s,
        )
        phase("start")
        try:
            while not (monitor.done or monitor.timed_out):
                observation = await self._next_observation()
                for name, data in monitor.feed(observation.timestamp, observation.state):
                    phase(name, **data)
        except _Operator:
            self._emit("skill_end", skill=skill.skill, ok=False, duration_s=round(self.clock.now() - started, 3),
                       reason="operator stop")
            raise
        except Exception as error:
            phase("error", message=f"{type(error).__name__}: {error}")
            self._emit("skill_end", skill=skill.skill, ok=False, duration_s=round(self.clock.now() - started, 3),
                       reason="runtime error")
            raise
        await self.rt.stop()  # set_control null: the follower holds
        ok = monitor.done
        self._emit("skill_end", skill=skill.skill, ok=ok, duration_s=round(self.clock.now() - started, 3),
                   reason="done" if ok else "timeout")
        if not ok:
            raise _Safety("policy_timeout", f"the policy did not finish within {self.c.act_timeout_s:g} s")
        return monitor.release_pan

    def _verify(self, target_id: str | None, plate: str, release_pan: float | None) -> dict[str, Any]:
        top = self.rt.latest_frame(self.c.top_key)
        frame_top = None if top is None else self.log.save_bytes(f"frames/c{self.cycle:03d}_verify_top.jpg", top.jpeg)
        result = verify_result(target_id, plate, release_pan, frame_top)
        self._emit("verify", **result)
        return result

    def _count(self, label: str, started: float) -> None:
        self.counts[label] = self.counts.get(label, 0) + 1
        self.durations.append(self.clock.now() - started)
        self._state("CYCLE_DONE")
        self._emit(
            "stats",
            cycles=len(self.durations),
            good=self.counts[GOOD],
            defective=self.counts[DEFECTIVE],
            uncertain=self.counts[UNCERTAIN],
            mean_cycle_s=round(sum(self.durations) / len(self.durations), 3),
        )
