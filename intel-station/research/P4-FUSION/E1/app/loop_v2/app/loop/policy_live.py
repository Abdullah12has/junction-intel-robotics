"""Live wiring of loop_v2 policy mode. NOT RUN in the sandbox and NOT RUN on the laptop (dry runs only, by the task).

It starts only when every input exists. Otherwise it names what is missing and stops before any device is opened:
the presence flag, the plate model ids (placeholders refuse), the DET2 scanner module and the still exposure lock.
Every name used here is the candidate Studio client API (app/loop/studio_runtime_client.py). The DET2 module reads cv2
(BGR) frames; the Studio frames and the still frames are RGB, so this file converts them at that boundary.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import importlib
import os
import sys
import time
from pathlib import Path
from typing import Any

from loop.events import ControlFile, RunDir
from loop.policy_cycle import PlateSkill, PolicyConfig, PolicyCycle
from loop.triage import BLUE, GREEN, ScanReport

TOP_SCAN_WIDTH = 640  # the DET2 scan frames are 640x480


class LiveRefused(RuntimeError):
    """A live input is missing. Nothing has been opened."""


def _rgb(image: Any) -> Any:
    return image.rgb() if hasattr(image, "rgb") else image


def _swap(image: Any) -> Any:
    """RGB <-> BGR as a contiguous array (a reversed channel axis)."""
    import numpy as np

    return np.ascontiguousarray(np.asarray(image)[..., ::-1])


def _bgr(image: Any) -> Any:
    return _swap(_rgb(image))


def missing_inputs(rules: dict[str, Any]) -> list[str]:
    missing = []
    flag = Path(rules["paths"]["presence_flag"])
    if not flag.exists():
        missing.append(f"presence flag {flag} (Hassan present)")
    for name in (BLUE, GREEN):
        model_id = str(rules["plates"][name]["model_id"])
        if "PLACEHOLDER" in model_id:
            missing.append(f"{name} model_id is still a placeholder")
    if not rules["det2"]["module_dir"]:
        missing.append("det2.module_dir (P2-B-DET2 scanner not delivered)")
    if rules["still"]["exposure"] is None:
        missing.append("still.exposure (locked exposure not calibrated)")
    return missing


def config_from_rules(rules: dict[str, Any]) -> PolicyConfig:
    plates = {
        name: PlateSkill(
            skill=str(rules["plates"][name]["skill"]),
            plate=name,
            model_id=str(rules["plates"][name]["model_id"]),
            task=str(rules["plates"][name]["task"]),
            backend=str(rules["plates"][name]["backend"]),
            device=str(rules["plates"][name]["device"]),
            hz=float(rules["studio"]["hz"]),
            mode=str(rules["plates"][name]["mode"]),
            request_threshold=float(rules["plates"][name]["request_threshold"]),
            grace_s=float(rules["plates"][name]["grace_s"]),
        )
        for name in (BLUE, GREEN)
    }
    timing = rules["timing"]
    telemetry_day = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
    return PolicyConfig(
        plates=plates,
        top_key=str(rules["studio"]["top_key"]),
        close_key=str(rules["studio"]["close_key"]),
        home=dict(rules["home_reference"]),
        home_tol=float(timing["home_tol"]),
        still_s=float(timing["still_s"]),
        eps=float(timing["motion_eps"]),
        grasp_below=float(timing["grasp_below"]),
        grasp_hold_s=float(timing["grasp_hold_s"]),
        release_at=float(timing["release_at"]),
        act_timeout_s=float(timing["act_timeout_s"]),
        home_wait_s=float(timing["home_wait_s"]),
        scan_retries=int(timing["scan_retries"]),
        scan_wait_s=float(timing["scan_wait_s"]),
        verify_settle_s=float(timing["verify_settle_s"]),
        frames_ready_s=float(timing["frames_ready_s"]),
        uncertain_margin=float(rules["triage"]["uncertain_margin"]),
        max_cycles=int(timing["max_cycles"]),
        live_source="studio",
        detector_backend=str(rules["detector"]["backend"]),
        detector_device=str(rules["detector"]["device"]),
        telemetry_path=str(Path(rules["paths"]["telemetry_dir"]) / f"{telemetry_day}.jsonl"),
    )


class RealClock:
    def now(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


def _record(brick: dict[str, Any]) -> dict[str, Any]:
    """A DET2 brick record for the policy. Its bbox [x, y, w, h] becomes [x0, y0, x1, y1] under top_bbox or close_bbox."""
    record = dict(brick)
    bbox = brick.get("bbox")
    xyxy = None
    if bbox is not None:
        x, y, w, h = (int(value) for value in bbox)
        xyxy = [x, y, x + w, y + h]
    record["top_bbox"] = xyxy if brick.get("view") == "top" else None
    record["close_bbox"] = xyxy if brick.get("view") == "close" else None
    return record


class DetScanner:
    """Binds to the P2-B-DET2 module (app/anomaly/det2_e2/inspector.py). scan(top, close) takes BGR frames and returns the
    brick records of both views; it raises HandsPresent when the top view shows a hand. A normal return means the top view
    is clear. The module has no close-view hand check, so hand_close stays None and the scan refuses motion (fail-safe)
    until DET2 reports the close view."""

    def __init__(self, module: Any) -> None:
        self._module = module

    def scan(self, top: Any, close: Any) -> ScanReport:
        try:
            bricks = self._module.scan(_bgr(top), _bgr(close))
        except Exception as error:
            if not isinstance(error, getattr(self._module, "HandsPresent", ())):
                raise
            return ScanReport(bricks=(), hand_top=True, hand_close=None)
        return ScanReport(bricks=tuple(_record(brick) for brick in bricks or ()), hand_top=False, hand_close=None)


class UnbuiltHome:
    """The scripted return home (candidates/code/motion, under the I-43 gates) is not part of this build. A policy timeout
    therefore holds the arm with torque on, and Opus returns it."""

    async def return_home(self) -> bool:
        return False


class InspectorDeep:
    """Per-camera deep check of the matched brick: the crop of its top_bbox (top) or close_bbox (close) in the still frames,
    run through the DET2 Inspector. A missing box is no_brick (UNCERTAIN, which routes to GREEN)."""

    def __init__(self, inspector: Any, top_key: str, close_key: str) -> None:
        self._inspector = inspector
        self._keys = {"top": top_key, "close": close_key}

    def check(self, top: Any, close: Any, target: dict[str, Any]) -> dict[str, dict[str, Any]]:
        images = {"top": _bgr(top), "close": _bgr(close)}
        out: dict[str, dict[str, Any]] = {}
        for name, bbox in (("top", target.get("top_bbox")), ("close", target.get("close_bbox"))):
            if bbox is None:
                out[name] = {"label": "UNCERTAIN", "score": None, "threshold": None, "reasons": ["no_brick"]}
                continue
            x0, y0, x1, y1 = (int(value) for value in bbox)
            result = dict(self._inspector.inspect_rgb(images[name][y0:y1, x0:x1], camera_key=self._keys[name]))
            if result.get("crop") is not None:
                result["crop"] = _swap(result["crop"])  # the module's crop is BGR; the saved crop is RGB
            out[name] = result
        return out


class Stills:
    """Both RealSense colour nodes at 1280x720 with exposure locked (V4L2 through cv2). Devices open only in capture()."""

    def __init__(self, log: RunDir, devices: dict[str, str], width: int, height: int, exposure: float) -> None:
        self._log = log
        self._devices = devices
        self._width = width
        self._height = height
        self._exposure = exposure

    async def capture(self) -> tuple[Any, Any, float]:
        import cv2

        frames = []
        for name in ("top", "close"):
            capture = cv2.VideoCapture(self._devices[name], cv2.CAP_V4L2)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # [unclear] manual mode value on this UVC driver
            capture.set(cv2.CAP_PROP_EXPOSURE, self._exposure)
            for _ in range(30):  # let the locked exposure settle
                capture.read()
            ok, frame = capture.read()
            capture.release()
            if not ok:
                raise LiveRefused(f"no still frame from {self._devices[name]}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return frames[0], frames[1], self._width / TOP_SCAN_WIDTH

    def save(self, relative: str, image: Any) -> str:
        import cv2

        cv2.imwrite(str(self._log.path / relative), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return relative


def follower_port_free(motion_dir: str, link: str) -> bool:
    """True when no process holds the follower's current node. Uses the candidate motion layer (port_holder_pid)."""
    if motion_dir not in sys.path:
        sys.path.insert(0, motion_dir)
    motion = importlib.import_module("so101_motion")
    return motion.port_holder_pid(os.path.realpath(link)) is None


async def run_live(rules_path: Path) -> str:
    import yaml

    rules = yaml.safe_load(Path(rules_path).read_text(encoding="utf-8"))
    missing = missing_inputs(rules)
    if missing:
        raise LiveRefused("; ".join(missing))
    det2_dir = str(rules["det2"]["module_dir"])
    if det2_dir not in sys.path:
        sys.path.insert(0, det2_dir)
    det2 = importlib.import_module(str(rules["det2"]["module"]))

    from loop.studio_runtime_client import StudioRuntimeClient  # candidate client, copied unchanged

    config = config_from_rules(rules)
    log = RunDir(Path(rules["paths"]["logs_root"]))
    studio = rules["studio"]
    draft = {
        "name": studio["environment_name"],
        "robots": [{"robot_id": studio["follower_id"], "tele_operator": {"type": "none"}}],
        "cameras": [{"camera_id": camera_id} for camera_id in studio["camera_ids"]],
    }
    still = rules["still"]
    deep_module = det2.Inspector(model_dir=None, device=str(rules["detector"]["device"]))
    async with StudioRuntimeClient(studio["project_id"], base_url=studio["base_url"], hz=float(studio["hz"])) as client:
        cycle = PolicyCycle(
            config,
            runtime=client,
            scanner=DetScanner(det2),
            deep=InspectorDeep(deep_module, studio["top_key"], studio["close_key"]),
            stills=Stills(log, dict(still["devices"]), int(still["width"]), int(still["height"]),
                          float(still["exposure"])),
            control=ControlFile(log.path / "control.json"),
            log=log,
            clock=RealClock(),
            env_draft=draft,
            port_free=lambda: follower_port_free(rules["paths"]["motion_dir"], rules["paths"]["follower_link"]),
            autostart=False,
            homer=UnbuiltHome(),
        )
        reason = await cycle.run()
    log.close()
    return reason
