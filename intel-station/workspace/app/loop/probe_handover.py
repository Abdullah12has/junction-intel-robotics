#!/usr/bin/env python3
"""P2-B-PROBE E1: handover probe through Studio's runtime session, run on the laptop.

Uses only load_environment, load_model, unload_model, unload_environment, close and reads.
Never sends set_control, start_policy or stop (those drive the arm).
Halts (unload_environment, then socket close) when a new holder appears on the follower,
leader or RealSense nodes, or when a joint moves more than 2 units from its reference.
Refuses to start unless the gate passes (arm_state FREE for 60 s, no holders, presence flag).
Run: R/envs/data-env/bin/python3 -I R/app/loop/probe_handover.py
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path

R = Path("/home/intel/intel-junction-2026")
OUT = R / "work" / "P2-B-PROBE" / "E1"
sys.path.insert(0, str(R / "app" / "loop"))

from studio_runtime_client import StudioRuntimeClient  # noqa: E402

PROJECT = "ab767473-d6fd-47ff-812e-8239c455dabb"
BASE_URL = "http://127.0.0.1:7860"
FOLLOWER = "e5600356-6296-496e-8344-17995e419baf"
CAMERAS = ["1d80a0dd-8f62-4f23-ae29-2f1fe406dde9", "b6e5da09-28f2-4208-9d15-526580074c75"]
MODEL = "fa42ee76-bcfb-43ba-ba21-124e678ee785"
BACKEND = "openvino"
DEVICE = "auto"
DRAFT = {
    "name": "P2-B-PROBE",
    "robots": [{"robot_id": FOLLOWER, "tele_operator": {"type": "none"}}],
    "cameras": [{"camera_id": camera} for camera in CAMERAS],
}
HZ = 5
FOLLOWER_NODE = "/dev/ttyACM0"
WATCHED = frozenset({"/dev/ttyACM0", "/dev/ttyACM1", "/dev/video8", "/dev/video14"})
JOINT_LIMIT = 2.0


def utc_hms() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%SZ")


def holders() -> set[tuple[int, str]]:
    """Processes holding a watched node. Reads /proc fd symlinks only; no device is opened."""
    found: set[tuple[int, str]] = set()
    me = os.getpid()
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == me:
            continue
        fd_dir = f"/proc/{pid}/fd"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(f"{fd_dir}/{fd}")
            except OSError:
                continue
            if target in WATCHED:
                found.add((int(pid), target))
    return found


def follower_held() -> bool:
    return any(dev == FOLLOWER_NODE for _, dev in holders())


def gate() -> str | None:
    """None when the run may start; otherwise the reason it may not."""
    text = (R / "run" / "arm_state").read_text().strip()
    match = re.match(r"FREE since (\S+)", text)
    if not match:
        return f"arm_state not FREE: {text!r}"
    since = dt.datetime.fromisoformat(match.group(1).replace("Z", "+00:00"))
    age = (dt.datetime.now(dt.timezone.utc) - since).total_seconds()
    if age < 60:
        return f"arm_state FREE for only {age:.0f} s: {text!r}"
    busy = sorted(holders())
    if busy:
        return f"holders present: {busy}"
    if not (R / "run" / "HASSAN_PRESENT").exists():
        return "HASSAN_PRESENT flag absent"
    return None


class Probe:
    def __init__(self) -> None:
        self.client = StudioRuntimeClient(PROJECT, base_url=BASE_URL, hz=HZ)
        self.result: dict = {"started_utc": utc_hms(), "timings_s": {}, "events": [], "frames": [], "holders": {}}
        self.watching = False
        self.reference: dict[str, float] | None = None
        self.latest: dict[str, float] | None = None
        self.max_delta = 0.0
        self.baseline: set[tuple[int, str]] | None = None
        self.abort: tuple[str, str] | None = None
        self.tasks: list[asyncio.Task] = []

    def event(self, text: str) -> None:
        line = f"{utc_hms()} {text}"
        self.result["events"].append(line)
        print(line, flush=True)

    async def observe(self) -> None:
        async for obs in self.client.observations(decode=False):
            if not obs.state:
                continue
            self.latest = dict(obs.state)
            if not self.watching:
                continue
            if self.reference is None:
                self.reference = dict(obs.state)
                self.event(f"joint reference: {json.dumps(self.reference, sort_keys=True)}")
                continue
            shared = [key for key in self.reference if key in obs.state]
            delta = max((abs(obs.state[key] - self.reference[key]) for key in shared), default=0.0)
            self.max_delta = max(self.max_delta, delta)
            if delta > JOINT_LIMIT:
                await self.halt(f"joint moved {delta:.2f} units (limit {JOINT_LIMIT})", "FAIL")
                return

    async def watch_holders(self) -> None:
        while True:
            await asyncio.sleep(2.0)
            if self.watching and self.baseline is not None:
                new = holders() - self.baseline
                if new:
                    await self.halt(f"new holder {sorted(new)}", "ABORTED-BUSY")
                    return

    async def halt(self, reason: str, status: str) -> None:
        """Stop at once: unload_environment, then close the socket, which also releases the devices."""
        if self.abort is not None:
            return
        self.abort = (reason, status)
        self.watching = False
        self.event(f"HALT ({status}): {reason}; unload_environment, then close")
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self.client.unload_environment(), timeout=10)
            self.event("halt: environment unloaded")
        with contextlib.suppress(Exception):
            await self.client.close()

    async def wait_until(self, predicate, timeout: float, what: str) -> None:
        deadline = time.monotonic() + timeout
        while not predicate():
            if self.abort is not None:
                return
            if time.monotonic() > deadline:
                raise TimeoutError(f"no {what} within {timeout:g} s")
            await asyncio.sleep(0.1)

    async def run(self) -> None:
        client = self.client
        started = time.monotonic()
        await client.connect()
        self.result["timings_s"]["connect"] = round(time.monotonic() - started, 3)
        self.event(f"connected; state.environment={client.state.get('environment')} state.model={client.state.get('model')}")
        self.tasks = [asyncio.create_task(self.observe()), asyncio.create_task(self.watch_holders())]

        started = time.monotonic()
        await client.load_environment(DRAFT, wait_status=True)
        self.result["timings_s"]["load_environment"] = round(time.monotonic() - started, 3)
        self.result["statuses_after_load"] = client.statuses
        self.event(f"environment loaded; statuses={ {n: s.get('state') for n, s in client.statuses.items()} }; failures={client.failures()}")
        if client.failures():
            await self.halt("a device failed on load", "FAIL")
            return

        keys = [cam["key"] for cam in (client.state.get("environment") or {}).get("cameras", [])]
        started = time.monotonic()
        await client.wait_for_frames(keys, timeout_s=20)
        self.result["timings_s"]["frames_ready"] = round(time.monotonic() - started, 3)
        for key in keys:
            frame = client.latest_frame(key)
            if frame is None:
                raise RuntimeError(f"no frame for {key!r}")
            name = "frame_" + re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_") + ".jpg"
            (OUT / name).write_bytes(frame.jpeg)
            self.result["frames"].append({"key": key, "file": name, "bytes": len(frame.jpeg), "backend_timestamp": frame.timestamp})
        self.event(f"frames saved for {keys}")

        self.baseline = holders()
        self.result["holders"]["baseline_after_load"] = sorted(self.baseline)
        self.watching = True
        await self.wait_until(lambda: self.reference is not None, 10, "joint reading")
        if self.abort is not None:
            return
        self.result["follower_state_before"] = self.reference
        self.event(f"follower state before model load recorded ({len(self.reference or {})} joints)")

        started = time.monotonic()
        await asyncio.wait_for(client.load_model(MODEL, BACKEND, DEVICE), timeout=600)
        self.result["timings_s"]["load_model"] = round(time.monotonic() - started, 3)
        self.result["model_after_load"] = client.state.get("model")
        self.result["follower_state_after_load_model"] = self.latest
        self.event(f"model loaded in {self.result['timings_s']['load_model']} s: {client.state.get('model')}")
        if self.abort is not None:
            return

        self.watching = False
        started = time.monotonic()
        await client.unload_environment()
        self.result["timings_s"]["unload_environment"] = round(time.monotonic() - started, 3)
        self.result["model_after_unload"] = client.state.get("model")
        self.event(f"environment unloaded; state.model={client.state.get('model')} state.environment={client.state.get('environment')}")

        started = time.monotonic()
        while follower_held() and time.monotonic() - started < 30:
            await asyncio.sleep(0.5)
        freed = not follower_held()
        waited = round(time.monotonic() - started, 1)
        self.result["port_free_after_unload"] = freed
        self.result["timings_s"]["follower_port_free_after_unload"] = waited if freed else None
        self.event(f"follower port free={freed} after {waited} s")
        if not freed:
            await self.halt("follower port still held 30 s after unload; reload skipped", "ABORTED-BUSY")
            return

        started = time.monotonic()
        await client.load_environment(DRAFT, wait_status=True)
        self.result["timings_s"]["reload_environment"] = round(time.monotonic() - started, 3)
        await client.wait_for_frames(keys, timeout_s=20)
        self.baseline = holders()
        self.result["holders"]["baseline_after_reload"] = sorted(self.baseline)
        self.result["model_after_reload"] = client.state.get("model")
        loaded = client.state.get("model") or {}
        self.result["model_kept_without_reload"] = loaded.get("model_id") == MODEL and loaded.get("backend") == BACKEND
        self.event(f"reloaded; model kept without reload={self.result['model_kept_without_reload']}")
        self.watching = True
        await asyncio.sleep(2.0)
        self.result["follower_state_after_reload"] = self.latest
        self.result["max_joint_delta"] = self.max_delta
        if self.abort is not None:
            return

        self.watching = False
        if client.state and client.state.get("model") is not None:
            await client.unload_model()
        await client.unload_environment()
        await client.close()
        started = time.monotonic()
        while follower_held() and time.monotonic() - started < 30:
            await asyncio.sleep(0.5)
        self.result["port_free_after_cleanup"] = not follower_held()
        self.result["holders"]["after_cleanup"] = sorted(holders())
        self.event(f"cleanup done; follower port free={self.result['port_free_after_cleanup']}")

    async def run_safely(self) -> str:
        status, note = "FAIL", ""
        try:
            await self.run()
            status = self.abort[1] if self.abort is not None else "PASS"
            note = self.abort[0] if self.abort is not None else "completed"
        except Exception as exc:  # any unexpected failure ends the session
            note = f"{type(exc).__name__}: {exc}"
            status = self.abort[1] if self.abort is not None else "FAIL"
            self.event(f"ERROR {note}")
            with contextlib.suppress(Exception):
                await self.client.close()
        finally:
            for task in self.tasks:
                task.cancel()
            with contextlib.suppress(BaseException):
                await asyncio.gather(*self.tasks, return_exceptions=True)
        client = self.client
        self.result.update(
            status=status,
            note=note,
            finished_utc=utc_hms(),
            statuses_final=client.statuses,
            errors=list(client.errors),
            unhandled=list(client.unhandled),
            failures=client.failures(),
            follower_state_final=self.latest,
        )
        (OUT / "probe_results.json").write_text(json.dumps(self.result, indent=2, default=str))
        return status


async def probe_main() -> str:
    return await Probe().run_safely()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    reason = gate()
    if reason is not None:
        print(f"{utc_hms()} ABORTED-BUSY gate: {reason}", flush=True)
        (OUT / "probe_results.json").write_text(json.dumps({"status": "ABORTED-BUSY", "note": f"gate: {reason}"}, indent=2))
        return 2
    status = asyncio.run(probe_main())
    print(f"{utc_hms()} STATUS {status}", flush=True)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
