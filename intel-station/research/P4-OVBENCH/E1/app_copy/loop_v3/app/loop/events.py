"""Run directory, JSON-safe events.jsonl, frame files and operator control (LOOP-CONTRACT sections 5, 7 and 8).

Layout: <logs_root>/<run_id>/ with events.jsonl, frames/ and control.json, and the symlink <logs_root>/current pointing at
the newest run. run_id is the UTC time as yyyymmdd-HHMMSS. Every event line carries t (UTC ISO 8601, ms), mono (monotonic
seconds), cycle (0 before the first brick) and type. NaN and infinities become null, and every line is written with allow_nan=False, so the dashboard
always parses strict JSON. Standard library only; write_jpeg_atomic imports Pillow when it is called.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

RUN_ID_PATTERN = re.compile(r"^\d{8}-\d{6}$")
EVENT_TYPES = frozenset(
    {
        # scripted mode (sections 5 and 7) and the candidate pick servo
        "run_start", "state", "skill_start", "skill_end", "inspection", "inspect_cameras", "handover", "align",
        "sort_start", "sort_end", "stats", "error", "heartbeat", "run_end", "control",
        # policy mode (section 8, OD-12)
        "scan", "triage", "decision", "policy_run", "verify",
    }
)
COMMANDS = ("start", "pause", "resume", "next", "stop", "quit")


def utc_iso_ms(now: dt.datetime | None = None) -> str:
    """UTC timestamp in ISO 8601 with milliseconds and a Z suffix."""
    moment = (now or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def new_run_id(now: dt.datetime | None = None) -> str:
    """The run id for a start time: UTC yyyymmdd-HHMMSS."""
    return (now or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")


def json_safe(value: Any) -> Any:
    """NaN and infinities become null, tuples become lists, mapping keys become strings, numpy values become Python values."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):  # numpy scalars and arrays
        return json_safe(value.tolist())
    return str(value)  # paths and anything else


def write_jpeg_atomic(path: Path, rgb: Any, quality: int = 85) -> None:
    """Write an RGB uint8 array as JPEG, atomically (tmp file, then rename). Imports Pillow and numpy when called."""
    import numpy as np
    from PIL import Image

    tmp = path.with_name(path.name + ".tmp")
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(tmp, format="JPEG", quality=quality)
    os.replace(tmp, path)


class RunDir:
    """One run's artefacts. Creating it moves the ``current`` symlink to the new run."""

    def __init__(self, logs_root: Path, run_id: str | None = None) -> None:
        self.logs_root = Path(logs_root)
        self.run_id = run_id or new_run_id()
        self.cycle = 0  # every event carries it (LOOP-CONTRACT section 9); the policy cycle keeps it current
        if not RUN_ID_PATTERN.match(self.run_id):
            raise ValueError(f"run_id must be yyyymmdd-HHMMSS, got {self.run_id!r}")
        self.logs_root.mkdir(parents=True, exist_ok=True)
        self.path = self.logs_root / self.run_id
        self.path.mkdir(exist_ok=False)
        (self.path / "frames").mkdir()
        self._file = (self.path / "events.jsonl").open("a", encoding="utf-8")
        self._lock = threading.Lock()  # the event loop and a worker thread both emit
        self.emitted: dict[str, int] = {}
        self._point_current()

    def _point_current(self) -> None:
        link = self.logs_root / "current"
        staging = self.logs_root / f".current-{os.getpid()}"
        with contextlib.suppress(FileNotFoundError):
            staging.unlink()
        os.symlink(self.run_id, staging)
        os.replace(staging, link)

    def emit(self, event_type: str, **fields: Any) -> dict[str, Any]:
        """Append one event line and return the record that was written."""
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown event type {event_type!r}")
        record = json_safe({"t": utc_iso_ms(), "mono": time.monotonic(), "type": event_type, "cycle": self.cycle, **fields})
        line = json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
        with self._lock:
            self._file.write(line)
            self._file.flush()
            self.emitted[event_type] = self.emitted.get(event_type, 0) + 1
        return record

    def save_bytes(self, relative: str, data: bytes) -> str:
        """Write a frame atomically under the run directory; returns the path relative to the run directory."""
        target = self.path / relative
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, target)
        return relative

    def close(self) -> None:
        self._file.close()


class ControlFile:
    """Reads the dashboard's control.json: {"cmd": "start|pause|resume|next|stop|quit", "id": <increasing int>}.

    The dashboard replaces the file atomically. Only ids above the last one seen are returned, so a repeated or older
    write is ignored. Two writes between two polls collapse into the newer one.
    """

    def __init__(self, path: Path, last_id: int = 0) -> None:
        self.path = Path(path)
        self.last_id = last_id

    def poll(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            return []
        if not isinstance(data, dict):
            return []
        command_id = data.get("id")
        if isinstance(command_id, bool) or not isinstance(command_id, int) or command_id <= self.last_id:
            return []
        self.last_id = command_id
        return [{"cmd": str(data.get("cmd", "")), "id": command_id}]


class ScheduledControl:
    """Dry-run control: schedule maps a poll number to a command name. Ids increase with each command."""

    def __init__(self, schedule: dict[int, str] | None = None) -> None:
        self._schedule = dict(schedule or {})
        self._polls = 0
        self._id = 0

    def poll(self) -> list[dict[str, Any]]:
        self._polls += 1
        name = self._schedule.get(self._polls)
        if name is None:
            return []
        self._id += 1
        return [{"cmd": name, "id": self._id}]
