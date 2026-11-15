"""Live demo dashboard server for the inspection loop (stdlib only).

Reads the run artefacts written by app/loop (LOOP-CONTRACT section 5):

  GET  /                          the page (static/index.html)
  GET  /static/<file>             static assets (app.js, style.css)
  GET  /api/events                Server-Sent Events: a "snapshot" event with the
                                  state derived from <runs>/current/events.jsonl,
                                  pushed whenever new complete lines appear
  GET  /api/state                 one-shot snapshot (same JSON as the SSE event)
  GET  /run/current/<path>        an image (.jpg/.jpeg/.png) inside the current run dir only
  POST /api/control               {"cmd": "start|next|stop|quit"} -> writes control.json
                                  (atomic replace, id strictly increasing) in the
                                  current run dir

Nothing else is ever written. Only loopback hosts are accepted.

Run (PYTHONPATH must contain R/app):
  python -m dashboard.server --runs R/logs/loop --port 8090 --host 127.0.0.1
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit

from dashboard.intel import IntelSources
from dashboard.policy import POLICY_KINDS, PolicyState

PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
DEFAULT_RUNS = PACKAGE_DIR.parents[1] / "logs" / "loop"

EVENTS_FILE = "events.jsonl"
CONTROL_FILE = "control.json"
CURRENT_LINK = "current"
CONTROL_CMDS = ("start", "next", "stop", "quit")
LOOPBACK_HOSTS = ("127.0.0.1", "localhost")
HISTORY_LEN = 8
MAX_BODY_BYTES = 512
MAX_PARTIAL_LINE_BYTES = 4 * 1024 * 1024
HEAD_BYTES = 256
SSE_POLL_S = 0.2
SSE_KEEPALIVE_S = 5.0

CSP = (
    "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
    "connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; "
    "frame-ancestors 'none'"
)
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".json": "application/json; charset=utf-8",
    ".jsonl": "application/x-ndjson; charset=utf-8",
    ".csv": "text/csv; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
}
STAGE_BY_STATE = {
    "IDLE": "IDLE",
    "WAIT_BRICK": "WAIT_BRICK",
    "HOME": "HOME",
    "SCAN": "SCAN",
    "TRIAGE": "TRIAGE",
    "ACT": "ACT",
    "VERIFY": "VERIFY",
    "S2": "S2_BLUE",
    "S3": "S3_GREEN",
    "CYCLE_DONE": "CYCLE_DONE",
    "HOLD": "HOLD",
    "STOPPED": "STOPPED",
}
LABEL_COUNTERS = {"GOOD": "good", "DEFECTIVE": "defective", "UNCERTAIN": "uncertain"}
CAMERA_VIEWS = {"243722064316": "close", "243622061421": "top"}  # RealSense serials (OD-9)
VIEW_ORDER = {"close": 0, "top": 1}
SORT_PLATES = {"S2": "BLUE", "S3": "GREEN", "BLUE": "BLUE", "GREEN": "GREEN",
               "GOOD": "BLUE", "DEFECTIVE": "GREEN", "UNCERTAIN": "GREEN"}
BACKEND_NAMES = {"openvino": "OpenVINO"}
RUN_IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def utc_now_iso() -> str:
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")
    return stamp.replace("+00:00", "Z")


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _as_num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _as_text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _json_safe(value: Any) -> Any:
    """A copy that JSON can carry: every NaN or Infinity becomes null (the stream keeps going)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_segments(rel: str) -> list[str] | None:
    """Split a decoded relative path; None when it could escape its root."""
    if not rel or rel.startswith("/") or "\x00" in rel or "\\" in rel:
        return None
    segments = rel.split("/")
    for segment in segments:
        if not segment or segment.startswith("."):
            return None
    return segments


def _run_url(run_id: str | None, rel: Any) -> str | None:
    if run_id is None or not isinstance(rel, str):
        return None
    segments = _safe_segments(rel)
    if segments is None:
        return None
    return "/run/" + CURRENT_LINK + "/" + "/".join(quote(s) for s in segments)


def _plate(label: str, route: str | None) -> str:
    if route == "S2":
        return "BLUE"
    if route == "S3":
        return "GREEN"
    return "BLUE" if label == "GOOD" else "GREEN"


def read_control(run_dir: Path | None) -> dict[str, Any] | None:
    """The {cmd, id} last written to control.json, or None if absent or unreadable."""
    if run_dir is None:
        return None
    try:
        with open(run_dir / CONTROL_FILE, "rb") as handle:
            data = json.loads(handle.read().decode("utf-8"))
    except (OSError, ValueError, RecursionError):
        return None
    if not isinstance(data, dict):
        return None
    control_id = _as_int(data.get("id"))
    if control_id is None:
        return None
    return {"cmd": _as_text(data.get("cmd")), "id": control_id}


def max_ack_id(run_dir: Path) -> int:
    """Largest control id the orchestrator acknowledged in events.jsonl (type control)."""
    best = 0
    try:
        with open(run_dir / EVENTS_FILE, "rb") as handle:
            for raw in handle:
                if b'"control"' not in raw:
                    continue
                try:
                    record = json.loads(raw)
                except (ValueError, RecursionError):
                    continue
                if isinstance(record, dict) and record.get("type") == "control":
                    cid = _as_int(record.get("id"))
                    if cid is not None and cid > best:
                        best = cid
    except OSError:
        pass
    return best


def _cam_url(cams: list[dict[str, Any]], key: str) -> str | None:
    """First camera URL for key, preferring the top view."""
    ordered = sorted(cams, key=lambda c: c.get("view") != "top")
    return next((c[key] for c in ordered if c.get(key)), None)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write via a temp file in the same directory and os.replace, so readers never see a partial file."""
    data = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=".control-", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


class RunLocator:
    """Resolves the `current` alias and run directories under the runs root."""

    def __init__(self, runs_root: Path) -> None:
        self.runs_root = Path(runs_root)

    def _root_real(self) -> Path:
        return Path(os.path.realpath(self.runs_root))

    def current_dir(self) -> Path | None:
        link = self.runs_root / CURRENT_LINK
        if not os.path.isdir(link):
            return None
        real = Path(os.path.realpath(link))
        return real if real.parent == self._root_real() else None


class EventTailer:
    """Reads complete JSON lines from events.jsonl; a partial last line waits for its newline."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.offset = 0
        self.pending = b""
        self.bad_lines = 0
        self._ident: tuple[int, int] | None = None
        self._head = b""  # first bytes already consumed; a rewrite in place shows up as a mismatch

    def _head_still_matches(self) -> bool:
        if not self._head:
            return True
        try:
            with open(self.path, "rb") as handle:
                return handle.read(len(self._head)) == self._head
        except OSError:
            return False

    def poll(self) -> tuple[bool, list[dict[str, Any]]]:
        """Return (reset, new_events). reset is True when the file was replaced, truncated or rewritten."""
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return False, []
        reset = False
        ident = (st.st_dev, st.st_ino)
        if self._ident is not None and (
            ident != self._ident or st.st_size < self.offset or not self._head_still_matches()
        ):
            self.offset = 0
            self.pending = b""
            self._head = b""
            reset = True
        self._ident = ident
        if st.st_size == self.offset:
            return reset, []
        try:
            with open(self.path, "rb") as handle:
                handle.seek(self.offset)
                chunk = handle.read()
        except OSError:  # removed between stat and open: try again on the next poll
            return reset, []
        if len(self._head) < HEAD_BYTES:
            self._head = (self._head + chunk)[:HEAD_BYTES]
        self.offset += len(chunk)
        *lines, self.pending = (self.pending + chunk).split(b"\n")
        if len(self.pending) > MAX_PARTIAL_LINE_BYTES:
            self.pending = b""
            self.bad_lines += 1
        events: list[dict[str, Any]] = []
        for line in lines:
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except (ValueError, RecursionError):
                self.bad_lines += 1
                continue
            if isinstance(record, dict):
                events.append(record)
            else:
                self.bad_lines += 1
        return reset, events


class RunReducer:
    """Folds the events of one run into the snapshot the page renders."""

    def __init__(self, run_id: str | None) -> None:
        self.run_id = run_id
        self.mode: str | None = None
        self.started_t: str | None = None
        self.ended: dict[str, Any] | None = None
        self.state: str | None = None
        self.state_t: str | None = None
        self.cycle = 0
        self.skill: dict[str, Any] | None = None
        self.last_skill_end: dict[str, Any] | None = None
        self.inspection: dict[str, Any] | None = None
        self.history: list[dict[str, Any]] = []
        self.inspected = 0
        self.derived = {"good": 0, "defective": 0, "uncertain": 0}
        self.stats: dict[str, Any] | None = None
        self.heartbeat_t: str | None = None
        self.error: dict[str, Any] | None = None
        self.acked: dict[str, Any] | None = None
        self.events = 0
        self.last_event_t: str | None = None
        self.live_source: str | None = None
        self.backend: str | None = None
        self.cameras: dict[str, Any] | None = None
        self.handover: dict[str, Any] | None = None
        self.sort: dict[str, Any] | None = None
        self.align: dict[str, Any] | None = None
        self.policy = PolicyState(lambda rel: _run_url(run_id, rel))
        self.error_count = 0

    def apply(self, ev: dict[str, Any]) -> None:
        self.events += 1
        t = _as_text(ev.get("t"))
        if t is not None:
            self.last_event_t = t
        kind = ev.get("type")
        if kind == "run_start":
            self.mode = _as_text(ev.get("mode"))
            self.started_t = t
            self._read_config(ev.get("config"))
            self.policy.read_config(ev.get("config"))
        elif kind == "state":
            self.state = _as_text(ev.get("state"))
            self.state_t = t
            cycle = _as_int(ev.get("cycle"))
            if cycle is not None:
                self.cycle = cycle
        elif kind == "skill_start":
            self.skill = {
                "name": _as_text(ev.get("skill")),
                "model": _as_text(ev.get("model")),
                "cycle": _as_int(ev.get("cycle")),
                "t": t,
            }
        elif kind == "skill_end":
            self.last_skill_end = {
                "name": _as_text(ev.get("skill")),
                "ok": ev.get("ok") is True,
                "duration_s": _as_num(ev.get("duration_s")),
                "reason": _as_text(ev.get("reason")),
                "t": t,
            }
            self.skill = None
        elif kind == "inspection":
            self._add_inspection(ev, t)
        elif kind == "stats":
            self.stats = {
                "cycles": _as_int(ev.get("cycles")),
                "good": _as_int(ev.get("good")),
                "defective": _as_int(ev.get("defective")),
                "uncertain": _as_int(ev.get("uncertain")),
                "mean_cycle_s": _as_num(ev.get("mean_cycle_s")),
            }
        elif kind == "error":
            self.error_count += 1
            self.error = {
                "where": _as_text(ev.get("where")) or "",
                "message": _as_text(ev.get("message")) or "",
                "t": t,
            }
        elif kind in POLICY_KINDS:
            self.policy.apply(kind, ev, t, self.cycle)
        elif kind == "heartbeat":
            self.heartbeat_t = t
        elif kind == "run_end":
            self.ended = {"reason": _as_text(ev.get("reason")) or "", "t": t}
        elif kind == "control":
            self.acked = {"id": _as_int(ev.get("id")), "cmd": _as_text(ev.get("cmd")), "t": t}
        elif kind == "inspect_cameras":
            self._add_cameras(ev, t)
        elif kind == "handover":
            self.handover = {
                "cycle": self.cycle,
                "phase": _as_text(ev.get("phase")),
                "duration_s": _as_num(ev.get("duration_s")),
                "model_kept": ev.get("model_kept") if isinstance(ev.get("model_kept"), bool) else None,
                "t": t,
            }
        elif kind == "align":
            self.align = {
                "cycle": self.cycle,
                "iteration": _as_int(ev.get("iteration")),
                "dx": _as_num(ev.get("dx")),
                "dy": _as_num(ev.get("dy")),
                "correction": _as_num(ev.get("correction")),
                "t": t,
            }
        elif kind == "sort_start":
            route = _as_text(ev.get("route"))
            self.sort = {"route": route, "plate": SORT_PLATES.get((route or "").upper()),
                         "cycle": self.cycle, "active": True, "ok": None, "duration_s": None, "t": t}
            self.align = None
        elif kind == "sort_end":
            base = dict(self.sort) if self.sort else {"route": None, "plate": None, "cycle": self.cycle}
            base.update({"active": False, "ok": ev.get("ok") is True,
                         "duration_s": _as_num(ev.get("duration_s")), "t": t})
            self.sort = base

    def _read_config(self, cfg: Any) -> None:
        if not isinstance(cfg, dict):
            return
        self.live_source = _as_text(cfg.get("live_source"))
        detector = cfg.get("detector") if isinstance(cfg.get("detector"), dict) else {}
        backend = _as_text(detector.get("backend"))
        # the policy emitter names its skills per plate (plates); older configs name them under skills
        skills = cfg.get("plates") if isinstance(cfg.get("plates"), dict) else (
            cfg.get("skills") if isinstance(cfg.get("skills"), dict) else {})
        for skill in skills.values():
            if backend is None and isinstance(skill, dict):
                backend = _as_text(skill.get("backend"))
        self.backend = BACKEND_NAMES.get(backend.lower(), backend) if backend else None

    def _add_cameras(self, ev: dict[str, Any], t: str | None) -> None:
        raw = ev.get("cameras")
        if isinstance(raw, dict):
            items = list(raw.items())
        elif isinstance(raw, list):
            items = [(None, item) for item in raw]
        else:
            items = []
        cams: list[dict[str, Any]] = []
        for key, item in items:
            if not isinstance(item, dict):
                continue
            name = (_as_text(item.get("camera")) or _as_text(item.get("key"))
                    or _as_text(item.get("name")) or key or "camera")
            view = (next((v for k, v in CAMERA_VIEWS.items() if name.endswith(k)), None)
                    or _as_text(item.get("view")) or name)
            reasons = item.get("reasons")
            cams.append({
                "view": view,
                "camera": name,
                "label": (_as_text(item.get("label")) or "").upper(),
                "score": _as_num(item.get("score")),
                "threshold": _as_num(item.get("threshold")),
                "reasons": [str(r) for r in reasons] if isinstance(reasons, list) else [],
                "frame_url": _run_url(self.run_id, item.get("frame")),
                "crop_url": _run_url(self.run_id, item.get("crop")),
            })
        cams.sort(key=lambda c: VIEW_ORDER.get(c["view"], 2))
        self.cameras = {"cycle": _as_int(ev.get("cycle")), "cameras": cams, "t": t}

    def _cameras_for(self, cycle: int | None) -> list[dict[str, Any]]:
        if self.cameras is None or self.cameras.get("cycle") != cycle:
            return []
        return list(self.cameras["cameras"])

    def _sort_plate(self) -> str | None:
        if self.sort and self.sort.get("cycle") == self.cycle and self.sort.get("plate"):
            return self.sort["plate"]
        if self.inspection and self.inspection.get("cycle") == self.cycle:
            return self.inspection.get("plate")
        return None

    def _add_inspection(self, ev: dict[str, Any], t: str | None) -> None:
        label = (_as_text(ev.get("label")) or "").upper()
        route = _as_text(ev.get("route"))
        reasons = ev.get("reasons")
        cycle = _as_int(ev.get("cycle"))
        if ev.get("cameras") is not None:  # policy inspections (still_hires) carry their per-camera verdicts
            self._add_cameras({"cycle": cycle, "cameras": ev.get("cameras")}, t)
        cams = self._cameras_for(cycle)
        record = {
            "cycle": cycle,
            "label": label,
            "score": _as_num(ev.get("score")),
            "threshold": _as_num(ev.get("threshold")),
            "reasons": [str(r) for r in reasons] if isinstance(reasons, list) else [],
            "geometry": _json_safe(ev.get("geometry")) if isinstance(ev.get("geometry"), dict) else {},
            "device": _as_text(ev.get("device")),
            "backend": _as_text(ev.get("backend")) or self.backend,
            "latency_ms": _as_num(ev.get("latency_ms")),
            "route": route,
            "plate": _plate(label, route),
            "safe": label == "UNCERTAIN",
            "frame_url": _run_url(self.run_id, ev.get("frame")) or _cam_url(cams, "frame_url"),
            "crop_url": _run_url(self.run_id, ev.get("crop")) or _cam_url(cams, "crop_url"),
            "cameras": cams,
            "t": t,
        }
        self.inspection = record
        self.history.append(record)
        del self.history[:-HISTORY_LEN]
        self.inspected += 1
        counter = LABEL_COUNTERS.get(label)
        if counter is not None:
            self.derived[counter] += 1
        self.policy.note_inspection(ev, t, self.cycle)

    def _current(self, part: dict[str, Any] | None) -> dict[str, Any] | None:
        """A handover, align or sort record counts only for the cycle that reported it."""
        return part if part is not None and part.get("cycle") == self.cycle else None

    def stage(self) -> str:
        if self.run_id is None:
            return "NO_RUN"
        if self.ended is not None:
            return "RUN_ENDED"
        if self.state is None:
            return "STARTING"
        if self.state == "S1":
            done = self.inspection is not None and self.inspection.get("cycle") == self.cycle
            return "INSPECT" if done else "S1_PICK"
        if self.state in ("DECIDE", "INSPECT"):
            return "INSPECT"
        if self.state in ("HANDOVER", "HANDOVER_OUT", "HANDOVER_IN"):
            return "HANDOVER"
        if self.state == "SORT":
            return {"BLUE": "SORT_BLUE", "GREEN": "SORT_GREEN"}.get(self._sort_plate() or "", "SORT")
        return STAGE_BY_STATE.get(self.state, self.state)

    def counts(self) -> dict[str, Any]:
        if self.stats is not None:
            return {"source": "stats", **self.stats}
        return {
            "source": "derived",
            "cycles": self.inspected,
            **self.derived,
            "mean_cycle_s": None,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "has_run": self.run_id is not None,
            "run_id": self.run_id,
            "mode": self.mode,
            "started_t": self.started_t,
            "ended": self.ended,
            "state": self.state,
            "stage": self.stage(),
            "state_t": self.state_t,
            "cycle": self.cycle,
            "skill": self.skill,
            "last_skill_end": self.last_skill_end,
            "inspection": self.inspection,
            "history": self.history,
            "live_source": self.live_source,
            "backend": self.backend,
            "handover": self._current(self.handover),
            "sort": self._current(self.sort),
            "align": self._current(self.align),
            "counts": self.counts(),
            "heartbeat_t": self.heartbeat_t,
            "error": self.error,
            "acked": self.acked,
            "events": self.events,
            "last_event_t": self.last_event_t,
            "errors": self.error_count,
            "policy": self.policy.snapshot(self.cycle),
        }


class LiveSession:
    """Follows the `current` run for one client; switches when the symlink moves."""

    def __init__(self, locator: RunLocator, intel: IntelSources | None = None) -> None:
        self.locator = locator
        self.intel = intel
        self.run_dir: Path | None = None
        self.tailer: EventTailer | None = None
        self.reducer = RunReducer(None)
        self.sent = False

    def step(self) -> bool:
        """Read new events; True when the snapshot changed."""
        current = self.locator.current_dir()
        changed = False
        if current != self.run_dir:
            self.run_dir = current
            self.tailer = EventTailer(current / EVENTS_FILE) if current is not None else None
            self.reducer = RunReducer(current.name if current is not None else None)
            changed = True
        if self.tailer is not None:
            reset, events = self.tailer.poll()
            if reset:
                self.reducer = RunReducer(self.run_dir.name)
                changed = True
            for event in events:
                self.reducer.apply(event)
            if events:
                changed = True
        return changed

    def snapshot(self) -> dict[str, Any]:
        snap = self.reducer.snapshot()
        snap["server_t"] = utc_now_iso()
        snap["control"] = {
            "issued": read_control(self.run_dir),
            "bad_lines": self.tailer.bad_lines if self.tailer is not None else 0,
        }
        snap["intel"] = self.intel.snapshot() if self.intel is not None else None
        self.sent = True
        return snap


class ControlWriter:
    """Writes control.json with an id strictly greater than any id this server issued or found."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_id: dict[str, int] = {}

    def issue(self, run_dir: Path, cmd: str) -> dict[str, Any]:
        with self._lock:
            on_disk = read_control(run_dir)
            base = max(on_disk["id"] if on_disk is not None else 0,
                       self._last_id.get(str(run_dir), 0), max_ack_id(run_dir))
            payload = {"cmd": cmd, "id": base + 1}
            atomic_write_json(run_dir / CONTROL_FILE, payload)
            self._last_id[str(run_dir)] = base + 1
            return payload


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, runs_root: Path, host: str = "127.0.0.1", port: int = 8090,
                 telemetry_dir: Path | None = None, bench_path: Path | None = None) -> None:
        if host not in LOOPBACK_HOSTS:
            raise ValueError(f"host must be a loopback name, got {host!r}")
        super().__init__((host, port), DashboardHandler)
        bound_port = self.server_address[1]
        self.allowed_hosts = {f"{name}:{bound_port}" for name in LOOPBACK_HOSTS}
        self.allowed_origins = {f"http://{host_port}" for host_port in self.allowed_hosts}
        self.locator = RunLocator(runs_root)
        self.control = ControlWriter()
        self.intel = IntelSources(telemetry_dir, bench_path)


class DashboardHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server_version = "LoopDashboard/1.0"
    sys_version = ""
    server: DashboardServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """The operator console stays quiet; the page shows the state."""

    def do_GET(self) -> None:
        if not self._host_allowed():
            self._text(403, "host refused")
            return
        path = unquote(urlsplit(self.path).path)
        if path in ("/", "/index.html"):
            self._static_file("index.html")
        elif path == "/api/events":
            self._events()
        elif path == "/api/state":
            session = LiveSession(self.server.locator, self.server.intel)
            session.step()
            self._json(200, session.snapshot())
        elif path.startswith("/static/"):
            self._static_file(path[len("/static/"):])
        elif path.startswith("/run/"):
            self._run_file(path[len("/run/"):])
        else:
            self._text(404, "not found")

    def do_POST(self) -> None:
        if not self._host_allowed():
            self._text(403, "host refused")
            return
        if urlsplit(self.path).path != "/api/control":
            self._text(404, "not found")
            return
        self._control()

    def _host_allowed(self) -> bool:
        return self.headers.get("Host") in self.server.allowed_hosts

    def _static_file(self, name: str) -> None:
        segments = _safe_segments(name)
        if segments is None or len(segments) != 1:
            self._text(403, "path refused")
            return
        root = Path(os.path.realpath(STATIC_DIR))
        target = Path(os.path.realpath(root / segments[0]))
        if target.parent != root or not target.is_file():
            self._text(404, "not found")
            return
        self._file(target)

    def _run_file(self, rest: str) -> None:
        segments = _safe_segments(rest)
        if segments is None:
            self._text(403, "path refused")
            return
        if len(segments) < 2 or segments[0] != CURRENT_LINK:
            self._text(404, "not found")
            return
        run_dir = self.server.locator.current_dir()
        if run_dir is None:
            self._text(404, "no current run")
            return
        target = Path(os.path.realpath(run_dir.joinpath(*segments[1:])))
        if not _inside(target, run_dir) or not target.is_file() or target.suffix.lower() not in RUN_IMAGE_EXTS:
            self._text(404, "not found")
            return
        self._file(target)

    def _file(self, path: Path) -> None:
        try:
            body = path.read_bytes()
        except OSError:
            self._text(404, "not found")
            return
        self._send(200, body, CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream"))

    def _events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", CSP)
        self.end_headers()
        session = LiveSession(self.server.locator, self.server.intel)
        last_write = 0.0
        try:
            self.wfile.write(b"retry: 2000\n\n")
            self.wfile.flush()
            while True:
                changed = session.step()
                now = time.monotonic()
                if changed or not session.sent:
                    self._sse("snapshot", session.snapshot())
                    last_write = now
                elif now - last_write >= SSE_KEEPALIVE_S:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    last_write = now
                time.sleep(SSE_POLL_S)
        except OSError:
            return

    def _sse(self, event: str, obj: dict[str, Any]) -> None:
        data = json.dumps(_json_safe(obj), separators=(",", ":"), allow_nan=False)
        self.wfile.write(f"event: {event}\ndata: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _control(self) -> None:
        origin = self.headers.get("Origin")
        if origin is not None and origin not in self.server.allowed_origins:
            self._json(403, {"ok": False, "error": "origin refused"})
            return
        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if content_type != "application/json":
            self._json(415, {"ok": False, "error": "Content-Type must be application/json"})
            return
        try:
            length = int(self.headers.get("Content-Length") or "")
        except ValueError:
            length = -1
        if length <= 0 or length > MAX_BODY_BYTES:
            self._json(400, {"ok": False, "error": "Content-Length missing or out of range"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except ValueError:
            self._json(400, {"ok": False, "error": "body is not JSON"})
            return
        cmd = payload.get("cmd") if isinstance(payload, dict) else None
        if cmd not in CONTROL_CMDS:
            self._json(400, {"ok": False, "error": "cmd must be start, next, stop or quit"})
            return
        run_dir = self.server.locator.current_dir()
        if run_dir is None:
            self._json(409, {"ok": False, "error": "no current run"})
            return
        try:
            written = self.server.control.issue(run_dir, cmd)
        except OSError as exc:
            self._json(500, {"ok": False, "error": f"control write failed: {exc}"})
            return
        self._json(200, {"ok": True, "run_id": run_dir.name, **written})

    def _json(self, status: int, obj: dict[str, Any]) -> None:
        body = json.dumps(_json_safe(obj), allow_nan=False) + "\n"
        self._send(status, body.encode("utf-8"), "application/json; charset=utf-8")

    def _text(self, status: int, message: str) -> None:
        self._send(status, (message + "\n").encode("utf-8"), "text/plain; charset=utf-8")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", CSP)
        self.end_headers()
        self.wfile.write(body)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live demo dashboard for the inspection loop.")
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS,
                        help="runs root that holds the `current` symlink (default: <R>/logs/loop)")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="127.0.0.1", help="loopback only: 127.0.0.1 or localhost")
    parser.add_argument("--telemetry-dir", type=Path, default=None,
                        help="folder of YYYYMMDD.jsonl telemetry (default: the runs root's sibling telemetry/)")
    parser.add_argument("--bench", type=Path, default=None,
                        help="bench.json of the benchmark run (P2-B-OPT); omitted = no benchmark table")
    args = parser.parse_args(argv)
    telemetry_dir = args.telemetry_dir if args.telemetry_dir is not None else args.runs.parent / "telemetry"
    try:
        server = DashboardServer(args.runs, args.host, args.port, telemetry_dir, args.bench)
    except (ValueError, OSError) as exc:
        print(f"dashboard: cannot start: {exc}", file=sys.stderr)
        return 2
    print(f"dashboard on http://{args.host}:{server.server_address[1]}/ watching {args.runs}", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
