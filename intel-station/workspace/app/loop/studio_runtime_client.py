"""Headless client for a Physical AI Studio runtime session.

Drives a trained policy through Studio's own runtime session socket, the same one the
Studio UI uses: ``/api/projects/{project_id}/runtime/ws``. One socket is one session.
Closing the socket ends the session and releases the robots and cameras it holds.

Protocol source (repos/physical-ai-studio, branch junction2026, HEAD 285924d):
application/backend/src/api/runtime_ws.py, application/backend/src/schemas/runtime_session.py,
application/backend/src/runtime_session/session.py and the UI client in
application/ui/src/features/runtime-session/runtime-session-provider.tsx.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import struct
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode, urlsplit, urlunsplit

import numpy as np
from PIL import Image
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterable, Mapping
    from types import TracebackType

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_HZ",
    "Frame",
    "MAX_MESSAGE_BYTES",
    "Observation",
    "StudioCommandError",
    "StudioConnectionError",
    "StudioError",
    "StudioRuntimeClient",
    "decode_jpeg_rgb",
    "environment_draft",
    "model_control",
    "parse_observation",
    "ws_url",
]

DEFAULT_BASE_URL = "http://127.0.0.1:7860"
"""Studio's backend on the laptop (shared/FACTS-LIVE.md, line 4)."""

RUNTIME_WS_PATH = "/api/projects/{project_id}/runtime/ws"
DEFAULT_HZ = 20.0
"""Observation rate; the Studio UI sends the same value (runtime-session-provider.tsx:132)."""

COMMAND_TIMEOUT_S = 660.0
"""Backend bound per command is 600 s (runtime_ws.py:53); the margin covers the reply."""

STATE_TIMEOUT_S = 15.0
CONNECT_TIMEOUT_S = 30.0
CLOSE_TIMEOUT_S = 15.0
FRAME_TIMEOUT_S = 10.0
OBSERVATION_QUEUE_SIZE = 8
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
"""One observation carries every camera's JPEG, so the library default of 1 MiB is too small."""

_END = object()


class StudioError(Exception):
    """Base class of the errors this client raises."""


class StudioCommandError(StudioError):
    """Studio refused a command; ``error_code`` is Studio's code, such as ``no_environment``."""

    def __init__(self, message: str, error_code: str, parts: list[dict[str, Any]] | None = None) -> None:
        super().__init__(f"{error_code}: {message}")
        self.message = message
        self.error_code = error_code
        self.parts = parts or []


class StudioConnectionError(StudioError):
    """The socket closed or the session ended before a command or wait could finish."""


@dataclass(frozen=True, slots=True)
class Observation:
    """One binary observation: the joints, and the camera frames that are new since the previous one."""

    timestamp: float
    """Backend monotonic clock in seconds, the clock the store stamps samples with. Not wall time."""
    state: dict[str, float] | None
    """Measured joints by name, such as ``shoulder_pan.pos``."""
    action: dict[str, float] | None
    """Commanded joints by name."""
    frames: dict[str, np.ndarray]
    """Camera key to an RGB ``uint8`` array of shape (height, width, 3). Empty when decoding is off."""
    jpegs: dict[str, bytes]
    """The JPEG bytes of the same frames, as sent."""
    frame_timestamps: dict[str, float] = field(default_factory=dict)
    """Camera key to the capture time of its frame in this message (schemas/runtime_session.py:283-287)."""


@dataclass(frozen=True, slots=True)
class Frame:
    """The latest JPEG of one camera, as the backend last sent it."""

    key: str
    timestamp: float
    """Backend monotonic clock in seconds when the frame was captured."""
    jpeg: bytes

    def rgb(self) -> np.ndarray:
        """The frame decoded to an RGB ``uint8`` array of shape (height, width, 3)."""
        return decode_jpeg_rgb(self.jpeg)


def _uuid_text(value: str | uuid.UUID) -> str:
    """A UUID in the lowercase hyphenated form the backend's JSON uses."""
    return str(uuid.UUID(str(value)))


def ws_url(base_url: str, project_id: str, hz: float = DEFAULT_HZ) -> str:
    """The runtime socket URL for ``project_id`` at observation rate ``hz``."""
    parts = urlsplit(base_url)
    scheme = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}.get(parts.scheme)
    if scheme is None:
        raise ValueError(f"base_url must use http, https, ws or wss, not {parts.scheme!r}")
    path = parts.path.rstrip("/") + RUNTIME_WS_PATH.format(project_id=project_id)
    return urlunsplit((scheme, parts.netloc, path, urlencode({"hz": float(hz)}), ""))


def parse_observation(data: bytes) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Split one binary observation into its JSON header and the JPEG bytes of each frame.

    The layout is ``[u32 big-endian header length][header JSON][JPEG bytes, in header["frames"] order]``
    (runtime_ws.py:218-221). Each frame's size is in its header entry (schemas/runtime_session.py:283-287).
    The message must end exactly after the last frame.

    Raises:
        ValueError: The length prefix, header or a frame runs past the end of the message, a key repeats,
            or bytes follow the last frame.
    """
    if len(data) < 4:
        raise ValueError("observation shorter than its 4-byte length prefix")
    (header_length,) = struct.unpack(">I", data[:4])
    header_end = 4 + header_length
    if header_end > len(data):
        raise ValueError("observation header runs past the end of the message")
    header: dict[str, Any] = json.loads(data[4:header_end])
    jpegs: dict[str, bytes] = {}
    offset = header_end
    for frame in header.get("frames", []):
        key = str(frame["key"])
        if key in jpegs:
            raise ValueError(f"frame {key!r} appears twice in one observation")
        end = offset + int(frame["size"])
        if end > len(data):
            raise ValueError(f"frame {key!r} runs past the end of the message")
        jpegs[key] = data[offset:end]
        offset = end
    if offset != len(data):
        raise ValueError(f"{len(data) - offset} bytes follow the last frame")
    return header, jpegs


def decode_jpeg_rgb(data: bytes) -> np.ndarray:
    """Decode one JPEG to an RGB ``uint8`` array of shape (height, width, 3).

    The backend encodes RGB frames to JPEG through OpenCV after reversing the channel order
    (application/backend/src/utils/jpeg.py:9-21), so the decoded colours are true RGB.
    """
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def environment_draft(saved: Mapping[str, Any]) -> dict[str, Any]:
    """The ``environment`` a ``load_environment`` command takes, built from a saved environment.

    ``saved`` is an EnvironmentWithRelations JSON (GET /api/projects/{id}/environments/{env_id}).
    The mapping is the one the Studio UI uses (runtime-session-provider.tsx:64-73).
    """
    robots: list[dict[str, Any]] = []
    for entry in saved.get("robots") or []:
        tele = entry["tele_operator"]
        tele_operator = {"type": "robot", "robot_id": tele["robot_id"]} if tele["type"] == "robot" else {"type": "none"}
        robots.append({"robot_id": entry["robot"]["id"], "tele_operator": tele_operator})
    cameras = [{"camera_id": camera["id"]} for camera in saved.get("cameras") or [] if camera.get("id")]
    return {"id": saved["id"], "name": saved["name"], "robots": robots, "cameras": cameras}


def model_control(
    task: str,
    *,
    hz: float = 30.0,
    mode: str = "sync",
    request_threshold: float = 0.5,
    grace_s: float = 3.0,
) -> dict[str, Any]:
    """The ``control`` a ``set_control`` command takes to run a loaded model.

    The bounds are the ones ``ModelControlConfig`` enforces (schemas/runtime_session.py:54-67).

    Raises:
        ValueError: A value is outside its bound.
    """
    if not 0 < hz <= 500:
        raise ValueError(f"hz must be in (0, 500], got {hz}")
    if mode not in ("sync", "async"):
        raise ValueError(f"mode must be 'sync' or 'async', got {mode!r}")
    if not 0 <= request_threshold <= 1:
        raise ValueError(f"request_threshold must be in [0, 1], got {request_threshold}")
    if not 0 <= grace_s <= 30:
        raise ValueError(f"grace_s must be in [0, 30], got {grace_s}")
    return {
        "type": "model",
        "task": task,
        "hz": float(hz),
        "execution": {"mode": mode, "request_threshold": float(request_threshold)},
        "grace_s": float(grace_s),
    }


class StudioRuntimeClient:
    """One runtime session over one socket. Use it as ``async with`` or call ``connect`` and ``close``.

    A single reader task takes every message off the socket: acks go to the command that
    awaits them, state and status update the attributes below, frames update :meth:`latest_frame`,
    and observations queue up for :meth:`observations`. The queue keeps the newest
    ``observation_queue_size`` messages.
    """

    def __init__(
        self,
        project_id: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        hz: float = DEFAULT_HZ,
        command_timeout_s: float = COMMAND_TIMEOUT_S,
        observation_queue_size: int = OBSERVATION_QUEUE_SIZE,
        connect_timeout_s: float = CONNECT_TIMEOUT_S,
    ) -> None:
        self.project_id = str(project_id)
        self.url = ws_url(base_url, self.project_id, hz)
        self.state: dict[str, Any] | None = None
        """The session's last ``state`` message, ``data`` included (schemas/runtime_session.py:241-243)."""
        self.statuses: dict[str, dict[str, Any]] = {}
        """The last ``status`` message of each part and device, by component."""
        self.errors: deque[dict[str, Any]] = deque(maxlen=50)
        """Session ``error`` messages, oldest first."""
        self.unhandled: deque[dict[str, Any]] = deque(maxlen=20)
        """Text messages of a type this client does not use, kept for inspection."""
        self._command_timeout_s = command_timeout_s
        self._connect_timeout_s = connect_timeout_s
        self._ws: Any = None
        self._reader: asyncio.Task[None] | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._observations: asyncio.Queue[Any] = asyncio.Queue(maxsize=observation_queue_size)
        self._frames: dict[str, Frame] = {}
        self._status_count = 0
        self._status_at: dict[str, int] = {}
        self._changed = asyncio.Event()
        self._closed = asyncio.Event()
        self._closed_reason: str | None = None

    async def __aenter__(self) -> StudioRuntimeClient:
        return await self.connect()

    async def __aexit__(
        self, _exc_type: type[BaseException] | None, _exc: BaseException | None, _tb: TracebackType | None
    ) -> None:
        await self.close()

    async def connect(self) -> StudioRuntimeClient:
        """Open the socket and wait for the session's first state, which the backend sends on accept."""
        if self._ws is not None:
            raise StudioError("the client is already connected")
        self._ws = await connect(self.url, max_size=MAX_MESSAGE_BYTES, open_timeout=self._connect_timeout_s)
        self._reader = asyncio.create_task(self._read(self._ws))
        try:
            await self._wait_for(lambda: self.state is not None, self._connect_timeout_s, "the first session state")
        except BaseException:
            await self.close()
            raise
        return self

    async def close(self) -> None:
        """Close the socket. The backend then unloads everything the session loaded."""
        ws, self._ws = self._ws, None
        if ws is not None:
            with contextlib.suppress(ConnectionClosed, OSError):
                await ws.close()
        reader, self._reader = self._reader, None
        if reader is not None:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(reader, timeout=CLOSE_TIMEOUT_S)
        self._finish("the client closed the socket")

    async def load_environment(
        self, environment: Mapping[str, Any], *, unload_incompatible: bool = False, wait_status: bool = False
    ) -> None:
        """Load an environment draft (see :func:`environment_draft`): connect its robots and cameras.

        A device that fails to connect still leaves the environment loaded, failed; see :meth:`failures`.
        The ack and the environment's own status reach the client on separate messages
        (runtime_ws.py:116-122), so with ``wait_status`` the call also waits until the environment
        reports ``running`` or ``failed`` from a status that arrives after the call began.
        """
        mark = self._status_count
        await self._command(
            {
                "command": "load_environment",
                "environment": dict(environment),
                "unload_incompatible": unload_incompatible,
            }
        )
        await self._wait_for(lambda: self._environment() is not None, STATE_TIMEOUT_S, "the environment to load")
        if wait_status:
            await self._wait_for(
                lambda: self._environment_reported_since(mark), STATE_TIMEOUT_S, "the environment's status"
            )

    async def unload_environment(self) -> None:
        """Stop any control and disconnect the environment's devices."""
        await self._command({"command": "unload_environment"})
        await self._wait_for(lambda: self._environment() is None, STATE_TIMEOUT_S, "the environment to unload")

    async def load_model(self, model_id: str | uuid.UUID, backend: str, device: str = "auto") -> None:
        """Load one of the model's exports into its own process. An environment must be loaded first.

        ``backend`` must be one of the model's available backends (model.py:89-91); the export is read
        from ``exports/<backend>`` (model.py:92).
        """
        await self._command(
            {"command": "load_model", "model_id": _uuid_text(model_id), "backend": backend, "device": device}
        )
        await self._wait_for(
            lambda: self._model_matches(model_id, backend, device), STATE_TIMEOUT_S, "the model to load"
        )

    async def unload_model(self) -> None:
        """Stop a model control, if one runs, and unload the model."""
        await self._command({"command": "unload_model"})
        await self._wait_for(lambda: self._loaded_model() is None, STATE_TIMEOUT_S, "the model to unload")

    async def start_policy(
        self,
        model_id: str | uuid.UUID,
        task: str,
        *,
        backend: str,
        device: str = "auto",
        hz: float = 30.0,
        mode: str = "sync",
        request_threshold: float = 0.5,
        grace_s: float = 3.0,
    ) -> None:
        """Run the loaded policy on the follower, asked to do ``task``.

        Loads the model first unless that exact model, backend and device is already loaded. The
        environment must be loaded, and every camera the model reads must have sent a frame (see
        :meth:`wait_for_frames`). The policy keeps driving the follower until :meth:`stop`.
        """
        control = model_control(task, hz=hz, mode=mode, request_threshold=request_threshold, grace_s=grace_s)
        if not self._model_matches(model_id, backend, device):
            await self.load_model(model_id, backend, device)
        await self._command({"command": "set_control", "control": control})
        await self._wait_for(
            lambda: (self._control() or {}).get("type") == "model", STATE_TIMEOUT_S, "the model control to start"
        )

    async def stop(self, *, unload_model: bool = False) -> None:
        """Stop the policy. The follower holds where it is; the model stays loaded unless asked otherwise."""
        await self._command({"command": "set_control", "control": None})
        await self.wait_until_idle(STATE_TIMEOUT_S)
        if unload_model and self._loaded_model() is not None:
            await self.unload_model()

    async def wait_until_idle(self, timeout: float = 60.0) -> None:
        """Return once no control drives the follower.

        Raises:
            TimeoutError: A control still runs after ``timeout`` seconds.
            StudioConnectionError: The session ended first.
        """
        await self._wait_for(lambda: self._control() is None, timeout, "the control to stop")

    async def wait_for_frames(self, keys: Iterable[str], timeout_s: float = FRAME_TIMEOUT_S) -> None:
        """Wait until every camera in ``keys`` has sent at least one frame.

        A model raises ``Camera ... has no frame yet`` on inference for a camera with no frame
        (workers/model_worker.py:74-75), so call this before :meth:`start_policy`.

        Raises:
            TimeoutError: A camera is still missing a frame after ``timeout_s`` seconds.
        """
        wanted = list(keys)
        await self._wait_for(
            lambda: all(key in self._frames for key in wanted), timeout_s, f"a frame from each of {wanted}"
        )

    def latest_frame(self, key: str) -> Frame | None:
        """The newest frame of camera ``key``, or ``None`` before its first one arrives."""
        return self._frames.get(key)

    async def observations(self, *, decode: bool = True) -> AsyncIterator[Observation]:
        """Yield observations as they arrive, until the socket or the session closes.

        With ``decode`` the camera JPEGs are decoded to RGB arrays on the way out. Each message
        carries joints every tick they change, and the frames new since the last message.
        """
        while True:
            item = await self._observations.get()
            if item is _END:
                self._observations.put_nowait(_END)
                return
            header, jpegs, stamps = item
            frames = {key: decode_jpeg_rgb(jpeg) for key, jpeg in jpegs.items()} if decode else {}
            yield Observation(
                timestamp=float(header["timestamp"]),
                state=header.get("state"),
                action=header.get("action"),
                frames=frames,
                jpegs=jpegs,
                frame_timestamps=stamps,
            )

    def failures(self) -> dict[str, str]:
        """The parts and devices whose last status is ``failed``, with the reason each gave."""
        return {
            component: status.get("error") or "failed"
            for component, status in self.statuses.items()
            if status.get("state") == "failed"
        }

    def _environment(self) -> dict[str, Any] | None:
        return (self.state or {}).get("environment")

    def _environment_reported_since(self, mark: int) -> bool:
        status = self.statuses.get("environment") or {}
        return self._status_at.get("environment", 0) > mark and status.get("state") in ("running", "failed")

    def _loaded_model(self) -> dict[str, Any] | None:
        return (self.state or {}).get("model")

    def _control(self) -> dict[str, Any] | None:
        return (self.state or {}).get("control")

    def _model_matches(self, model_id: str | uuid.UUID, backend: str, device: str) -> bool:
        loaded = self._loaded_model()
        return (
            loaded is not None
            and loaded.get("model_id") == _uuid_text(model_id)
            and loaded.get("backend") == backend
            and loaded.get("device") == device
        )

    async def _command(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one command and return its ack. Raises when Studio refuses it."""
        if self._ws is None or self._closed.is_set():
            raise StudioConnectionError(f"{payload['command']}: {self._closed_reason or 'not connected'}")
        request_id = uuid.uuid4().hex
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._ws.send(json.dumps({**payload, "request_id": request_id}))
            ack = await asyncio.wait_for(future, timeout=self._command_timeout_s)
        except ConnectionClosed as error:
            raise StudioConnectionError(f"{payload['command']}: the socket closed ({error})") from error
        except TimeoutError as error:
            raise TimeoutError(f"{payload['command']} had no ack within {self._command_timeout_s:g} s") from error
        finally:
            self._pending.pop(request_id, None)
        if not ack.get("ok", False):
            refusal = ack.get("error") or {}
            raise StudioCommandError(
                str(refusal.get("message", "")), str(refusal.get("error_code", "unknown")), refusal.get("parts")
            )
        return ack

    async def _wait_for(self, predicate: Callable[[], bool], timeout: float, what: str) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not predicate():
            if self._closed.is_set():
                raise StudioConnectionError(f"waiting for {what}: {self._closed_reason}")
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(f"no {what} within {timeout:g} s")
            waiter = self._changed.wait()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(waiter, timeout=remaining)

    def _bump(self) -> None:
        self._changed.set()
        self._changed = asyncio.Event()

    async def _read(self, ws: Any) -> None:
        reason = "the socket closed"
        try:
            async for message in ws:
                self._dispatch(message)
        except ConnectionClosed as error:
            reason = f"the socket closed ({error})"
        except Exception as error:  # a message this client cannot read ends its session
            reason = f"the client could not read a message ({error!r})"
        finally:
            self._finish(reason)

    def _dispatch(self, message: str | bytes) -> None:
        if isinstance(message, bytes):
            header, jpegs = parse_observation(message)
            stamps = {str(frame["key"]): float(frame["timestamp"]) for frame in header.get("frames", [])}
            for key, jpeg in jpegs.items():
                self._frames[key] = Frame(key=key, timestamp=stamps[key], jpeg=jpeg)
            self._put_observation((header, jpegs, stamps))
            self._bump()
            return
        data = json.loads(message)
        kind = data.get("type")
        if kind == "ack":
            self._resolve_ack(data)
        elif kind == "state":
            self.state = data["data"]
            self._bump()
        elif kind == "status":
            component = str(data["component"])
            if data.get("state") in ("starting", "stopped"):
                # A part that starts or stops takes its devices with it (schemas/runtime_session.py:258-266).
                for stale in [name for name in self.statuses if name.startswith(component + "/")]:
                    del self.statuses[stale]
            self._status_count += 1
            self._status_at[component] = self._status_count
            self.statuses[component] = data
            self._bump()
        elif kind == "error":
            self.errors.append(data)
            if not data.get("session_alive", True):
                self._finish(f"the session ended ({data.get('error_code')})")
            self._bump()
        else:
            self.unhandled.append(data)

    def _resolve_ack(self, ack: dict[str, Any]) -> None:
        request_id = str(ack.get("request_id", ""))
        future = self._pending.pop(request_id, None)
        if future is None and request_id == "" and self._pending:
            # The backend could not read a request_id from a command it refused as malformed.
            future = self._pending.pop(next(iter(self._pending)))
        if future is not None and not future.done():
            future.set_result(ack)

    def _put_observation(self, item: Any) -> None:  # noqa: ANN401
        if self._observations.full():
            self._observations.get_nowait()
        self._observations.put_nowait(item)

    def _finish(self, reason: str) -> None:
        if self._closed.is_set():
            return
        self._closed_reason = reason
        self._closed.set()
        for future in self._pending.values():
            if not future.done():
                future.set_exception(StudioConnectionError(reason))
        self._pending.clear()
        self._put_observation(_END)
        self._bump()
