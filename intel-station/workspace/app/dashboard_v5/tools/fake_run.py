"""Synthetic run writer for the dashboard: no robot, no Studio, no camera (OD-9 cycle).

One cycle, as LOOP-CONTRACT section 7 (OD-9) describes it:
  WAIT_BRICK -> S1 (policy moves the brick to the inspection spot)
  -> INSPECT (both RealSense views, one verdict per camera, then the fused verdict)
  -> HANDOVER_OUT -> SORT (scripted sorter: align, grasp, carry, drop on BLUE or GREEN)
  -> HANDOVER_IN -> CYCLE_DONE.
--sort-mode policy replaces SORT by the team S2/S3 skills (spot -> plate).

Writes a run dir exactly as LOOP-CONTRACT section 5 describes:
  <runs>/<run_id>/events.jsonl                 run_start, state, skill_start/end, inspect_cameras,
                                               inspection, handover, sort_start, align, sort_end,
                                               stats, error, heartbeat (1 s), control, run_end
  <runs>/<run_id>/live.jpg                     wrist preview, rewritten atomically about 5 times a second
  <runs>/<run_id>/frames/cNNN_{close,top}_pre.jpg    frame per RealSense view at the spot
  <runs>/<run_id>/frames/cNNN_{close,top}_crop.jpg   brick crop per RealSense view
  <runs>/current                               symlink to the run dir (replaced atomically)
It obeys control.json like the orchestrator: start, next, stop (hold, then STOPPED), quit.
Images are generated with Pillow. Everything it writes is synthetic and says so in run_start.

Examples:
  python3 -I tools/fake_run.py --runs logs/loop --auto-next
  python3 -I tools/fake_run.py --runs /tmp/x/loop --speed 20 --cycles 4 --auto-next --run-id demo
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw

W, H = 640, 480
THRESHOLD = 0.42
UNCERTAIN_MARGIN = 0.05
JAW_OPEN, JAW_CLOSED = 250.0, 130.0
CENTER_X, GRASP_Y = 320.0, 300.0
SPOT_X, SPOT_Y = 320.0, 200.0  # inspection spot, synthetic coordinates in the wrist view
BLUE_PLATE, GREEN_PLATE = (110.0, 400.0), (530.0, 400.0)
MAT = (34, 36, 40)
RED_JAW = (196, 36, 40)
COLOURS = {
    "pink": (232, 76, 150),
    "blue": (40, 96, 220),
    "yellow": (246, 200, 40),
    "orange": (240, 120, 30),
}
WHITE_2X4 = (241, 245, 249)
GREEN_PLATE_RGB = (34, 160, 80)
BLUE_PLATE_RGB = (40, 96, 220)
# RealSense serials from LOOP-CONTRACT section 5 (OD-9): close low view and top view.
CAMERAS = {"close": "realsense 243722064316", "top": "realsense 243622061421"}
SCRIPT = [
    "GOOD", "DEFECTIVE_TAPE", "GOOD", "UNCERTAIN", "GOOD",
    "DEFECTIVE_NOT_2X2", "GOOD", "DEFECTIVE_TAPE", "GOOD", "UNCERTAIN",
]
MODELS = {"S1": "synthetic-smolvla-pick-to-spot", "S2": "synthetic-act-blue", "S3": "synthetic-act-green"}
# Per brick kind and camera view: (label, score factor on THRESHOLD, reasons).
# Reasons use the LOOP-CONTRACT keys; the page shows tape/appearance as "tape" and not_2x2 as "not 2x2" (OD-10).
CAMERA_VERDICTS = {
    "GOOD": {"close": ("GOOD", 0.45, []), "top": ("GOOD", 0.40, [])},
    "DEFECTIVE_TAPE": {"close": ("DEFECTIVE", 1.9, ["tape/appearance"]),
                       "top": ("DEFECTIVE", 1.6, ["tape/appearance"])},
    "DEFECTIVE_NOT_2X2": {"close": ("GOOD", 0.30, []), "top": ("DEFECTIVE", 1.8, ["not_2x2"])},
    "UNCERTAIN": {"close": ("GOOD", 0.35, []), "top": ("UNCERTAIN", 1.0, ["tape/appearance"])},
}


class Stop(Exception):
    """Operator STOP while a cycle is running."""


class Quit(Exception):
    """Operator quit."""


class Fault(Exception):
    """Injected loop error: the loop goes HOLD, then STOPPED."""

    def __init__(self, where: str, message: str) -> None:
        super().__init__(message)
        self.where = where
        self.message = message


class MaxCycles(Exception):
    """Configured number of cycles reached."""


def utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def shade(rgb: tuple[int, int, int], k: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(c * k))) for c in rgb)  # type: ignore[return-value]


def fuse(labels: list[str]) -> str:
    """Fused verdict: DEFECTIVE if any view says DEFECTIVE, UNCERTAIN if any says UNCERTAIN, else GOOD."""
    if "DEFECTIVE" in labels:
        return "DEFECTIVE"
    if "UNCERTAIN" in labels:
        return "UNCERTAIN"
    return "GOOD"


def draw_brick(img: Image.Image, cx: float, cy: float, scale: float, kind: str, colour: str) -> Image.Image:
    """Top view of a Duplo-size brick on an RGBA image; returns the composited image."""
    if kind == "DEFECTIVE_NOT_2X2":
        cols, rows, body = 4, 2, WHITE_2X4
    else:
        cols, rows, body = 2, 2, COLOURS[colour]
    pitch = 60.0 * scale
    w, h = cols * pitch, rows * pitch
    x0, y0 = cx - w / 2, cy - h / 2
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle(
        [x0, y0, x0 + w, y0 + h], radius=max(2, int(4 * scale)),
        fill=body + (255,), outline=shade(body, 0.6) + (255,), width=max(1, int(2 * scale)),
    )
    stud = 0.22 * pitch
    for r in range(rows):
        for c in range(cols):
            sx, sy = x0 + (c + 0.5) * pitch, y0 + (r + 0.5) * pitch
            d.ellipse([sx - stud, sy - stud, sx + stud, sy + stud],
                      fill=shade(body, 1.12) + (255,), outline=shade(body, 0.7) + (255,))
    if kind == "DEFECTIVE_NOT_2X2":
        d.rectangle([x0 + 0.08 * w, y0 + h - 9 * scale, x0 + 0.08 * w + 28 * scale, y0 + h - 3 * scale],
                    fill=(20, 20, 20, 255))
    if kind == "DEFECTIVE_TAPE":
        band = [(x0 - 6, y0 + 0.30 * h), (x0 + w + 6, y0 + 0.08 * h),
                (x0 + w + 6, y0 + 0.08 * h + 0.34 * h), (x0 - 6, y0 + 0.30 * h + 0.34 * h)]
        d.polygon(band, fill=(206, 212, 220, 225))
    if kind == "UNCERTAIN":
        band = [(x0 - 6, y0 + 0.55 * h), (x0 + w + 6, y0 + 0.40 * h),
                (x0 + w + 6, y0 + 0.40 * h + 0.14 * h), (x0 - 6, y0 + 0.55 * h + 0.14 * h)]
        d.polygon(band, fill=(206, 212, 220, 70))
    return Image.alpha_composite(img, layer)


def draw_jaws(img: Image.Image, gap: float) -> Image.Image:
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    half = gap / 2.0
    d.rectangle([CENTER_X - half - 60, 0, CENTER_X - half, GRASP_Y + 8], fill=RED_JAW + (255,))
    d.rectangle([CENTER_X + half, 0, CENTER_X + half + 60, GRASP_Y + 8], fill=RED_JAW + (255,))
    return Image.alpha_composite(img, layer)


def render_scene(scene: dict[str, Any], caption: str = "synthetic wrist view") -> Image.Image:
    img = Image.new("RGBA", (W, H), MAT + (255,))
    d = ImageDraw.Draw(img)
    for x in range(0, W, 48):
        d.line([(x, 0), (x, H)], fill=(44, 47, 52, 255), width=1)
    for y in range(0, H, 48):
        d.line([(0, y), (W, y)], fill=(44, 47, 52, 255), width=1)
    for (px, py), name, rgb in ((BLUE_PLATE, "BLUE", BLUE_PLATE_RGB), (GREEN_PLATE, "GREEN", GREEN_PLATE_RGB)):
        d.ellipse([px - 60, py - 60, px + 60, py + 60], fill=rgb + (255,), outline=(230, 235, 240, 255), width=3)
        d.text((px - 14, py - 5), name, fill=(255, 255, 255, 255))
    brick = scene.get("brick")
    if brick:
        img = draw_brick(img, brick["x"], brick["y"], brick["scale"], brick["kind"], brick["colour"])
    if scene.get("jaws", True):
        img = draw_jaws(img, float(scene.get("gap", JAW_OPEN)))
    ImageDraw.Draw(img).text((10, H - 22), caption, fill=(120, 126, 136, 255))
    return img.convert("RGB")


def render_crop(kind: str, colour: str, size: int = 224) -> Image.Image:
    scale = 0.85 if kind == "DEFECTIVE_NOT_2X2" else 1.25
    img = Image.new("RGBA", (size, size), MAT + (255,))
    img = draw_brick(img, size / 2.0, size / 2.0, scale, kind, colour)
    return img.convert("RGB")


def jpeg_bytes(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class LiveWriter(threading.Thread):
    """Rewrites live.jpg (the wrist preview) about five times a second from the shared scene."""

    def __init__(self, run_dir: Path, scene: dict[str, Any], lock: threading.Lock, period_s: float = 0.2) -> None:
        super().__init__(daemon=True)
        self.run_dir = run_dir
        self.scene = scene
        self.lock = lock
        self.period_s = period_s
        self.done = threading.Event()

    def run(self) -> None:
        while not self.done.is_set():
            with self.lock:
                snapshot = dict(self.scene)
                if snapshot.get("brick"):
                    snapshot["brick"] = dict(snapshot["brick"])
            atomic_write(self.run_dir / "live.jpg", jpeg_bytes(render_scene(snapshot), 85))
            self.done.wait(self.period_s)


class FakeRun:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.speed = max(float(args.speed), 0.01)
        self.rng = random.Random(args.seed)
        self.runs_root = Path(args.runs)
        self.run_id = args.run_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.run_dir = self.runs_root / self.run_id
        self.control_path = self.run_dir / "control.json"
        self.events_fd: int | None = None
        self.last_control_id = 0
        self.completed = 0
        self.counts = {"good": 0, "defective": 0, "uncertain": 0}
        self.cycle_seconds: list[float] = []
        self.running = False
        self.start_requested = False
        self.next_requested = False
        self.skill: str | None = None
        self.skill_t0 = 0.0
        self.next_brick: dict[str, Any] | None = None
        self.scene: dict[str, Any] = {"brick": None, "gap": JAW_OPEN}
        self.scene_lock = threading.Lock()
        self.last_heartbeat = 0.0
        self.live: LiveWriter | None = None

    # ----- setup and output -------------------------------------------------

    def setup(self) -> None:
        self.runs_root.mkdir(parents=True, exist_ok=True)
        if self.run_dir.exists():
            raise SystemExit(f"run dir already exists: {self.run_dir}")
        (self.run_dir / "frames").mkdir(parents=True)
        self.events_fd = os.open(self.run_dir / "events.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        self.set_current()
        self.live = LiveWriter(self.run_dir, self.scene, self.scene_lock)
        self.live.start()

    def set_current(self) -> None:
        link = self.runs_root / "current"
        tmp = self.runs_root / f"current.tmp-{os.getpid()}"
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        os.symlink(self.run_id, tmp)
        os.replace(tmp, link)

    def _write(self, data: bytes) -> None:
        assert self.events_fd is not None
        view = memoryview(data)
        while view:
            written = os.write(self.events_fd, view)
            view = view[written:]

    def emit(self, kind: str, **fields: Any) -> None:
        record = {"t": utc_iso(), "mono": round(time.monotonic(), 3), "type": kind}
        record.update(fields)
        data = (json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8")
        if self.args.partial_lines and len(data) > 8 and self.rng.random() < 0.35:
            cut = len(data) // 2
            self._write(data[:cut])
            time.sleep(0.15)  # real time, so the server really sees a partial last line
            self._write(data[cut:])
        else:
            self._write(data)
        if kind != "heartbeat":
            detail = " ".join(f"{k}={v}" for k, v in fields.items() if k not in ("config", "cameras"))
            print(f"{record['t']} {kind} {detail}"[:160], flush=True)

    # ----- control and timing ----------------------------------------------

    def pump(self) -> None:
        now = time.monotonic()
        if now - self.last_heartbeat >= 1.0:
            self.last_heartbeat = now
            self.emit("heartbeat")
        self._poll_control()

    def _poll_control(self) -> None:
        try:
            data = json.loads(self.control_path.read_bytes())
        except (FileNotFoundError, ValueError):
            return
        cid = data.get("id")
        if not isinstance(cid, int) or cid <= self.last_control_id:
            return
        self.last_control_id = cid
        cmd = data.get("cmd")
        self.emit("control", cmd=cmd, id=cid)
        if cmd == "start":
            self.start_requested = True
        elif cmd == "next":
            self.next_requested = True
        elif cmd == "stop" and self.running:
            raise Stop()
        elif cmd == "quit":
            raise Quit()

    def wait(self, seconds: float, tick: Callable[[float], None] | None = None) -> float:
        """Sleep the scaled duration, pumping heartbeats and control; tick(fraction) drives the scene.
        Returns the real elapsed seconds."""
        start = time.monotonic()
        end = start + seconds / self.speed
        while True:
            now = time.monotonic()
            fraction = 1.0 if now >= end else (now - start) / (end - start)
            if tick is not None:
                tick(fraction)
            self.pump()
            if now >= end:
                break
            time.sleep(min(0.05, end - now))
        return time.monotonic() - start

    def set_scene(self, **fields: Any) -> None:
        with self.scene_lock:
            self.scene.update(fields)

    # ----- loop states ------------------------------------------------------

    def start_skill(self, name: str, cycle: int) -> None:
        self.skill = name
        self.skill_t0 = time.monotonic()
        self.emit("state", state=name, cycle=cycle)
        self.emit("skill_start", skill=name, model=MODELS[name], cycle=cycle)

    def end_skill(self, ok: bool, reason: str) -> None:
        if self.skill is None:
            return
        self.emit("skill_end", skill=self.skill, ok=ok,
                  duration_s=round(time.monotonic() - self.skill_t0, 2), reason=reason)
        self.skill = None

    def plan_next_brick(self) -> None:
        cycle = self.completed + 1
        if self.args.random:
            kind = self.rng.choices(
                ["GOOD", "DEFECTIVE_TAPE", "DEFECTIVE_NOT_2X2", "UNCERTAIN"], [55, 20, 10, 15])[0]
        else:
            kind = SCRIPT[(cycle - 1) % len(SCRIPT)]
        colour = "white" if kind == "DEFECTIVE_NOT_2X2" else self.rng.choice(sorted(COLOURS))
        start_x = self.rng.uniform(170, 470)
        start_y = self.rng.uniform(110, 190)
        scale = 1.0
        self.next_brick = {"kind": kind, "colour": colour, "start": (start_x, start_y), "scale": scale}
        self.set_scene(brick={"x": start_x, "y": start_y, "scale": scale, "kind": kind, "colour": colour},
                       gap=JAW_OPEN, jaws=True)

    def await_start(self) -> None:
        self.start_requested = False
        while not self.start_requested:
            self.pump()
            time.sleep(0.05)
        self.start_requested = False

    def await_brick(self) -> None:
        self.next_requested = False
        began = time.monotonic()
        while not self.next_requested:
            if self.args.auto_next and time.monotonic() - began >= self.args.auto_gap / self.speed:
                break
            self.pump()
            time.sleep(0.05)
        self.next_requested = False

    def camera_verdict(self, kind: str, view: str) -> tuple[str, float, list[str]]:
        label, factor, reasons = CAMERA_VERDICTS[kind][view]
        if label == "UNCERTAIN":
            score = THRESHOLD + self.rng.uniform(-0.6, 0.6) * UNCERTAIN_MARGIN
        else:
            score = THRESHOLD * factor * self.rng.uniform(0.92, 1.08)
        return label, round(score, 4), list(reasons)

    def write_views(self, prefix: str, kind: str, colour: str) -> list[dict[str, Any]]:
        """Frame and crop per RealSense view; the views share the same spot scene."""
        views: list[dict[str, Any]] = []
        for view, key in CAMERAS.items():
            scene = {"brick": {"x": SPOT_X, "y": SPOT_Y, "scale": 1.0, "kind": kind, "colour": colour},
                     "gap": JAW_OPEN, "jaws": False}
            frame = render_scene(scene, caption=f"synthetic {view} view")
            if view == "close":
                frame = frame.crop((160, 120, 480, 360)).resize((W, H))
            crop = render_crop(kind, colour, size=256 if view == "close" else 224)
            frame_rel = f"frames/{prefix}_{view}_pre.jpg"
            crop_rel = f"frames/{prefix}_{view}_crop.jpg"
            atomic_write(self.run_dir / frame_rel, jpeg_bytes(frame, 92))
            atomic_write(self.run_dir / crop_rel, jpeg_bytes(crop, 92))
            views.append({"view": view, "key": key, "frame": frame_rel, "crop": crop_rel})
        return views

    def run_cycle(self) -> None:
        assert self.next_brick is not None
        cycle = self.completed + 1
        brick = self.next_brick
        kind, colour = brick["kind"], brick["colour"]
        start_x, start_y = brick["start"]
        cycle_t0 = time.monotonic()
        prefix = f"c{cycle:03d}"

        # S1: the policy moves the brick to the inspection spot.
        self.start_skill("S1", cycle)
        self.wait(1.6, tick=lambda f: self.set_scene(brick={
            "x": start_x + (SPOT_X - start_x) * f, "y": start_y + (SPOT_Y - start_y) * f,
            "scale": 1.0, "kind": kind, "colour": colour}, gap=JAW_OPEN, jaws=True))
        self.end_skill(True, "brick at the inspection spot, arm stationary")

        # INSPECT: one verdict per RealSense view, then the fused verdict.
        self.emit("state", state="INSPECT", cycle=cycle)
        views = self.write_views(prefix, kind, colour)
        cameras: list[dict[str, Any]] = []
        for view in views:
            label, score, reasons = self.camera_verdict(kind, view["view"])
            cameras.append({"camera": view["key"], "label": label, "score": score, "threshold": THRESHOLD,
                            "reasons": reasons, "frame": view["frame"], "crop": view["crop"]})
        self.emit("inspect_cameras", cycle=cycle, cameras=cameras)
        fused = fuse([c["label"] for c in cameras])
        route = "S2" if fused == "GOOD" else "S3"
        top = cameras[-1]
        self.emit("inspection", cycle=cycle, label=fused, score=max(c["score"] for c in cameras),
                  threshold=THRESHOLD, reasons=sorted({r for c in cameras for r in c["reasons"]}),
                  geometry={"studs": 8 if kind == "DEFECTIVE_NOT_2X2" else 4,
                            "aspect": 2.0 if kind == "DEFECTIVE_NOT_2X2" else round(1.0 + self.rng.uniform(-0.03, 0.03), 2)},
                  device="GPU", latency_ms=round(self.rng.uniform(9.0, 14.0), 1),
                  frame=top["frame"], crop=top["crop"], route=route)
        self.wait(1.2)

        # HANDOVER_OUT: the arm is released, the follower is unloaded, the loaded model stays.
        self.emit("state", state="HANDOVER_OUT", cycle=cycle)
        took = self.wait(0.8)
        self.emit("handover", phase="out", duration_s=round(took, 2), model_kept=True)

        if self.args.error_after and self.completed == self.args.error_after:
            raise Fault("SORT" if self.args.sort_mode == "scripted" else route, "skill timeout after 60 s")

        if self.args.sort_mode == "policy":
            self.start_skill(route, cycle)
            self.wait(2.6, tick=lambda f: self.set_scene(brick={
                "x": SPOT_X + (self.plate_x(route) - SPOT_X) * f, "y": SPOT_Y + (GREEN_PLATE[1] - SPOT_Y) * f,
                "scale": 0.9, "kind": kind, "colour": colour}, gap=JAW_CLOSED, jaws=False))
            self.end_skill(True, "placed, arm stationary")
        else:
            self.emit("state", state="SORT", cycle=cycle)
            self.emit("sort_start", route=route, cycle=cycle)
            sort_t0 = time.monotonic()
            self.wait(0.6)  # home -> spot_hover
            self.emit("align", iteration=1, dx=6, dy=-3, correction=0.8)
            self.wait(0.5)
            self.emit("align", iteration=2, dx=1, dy=0, correction=0.1)
            self.wait(0.4, tick=lambda f: self.set_scene(gap=JAW_OPEN + (JAW_CLOSED - JAW_OPEN) * f, jaws=True))
            self.wait(1.4, tick=lambda f: self.set_scene(brick={
                "x": SPOT_X + (self.plate_x(route) - SPOT_X) * f, "y": SPOT_Y + (GREEN_PLATE[1] - SPOT_Y) * f,
                "scale": 1.0 - 0.1 * f, "kind": kind, "colour": colour}, gap=JAW_CLOSED, jaws=True))
            self.wait(0.6, tick=lambda f: self.set_scene(gap=JAW_CLOSED + (JAW_OPEN - JAW_CLOSED) * f))
            self.emit("sort_end", ok=True, duration_s=round(time.monotonic() - sort_t0, 2))

        # HANDOVER_IN: the same spec reuses the store; the model is still loaded.
        self.emit("state", state="HANDOVER_IN", cycle=cycle)
        took = self.wait(0.8)
        self.emit("handover", phase="in", duration_s=round(took, 2), model_kept=True)

        self.emit("state", state="CYCLE_DONE", cycle=cycle)
        self.wait(1.5)  # CYCLE_DONE stays on screen before the next brick is asked for
        self.completed += 1
        key = {"GOOD": "good", "DEFECTIVE": "defective", "UNCERTAIN": "uncertain"}[fused]
        self.counts[key] += 1
        self.cycle_seconds.append(time.monotonic() - cycle_t0)
        self.set_scene(brick=None, gap=JAW_OPEN, jaws=True)
        self.emit("stats", cycles=self.completed, good=self.counts["good"],
                  defective=self.counts["defective"], uncertain=self.counts["uncertain"],
                  mean_cycle_s=round(sum(self.cycle_seconds) / len(self.cycle_seconds), 2))

    @staticmethod
    def plate_x(route: str) -> float:
        return BLUE_PLATE[0] if route == "S2" else GREEN_PLATE[0]

    def abort(self, reason: str) -> None:
        self.end_skill(False, reason)
        self.emit("state", state="HOLD", cycle=self.completed + 1)
        self.set_scene(brick=None, gap=JAW_OPEN, jaws=True)
        self.emit("state", state="STOPPED", cycle=self.completed + 1)

    def main(self) -> str:
        reason = "quit"
        self.setup()
        self.emit("run_start", run_id=self.run_id, mode="dry-run", config={
            "synthetic": True,
            "note": "fake_run: no robot, no Studio, no camera; images are generated",
            "live_source": "wrist camera USB2.0_CAM1 (live.jpg)",
            "cameras": CAMERAS,
            "detector": {"device": "GPU", "backend": "openvino", "threshold": THRESHOLD,
                         "uncertain_margin": UNCERTAIN_MARGIN},
            "skills": {name: {"model_id": MODELS[name], "backend": "openvino", "device": "GPU"}
                       for name in MODELS},
            "routing": {"GOOD": "S2", "DEFECTIVE": "S3", "UNCERTAIN": "S3"},
            "sort_mode": self.args.sort_mode,
        })
        self.emit("state", state="IDLE", cycle=0)
        try:
            while True:
                self.await_start()
                self.running = True
                try:
                    while True:
                        self.plan_next_brick()
                        self.emit("state", state="WAIT_BRICK", cycle=self.completed + 1)
                        self.await_brick()
                        self.run_cycle()
                        if self.args.cycles and self.completed >= self.args.cycles:
                            raise MaxCycles()
                except Stop:
                    self.abort("operator stop")
                except Fault as fault:
                    self.emit("error", where=fault.where, message=fault.message)
                    self.abort(fault.message)
                finally:
                    self.running = False
        except Quit:
            reason = "quit"
        except MaxCycles:
            reason = "max_cycles"
        self.emit("run_end", reason=reason)
        if self.live is not None:
            self.live.done.set()
        if self.events_fd is not None:
            os.close(self.events_fd)
        return reason


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a synthetic loop run for the dashboard (OD-9 cycle).")
    parser.add_argument("--runs", default=str(Path(__file__).resolve().parents[1] / "logs" / "loop"),
                        help="runs root (gets the current symlink); default: <code folder>/logs/loop")
    parser.add_argument("--run-id", default=None, help="run dir name (default: UTC yyyymmdd-HHMMSS)")
    parser.add_argument("--speed", type=float, default=1.0, help="time scale; 1 = real time")
    parser.add_argument("--auto-next", action="store_true", help="place the next brick automatically")
    parser.add_argument("--auto-gap", type=float, default=2.0, help="seconds before auto-next (scaled by speed)")
    parser.add_argument("--cycles", type=int, default=0, help="stop after N cycles (0 = unlimited)")
    parser.add_argument("--error-after", type=int, default=0, help="inject a fault after N completed cycles (0 = off)")
    parser.add_argument("--sort-mode", choices=["scripted", "policy"], default="scripted",
                        help="scripted: SORT with align and carry; policy: team S2/S3 skills")
    parser.add_argument("--partial-lines", action="store_true", help="split some event lines across writes")
    parser.add_argument("--random", action="store_true", help="random verdict mix instead of the demo script")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run = FakeRun(args)
    print(f"fake run {run.run_id} in {run.runs_root}", flush=True)
    reason = run.main()
    print(f"run_end {reason}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
