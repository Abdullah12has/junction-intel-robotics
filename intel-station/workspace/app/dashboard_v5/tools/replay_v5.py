"""Synthetic but schema-exact policy run for the v5 dashboard: no robot, no Studio, no camera.

Writes, under --out:
  logs/loop/<run_id>/events.jsonl   run_start, state, scan, triage, inspection (still_hires, per-camera
                                    cameras dict), decision, skill_start, policy_run (start, grasp, carry,
                                    release, return, done), skill_end, verify, stats, heartbeat, run_end
  logs/loop/<run_id>/frames/        scan frames, brick crops, still frames, verify frame (Pillow JPEGs)
  logs/loop/current                 symlink to the run dir
  logs/telemetry/<UTC date>.jsonl   one sample per second in the shape app/telemetry writes (pkg_w null)
  bench.json                        a bench file in the bench.json shape (synthetic rows)
Everything is labelled synthetic: run_start.config.synthetic, every telemetry src, bench status.

The event fields follow app/loop_v2/app/loop/policy_cycle.py (the emitter) and LOOP-CONTRACT section 8.
The brick renderer comes from fake_run.py in this folder.

Examples:
  python3 -I tools/replay_v5.py --out /tmp/v5demo --cycles 5
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image  # noqa: E402

import fake_run  # noqa: E402  (renderer only: draw_brick, render_crop, jpeg_bytes, atomic_write, MAT, W, H)

TOP_KEY = "realsense 243622061421"
CLOSE_KEY = "realsense 243722064316"
PLAN = [
    # kind (renderer), colour, scan verdict, route, label after inspection, plate, path, skill
    {"kind": "DEFECTIVE_NOT_2X2", "colour": "white", "verdict": "CONFIDENT_DEFECT", "route": "GREEN_DIRECT",
     "result": None, "plate": "GREEN", "via": "direct", "skill": "S3", "reasons": ["not_2x2"]},
    {"kind": "GOOD", "colour": "blue", "verdict": "GOOD", "route": "INSPECT",
     "result": "GOOD", "plate": "BLUE", "via": "inspected", "skill": "S2", "reasons": []},
    {"kind": "DEFECTIVE_TAPE", "colour": "yellow", "verdict": "UNCERTAIN", "route": "INSPECT",
     "result": "DEFECTIVE", "plate": "GREEN", "via": "inspected", "skill": "S3", "reasons": ["tape/appearance"]},
    {"kind": "UNCERTAIN", "colour": "pink", "verdict": "UNCERTAIN", "route": "INSPECT",
     "result": "UNCERTAIN", "plate": "GREEN", "via": "fail_safe", "skill": "S3", "reasons": ["no_brick"]},
    {"kind": "GOOD", "colour": "orange", "verdict": "GOOD", "route": "INSPECT",
     "result": "GOOD", "plate": "BLUE", "via": "inspected", "skill": "S2", "reasons": []},
]
MODELS = {"S2": "replay-act-blue-openvino", "S3": "replay-act-green-openvino"}
TASKS = {"S2": "move the red brick to the blue plate", "S3": "move the red brick to the green plate"}
PHASE_STEP_S = [("start", 0.0), ("grasp", 1.1), ("carry", 1.4), ("release", 1.2), ("return", 1.3), ("done", 0.9)]
SCAN_LAYOUT = [(190.0, 300.0, "b2", 0.92), (330.0, 210.0, "b1", 1.0), (460.0, 330.0, "b3", 0.95)]


def stamp(moment: dt.datetime) -> str:
    return moment.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


class Writer:
    """Collects events with increasing synthetic times; --pace > 0 sleeps between them (live look)."""

    def __init__(self, run_dir: Path, start: dt.datetime, pace: float) -> None:
        self.run_dir = run_dir
        self.now = start
        self.pace = pace
        self.mono = 0.0
        self.fd = os.open(run_dir / "events.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)

    def advance(self, seconds: float) -> None:
        self.now += dt.timedelta(seconds=seconds)
        self.mono += seconds
        if self.pace > 0:
            import time as _time
            _time.sleep(seconds / self.pace)

    def emit(self, kind: str, cycle: int, **fields: Any) -> None:
        record = {"t": stamp(self.now), "mono": round(self.mono, 3), "type": kind, "cycle": cycle}
        record.update(fields)
        data = (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        os.write(self.fd, data)

    def close(self) -> None:
        os.close(self.fd)


def write_image(run_dir: Path, rel: str, img: Image.Image, quality: int = 92) -> str:
    fake_run.atomic_write(run_dir / rel, fake_run.jpeg_bytes(img, quality))
    return rel


def scan_scene(bricks: list[dict[str, Any]]) -> Image.Image:
    base = Image.new("RGBA", (fake_run.W, fake_run.H), fake_run.MAT + (255,))
    for brick in bricks:
        base = fake_run.draw_brick(base, brick["x"], brick["y"], brick["scale"], brick["kind"], brick["colour"])
    return base.convert("RGB")


def telemetry_samples(start: dt.datetime, seconds: int, rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    for second in range(seconds):
        moment = start + dt.timedelta(seconds=second)
        cpu = round(rng.uniform(0.15, 0.55), 4)
        rows.append({
            "t": stamp(moment),
            "epoch": round(moment.timestamp(), 3),
            "pkg_w": None,
            "temp_c": {"pkg": round(rng.uniform(64.0, 84.0), 2), "coretemp_pkg": 0.0, "nvme": 36.8, "core_max": 0.0},
            "cpu_util": cpu,
            "gpu": {"freq_mhz": round(rng.uniform(300, 1900), 0), "max_mhz": 2500, "freq_ratio": round(rng.uniform(0.1, 0.75), 3), "media_mhz": 0},
            "npu": {"busy": round(rng.uniform(0.0, 0.35), 4), "freq_mhz": 0},
            "mem": {"used_pct": round(rng.uniform(24.0, 27.0), 2), "avail_mb": 47000.0},
            "bat": {"status": "Full", "power_w": 0.0, "capacity": 100},
            "ac_online": 1,
            "src": "replay synthetic (not measured); pkg_w:none",
        })
    return rows


def bench_file(now: dt.datetime, rng: random.Random) -> dict[str, Any]:
    rows = []
    for model in ("act_389da0fd_fp32", "act_389da0fd_fp16", "patchcore_detector_native_b2", "padim_b1_detector"):
        for device in ("CPU", "GPU", "NPU"):
            p50 = round(rng.uniform(3.0, 120.0), 4)
            rows.append({
                "model": model, "device": device, "status": "ok", "stage": "done",
                "latency_ms": {"p50": p50, "p90": round(p50 * 1.04, 4), "p95": round(p50 * 1.08, 4),
                               "mean": p50, "std": 1.0, "min": round(p50 * 0.95, 4), "max": round(p50 * 1.2, 4)},
                "throughput_ips": round(1000.0 / p50, 3), "images_per_s": round(1000.0 / p50, 3),
                "telemetry_during_timed": {"pkg_temp_c_mean": round(rng.uniform(70, 95), 2),
                                           "cpu_util_mean": round(rng.uniform(0.1, 0.5), 4),
                                           "npu_busy_mean": round(rng.uniform(0.0, 0.5), 4),
                                           "gpu_freq_mhz_mean": round(rng.uniform(300, 1800), 1)},
                "energy_per_inference_j": None,
            })
    return {
        "status": "synthetic-replay", "started_utc": stamp(now), "finished_utc": stamp(now),
        "openvino": "replay (synthetic)", "warmup": 20, "timed": 200, "batch": 1,
        "full_device_name": {"CPU": "replay CPU", "GPU": "replay GPU", "NPU": "replay NPU"},
        "power": {"pkg_w": "not measured (synthetic replay)", "energy_per_inference_j": None},
        "runs": rows,
    }


def build(out: Path, cycles: int, run_id: str, pace: float, seed: int, ended: bool = False) -> Path:
    rng = random.Random(seed)
    now = dt.datetime.now(dt.timezone.utc)
    seconds_total = max(60, cycles * 12 + 12)
    start = now - dt.timedelta(seconds=seconds_total)
    runs_root = out / "logs" / "loop"
    run_dir = runs_root / run_id
    (run_dir / "frames").mkdir(parents=True, exist_ok=False)
    link_tmp = runs_root / f"current.tmp-{os.getpid()}"
    writer = Writer(run_dir, start, pace)
    plates = {
        plate: {"skill": skill, "model_id": MODELS[skill], "task": TASKS[skill], "backend": "openvino",
                "device": "GPU", "hz": 10, "mode": "sync", "request_threshold": 0.5, "grace_s": 1.0}
        for plate, skill in (("BLUE", "S2"), ("GREEN", "S3"))
    }
    writer.emit("run_start", 0, run_id=run_id, mode="policy", config={
        "synthetic": True, "note": "replay_v5: synthetic policy run; no robot, Studio or camera",
        "mode": "policy", "top_key": TOP_KEY, "close_key": CLOSE_KEY,
        "home": {"pan": -6.6, "lift": -98.1, "elbow": 99.4, "wrist_flex": 54.3, "wrist_roll": -53.1, "gripper": 25.7},
        "home_tol": 3.0, "act_timeout_s": 60.0, "max_cycles": cycles, "plates": plates,
    })
    writer.advance(0.4)
    writer.emit("state", 0, state="IDLE")
    for number in range(1, cycles + 1):
        step = PLAN[(number - 1) % len(PLAN)]
        prefix = f"frames/c{number:03d}"
        writer.advance(0.8)
        writer.emit("state", number, state="HOME")
        writer.advance(0.9)
        writer.emit("state", number, state="SCAN")
        bricks: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        target_id = "b1"
        for index, (x, y, brick_id, scale) in enumerate(SCAN_LAYOUT):
            if brick_id == target_id:
                kind, colour, verdict = step["kind"], step["colour"], step["verdict"]
                reasons = list(step["reasons"])
                confident = verdict == "CONFIDENT_DEFECT"
            else:
                if index == 0:
                    kind, colour, verdict, reasons, confident = "GOOD", "pink", "GOOD", [], False
                else:  # a confident defect next to the target: routed straight to GREEN (I-52)
                    kind, colour, verdict, reasons, confident = ("DEFECTIVE_NOT_2X2", "white", "CONFIDENT_DEFECT",
                                                                 ["not_2x2"], True)
            bricks.append({"x": x, "y": y, "scale": scale, "kind": kind, "colour": colour})
            crop_top = write_image(run_dir, f"{prefix}_scan_{brick_id}_top.jpg",
                                   fake_run.render_crop(kind, colour, size=224))
            crop_close = write_image(run_dir, f"{prefix}_scan_{brick_id}_close.jpg",
                                     fake_run.render_crop(kind, colour, size=256))
            records.append({
                "id": brick_id, "colour": colour, "centre_px": [round(x, 1), round(y, 1)],
                "angle_deg": round(rng.uniform(-6, 6), 1), "shape": "4x2" if kind == "DEFECTIVE_NOT_2X2" else "2x2",
                "shape_conf": round(rng.uniform(0.82, 0.99), 3), "tape_score": round(
                    rng.uniform(0.6, 0.9) if kind == "DEFECTIVE_TAPE" else rng.uniform(0.02, 0.2), 3),
                "tape_conf": round(rng.uniform(0.7, 0.98), 3), "verdict": verdict, "confident": confident,
                "reasons": reasons, "crop_top": crop_top, "crop_close": crop_close,
            })
        frame_top = write_image(run_dir, f"{prefix}_scan_top.jpg", scan_scene(bricks))
        frame_close = write_image(run_dir, f"{prefix}_scan_close.jpg", scan_scene(bricks).rotate(4))
        writer.advance(1.1)
        writer.emit("scan", number, bricks=records, target_id=target_id, target_how="nearest",
                    hands={"top": False, "close": False}, frame_top=frame_top, frame_close=frame_close)
        writer.advance(0.4)
        writer.emit("state", number, state="TRIAGE")
        route_reasons = ["confident defect: straight to GREEN (I-52)"] if step["route"] == "GREEN_DIRECT" else [
            "not confident: inspect"]
        writer.advance(0.5)
        writer.emit("triage", number, target_id=target_id, route=step["route"], reasons=route_reasons)
        if step["route"] == "INSPECT":
            writer.advance(0.4)
            writer.emit("state", number, state="INSPECT")
            writer.advance(1.2)
            label = step["result"] or "UNCERTAIN"
            top_scene = scan_scene([{"x": 330.0, "y": 210.0, "scale": 1.0, "kind": step["kind"], "colour": step["colour"]}])
            cam_top = write_image(run_dir, f"{prefix}_still_top.jpg", top_scene)
            cam_close = write_image(run_dir, f"{prefix}_still_close.jpg", fake_run.render_crop(step["kind"], step["colour"], size=256))
            cameras = {
                TOP_KEY: {"label": label, "score": round(0.3 + rng.uniform(0.0, 0.5), 3), "threshold": 0.42,
                          "reasons": step["reasons"], "device": "GPU", "latency_ms": round(rng.uniform(8.0, 13.0), 1),
                          "frame": cam_top, "crop": cam_top},
                CLOSE_KEY: {"label": label, "score": round(0.25 + rng.uniform(0.0, 0.4), 3), "threshold": 0.42,
                            "reasons": step["reasons"], "device": "GPU", "latency_ms": round(rng.uniform(8.0, 13.0), 1),
                            "frame": cam_close, "crop": cam_close},
            }
            writer.emit("inspection", number, target_id=target_id, mode="still_hires", result=label, label=label,
                        reasons=step["reasons"], score=0.61, threshold=0.42, cameras=cameras, geometry=None,
                        device="GPU", latency_ms=round(rng.uniform(8.0, 13.0), 1), frame=cam_top, crop=cam_close,
                        route=("S2" if label == "GOOD" else "S3"))
        writer.advance(0.6)
        writer.emit("decision", number, target_id=target_id, plate=step["plate"], rule="OD-10", via=step["via"])
        skill = plates[step["plate"]]
        writer.advance(0.3)
        writer.emit("state", number, state="ACT")
        writer.emit("skill_start", number, skill=skill["skill"], model=skill["model_id"], plate=step["plate"],
                    task_text=skill["task"])
        for phase, seconds in PHASE_STEP_S:
            writer.advance(seconds)
            writer.emit("policy_run", number, phase=phase, model_id=skill["model_id"], task_text=skill["task"],
                        plate=step["plate"], target_id=target_id)
        writer.emit("skill_end", number, skill=skill["skill"], ok=True, duration_s=5.2, reason="done")
        writer.advance(0.6)
        writer.emit("state", number, state="VERIFY")
        writer.advance(0.8)
        plate_x, plate_y = fake_run.BLUE_PLATE if step["plate"] == "BLUE" else fake_run.GREEN_PLATE
        placed = fake_run.render_scene({"brick": {"x": plate_x, "y": plate_y, "scale": 0.9, "kind": step["kind"],
                                                  "colour": step["colour"]}, "gap": 250, "jaws": False},
                                       caption="replay verify, top view")
        frame_verify = write_image(run_dir, f"{prefix}_verify_top.jpg", placed)
        writer.emit("verify", number, target_id=target_id, plate_intended=step["plate"],
                    release_side=step["plate"], ok=True, frame_top=frame_verify)
        writer.advance(0.4)
        writer.emit("stats", number, cycles=number, good=sum(1 for s in PLAN[:number] if s["plate"] == "BLUE"),
                    defective=sum(1 for s in PLAN[:number] if s["plate"] == "GREEN"), uncertain=0,
                    mean_cycle_s=round(seconds_total / max(1, cycles), 2))
    writer.advance(0.6)
    writer.emit("state", cycles, state="IDLE")
    writer.emit("heartbeat", cycles)
    if ended:
        writer.emit("run_end", cycles, reason="max_cycles")
    writer.close()
    write_image(run_dir, "live.jpg", scan_scene([]), quality=85)
    os.symlink(run_id, link_tmp)
    os.replace(link_tmp, runs_root / "current")
    (out / "logs" / "telemetry").mkdir(parents=True, exist_ok=True)
    samples = telemetry_samples(start, seconds_total + 1, rng)
    with open(out / "logs" / "telemetry" / f"{now:%Y%m%d}.jsonl", "w", encoding="utf-8") as handle:
        for row in samples:
            if dt.datetime.fromisoformat(row["t"].replace("Z", "+00:00")) <= now:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    (out / "bench.json").write_text(json.dumps(bench_file(now, rng), indent=1), encoding="utf-8")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic policy run for the v5 dashboard (schema-exact).")
    parser.add_argument("--out", required=True, type=Path, help="folder that gets logs/loop, logs/telemetry, bench.json")
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--run-id", default="20261114-120000")
    parser.add_argument("--pace", type=float, default=0.0, help="0 = write at once; 1 = real time")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ended", action="store_true", help="write run_end (default: the run stays open)")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    (out / "logs" / "loop").mkdir(parents=True, exist_ok=True)
    run_dir = build(out, args.cycles, args.run_id, args.pace, args.seed, args.ended)
    print(f"replay run {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
