"""Motion for the spot cell (B8): one subprocess per segment under envs/motion-env, reading R/app/motion.

Parent side: SubprocessArm.run(sequence) and SubprocessArm.teardown() start this file with the motion interpreter and
read the last JSON line of stdout. Child side plays one named sequence with so101_motion (torque held on at exit), or,
with --teardown, opens the arm passive (torque off) and disconnects. Importing this file touches no hardware.

Why this wraps so101_motion instead of calling run_sequence.py: run_sequence.py prints no gripper reading, and the
grasp check needs gripper.pos after the pick. The calls below are the ones run_sequence.py makes (see runbook B8).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

EXIT_STALLED, EXIT_BUSY, EXIT_CONFIG, EXIT_HARDWARE = 2, 3, 4, 5


class SubprocessArm:
    """Arm for the spot cell: every segment is a fresh process, so the torque state is carried by so101_motion."""

    def __init__(self, python: str, script: Path, poses: str, studio_export: str, motion_dir: str,
                 robot: str = "SO101 Follower", timeout_s: float = 240.0) -> None:
        self.python = python
        self.script = Path(script)
        self.base = ["--motion-dir", motion_dir, "--poses", poses, "--studio-export", studio_export, "--robot", robot]
        self.timeout_s = timeout_s

    def run(self, sequence: str) -> Any:
        return self._call(["--sequence", sequence])

    def teardown(self) -> Any:
        return self._call(["--teardown"])

    def _call(self, extra: list[str]) -> Any:
        from loop.spot_cell import SegmentResult  # imported here so the child process never needs the loop package

        command = [self.python, "-B", str(self.script), *self.base, *extra]
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=self.timeout_s, check=False)
        except subprocess.TimeoutExpired:
            return SegmentResult(False, EXIT_HARDWARE, self.timeout_s * 1000.0, None, f"timeout after {self.timeout_s:.0f} s")
        lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
        try:
            payload = json.loads(lines[-1]) if lines else {}
        except json.JSONDecodeError:
            payload = {}
        error = (done.stderr.strip().splitlines() or ["no stderr"])[-1]
        return SegmentResult(ok=done.returncode == 0 and payload.get("ok") is True, exit_code=done.returncode,
                             latency_ms=float(payload.get("latency_ms", 0.0)), gripper=payload.get("gripper"),
                             detail="" if done.returncode == 0 else error)


def child(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One spot-cell motion segment on the SO-101 follower (motion-env).")
    parser.add_argument("--motion-dir", required=True)
    parser.add_argument("--poses", required=True)
    parser.add_argument("--studio-export", required=True)
    parser.add_argument("--robot", default="SO101 Follower")
    parser.add_argument("--sequence")
    parser.add_argument("--teardown", action="store_true")
    args = parser.parse_args(argv)
    sys.path.insert(0, args.motion_dir)
    try:
        import so101_motion as motion
    except ImportError as error:
        print(json.dumps({"ok": False, "latency_ms": 0.0, "gripper": None}))
        print(f"missing dependency: {error}", file=sys.stderr)
        return EXIT_CONFIG
    began = time.perf_counter()
    payload: dict[str, Any] = {"ok": False, "latency_ms": 0.0, "gripper": None}
    failure: Exception | None = None
    try:
        port, calibration = motion.resolve_arm(studio_export=Path(args.studio_export), name=args.robot, port=None,
                                               calibration_path=None, follower=True)
        with motion.claim_devices(motion.device_ids_for(port), ports=(port,)):
            if args.teardown:
                arm = motion.open_passive(port, calibration)  # torque off at the current (home) pose
                arm.disconnect()
            else:
                steps = motion.load_sequences(args.poses)[args.sequence]
                poses = motion.load_poses(args.poses)
                arm = motion.open_follower_held(port, calibration)  # goal = present pose, then torque on
                try:
                    player = motion.MotionPlayer(arm, max_speed=motion.DEFAULT_MAX_SPEED, rate_hz=motion.DEFAULT_RATE_HZ,
                                                 tolerance=motion.DEFAULT_TOLERANCE)
                    player.run(steps, poses)
                    payload["gripper"] = float(player.current()[5])
                finally:
                    arm.disconnect()  # torque_on_disconnect keeps a held arm holding between segments
        payload["ok"] = True
        code = 0
    except motion.DeviceOwnedError as error:
        failure, code = error, EXIT_BUSY
    except motion.MotionStalledError as error:
        failure, code = error, EXIT_STALLED
    except (motion.MotionError, FileNotFoundError, KeyError) as error:
        failure, code = error, EXIT_CONFIG
    except (ConnectionError, RuntimeError, OSError) as error:
        failure, code = error, EXIT_HARDWARE
    payload["latency_ms"] = round((time.perf_counter() - began) * 1000.0, 1)
    print(json.dumps(payload))
    if failure is not None:
        print(f"{type(failure).__name__}: {failure}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(child())
