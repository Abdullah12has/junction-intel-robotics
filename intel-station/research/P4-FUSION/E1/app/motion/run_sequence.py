"""Run a named sequence from the poses YAML on the SO-101 follower.

Default is a dry run: the plan (setpoints, duration, peak rate per step) is computed and
printed, and no serial port is opened. The arm moves only with --confirm-motion.

Before moving, the follower is connected so that its goal equals its present pose
(open_follower_held). The device lock is taken first, so a running Studio session makes
this script exit with code 3 instead of sharing the port.
The follower port is always the by-id link in so101_motion.FOLLOWER_PORT_BY_ID. The export's
connection_string for the follower is stale and ignored.

A step that ends outside the tolerance stops the sequence (later steps are not sent) unless
the step sets allow_stall: true. A poses file with the top-level key placeholder: true
is refused for motion until its poses are taught and the key is removed.

Limits: every speed is at most 50 units/s (HARD_MAX_SPEED); a larger --max-speed or YAML speed
exits 4. Stop the Studio runtime session first: the port is refused while another process of this
user has it open. Before the first --confirm-motion run, hand-check which gripper value is open and
which is closed (poses.example.yaml marks it unclear). That is a physical step this script does not check.

Exit codes: 0 ok, 2 a step was not reached (sequence stopped), 3 device owned by another
process, 4 configuration or validation error, 5 hardware error.

Examples:
  python3 -I app/motion/run_sequence.py --poses poses.yaml --list
  python3 -I app/motion/run_sequence.py --poses poses.yaml --sequence pick_to_inspection
  python3 -I app/motion/run_sequence.py --poses poses.yaml --sequence pick_to_inspection \
      --studio-export project_robots.json --confirm-motion
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from so101_motion import (  # noqa: E402
    DEFAULT_BAUDRATE,
    DEFAULT_MAX_SPEED,
    DEFAULT_RATE_HZ,
    DEFAULT_TOLERANCE,
    DeviceOwnedError,
    MotionError,
    MotionPlayer,
    MotionStalledError,
    check_speed,
    claim_devices,
    describe_plan,
    device_ids_for,
    is_placeholder,
    load_poses,
    load_sequences,
    open_follower_held,
    resolve_arm,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a named SO-101 sequence (normalized units).")
    parser.add_argument("--poses", required=True, type=Path, help="poses YAML with poses and sequences")
    parser.add_argument("--list", action="store_true", help="list sequence names and exit")
    parser.add_argument("--sequence", help="sequence name to run")
    parser.add_argument("--studio-export", type=Path, help="Studio project robots JSON (project_robots.json)")
    parser.add_argument("--robot", default="SO101 Follower", help="follower name inside --studio-export")
    parser.add_argument("--port", help="follower serial port; only the by-id path is accepted (the export's connection_string is ignored)")
    parser.add_argument("--calibration", type=Path, help="follower calibration JSON (LeRobot format)")
    parser.add_argument("--max-speed", type=float, default=DEFAULT_MAX_SPEED, help="normalized units per second, at most 50")
    parser.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ, help="setpoint rate")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE, help="allowed read-back error")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--start", nargs=6, type=float, metavar="V", help="assumed start pose for the dry-run plan")
    parser.add_argument("--confirm-motion", action="store_true", help="allow the arm to move")
    return parser.parse_args(argv)


def format_plan(rows: list[dict[str, object]]) -> str:
    header = (
        f"{'step':>4}  {'pose':<16} {'speed':>7} {'setpoints':>9} {'duration_s':>10} "
        f"{'peak_rate':>10} {'peak_accel':>11} {'dwell_s':>7}"
    )
    lines = [header]
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"{index:>4}  {row['pose']!s:<16} {row['speed']!s:>7} {row['setpoints']!s:>9} "
            f"{row['duration_s']!s:>10} {row['peak_rate_units_s']!s:>10} {row['peak_accel_units_s2']!s:>11} "
            f"{row['dwell_s']!s:>7}"
        )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    poses = load_poses(args.poses)
    sequences = load_sequences(args.poses)
    if args.list:
        for name, steps in sequences.items():
            print(f"{name}: {' -> '.join(step.pose for step in steps)}")
        return 0
    if not args.sequence:
        msg = "pass --sequence NAME (or --list)"
        raise MotionError(msg)
    if args.sequence not in sequences:
        msg = f"unknown sequence {args.sequence!r}; known: {sorted(sequences)}"
        raise MotionError(msg)
    steps = sequences[args.sequence]
    if not steps:
        msg = f"sequence {args.sequence!r} has no steps"
        raise MotionError(msg)
    max_speed = check_speed(args.max_speed, "--max-speed")

    if not args.confirm_motion:
        start = np.asarray(args.start, dtype=np.float64) if args.start else poses.get("rest", np.zeros(6))
        rows = describe_plan(steps, poses, start=start, max_speed=max_speed, rate_hz=args.rate_hz)
        print(format_plan(rows))
        print("dry run: no serial port was opened; add --confirm-motion to move the arm")
        return 0

    if is_placeholder(args.poses):
        msg = f"{args.poses} has placeholder: true; teach the poses, then remove that key before --confirm-motion"
        raise MotionError(msg)

    port, calibration = resolve_arm(
        studio_export=args.studio_export,
        name=args.robot,
        port=args.port,
        calibration_path=args.calibration,
        follower=True,
    )
    with claim_devices(device_ids_for(port), ports=(port,)):
        driver = open_follower_held(port, calibration, baudrate=args.baudrate)
        try:
            player = MotionPlayer(driver, max_speed=max_speed, rate_hz=args.rate_hz, tolerance=args.tolerance)
            results = player.run(steps, poses)
        finally:
            driver.disconnect()
    for index, (step, result) in enumerate(zip(steps, results, strict=True), start=1):
        print(
            f"step {index} {step.pose}: arrived={result.arrived} max_error={result.max_error:.2f} "
            f"duration_s={result.duration_s:.2f}"
        )
    reached = all(result.arrived or step.allow_stall for step, result in zip(steps, results, strict=True))
    return 0 if reached else 2


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except DeviceOwnedError as error:
        print(f"device busy: {error}", file=sys.stderr)
        return 3
    except MotionStalledError as error:
        for index, result in enumerate(error.results, start=1):
            print(f"step {index}: arrived={result.arrived} max_error={result.max_error:.2f}")
        print(f"stalled: {error}", file=sys.stderr)
        return 2
    except MotionError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 4
    except FileNotFoundError as error:
        print(f"file not found: {error}", file=sys.stderr)
        return 4
    except ImportError as error:
        print(f"missing dependency: {error}. Install physicalai-so101-plugin (brings feetech-servo-sdk).", file=sys.stderr)
        return 4
    except (ConnectionError, RuntimeError, OSError) as error:
        print(f"hardware error: {error}", file=sys.stderr)
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
