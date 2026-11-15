"""Teach named SO-101 poses into the poses YAML file.

Hand mode: the follower connects with torque OFF. Move the arm by hand, then save.
With torque off the arm falls under gravity (so101.py:431-455), so support it first.

Leader mode: the leader arm connects with torque OFF. Move the leader and save its
normalized pose. The follower uses leader values the same way Studio teleop does
(physicalai/robot/interface.py:116).

Examples:
  python3 -I app/motion/teach_poses.py --studio-export project_robots.json --out poses.yaml
  python3 -I app/motion/teach_poses.py --studio-export project_robots.json --mode leader \
      --leader-robot "SO101 Leader" --out poses.yaml
  python3 -I app/motion/teach_poses.py --calibration cal.json --capture rest --out poses.yaml
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

if TYPE_CHECKING:
    from physicalai_so101_plugin.calibration import SO101Calibration

from so101_motion import (  # noqa: E402
    DEFAULT_BAUDRATE,
    JOINT_ORDER,
    DeviceOwnedError,
    MotionError,
    claim_devices,
    device_ids_for,
    hold_and_enable_torque,
    is_placeholder,
    open_passive,
    resolve_arm,
    write_pose,
)

NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teach named SO-101 poses (normalized units).")
    parser.add_argument("--out", required=True, type=Path, help="poses YAML to create or update")
    parser.add_argument("--mode", choices=("hand", "leader"), default="hand", help="which arm to read")
    parser.add_argument("--studio-export", type=Path, help="Studio project robots JSON (project_robots.json)")
    parser.add_argument("--robot", default="SO101 Follower", help="robot name inside --studio-export")
    parser.add_argument("--leader-robot", default="SO101 Leader", help="leader name inside --studio-export")
    parser.add_argument("--port", help="follower serial port; only the by-id path is accepted")
    parser.add_argument("--calibration", type=Path, help="follower calibration JSON (LeRobot format)")
    parser.add_argument("--leader-port", help="leader serial port (leader mode without an export)")
    parser.add_argument("--leader-calibration", type=Path, help="leader calibration JSON")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--capture", metavar="NAME", help="save one pose non-interactively and exit")
    return parser.parse_args(argv)


def read_normalized(driver: object) -> np.ndarray:
    """Return the present pose from a connected driver (get_observation, so101.py:487-519)."""
    return np.asarray(driver.get_observation().joint_positions, dtype=np.float64)  # type: ignore[attr-defined]


def format_pose(values: np.ndarray) -> str:
    return "  ".join(f"{name}={value:.2f}" for name, value in zip(JOINT_ORDER, values, strict=True))


def teach_loop(
    read: Callable[[], np.ndarray],
    out: Path,
    *,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> list[str]:
    """Save poses interactively: a valid name saves the current pose, '?' shows it, 'q' quits.

    Returns:
        Names saved during this session, in order.
    """
    saved: list[str] = []
    print_fn("Type a pose name and press Enter to save the current pose. '?' shows it, 'q' quits.")
    while True:
        try:
            line = input_fn("pose> ").strip()
        except EOFError:
            break
        if line in {"q", "quit", "exit"}:
            break
        if line in {"", "?"}:
            print_fn(format_pose(read()))
            continue
        if not NAME_RE.fullmatch(line):
            print_fn("Names start with a letter and use letters, digits, '_' or '-' (max 64).")
            continue
        values = read()
        write_pose(out, line, values)
        saved.append(line)
        print_fn(f"saved {line}: {format_pose(values)}")
    return saved


def resolve(args: argparse.Namespace) -> tuple[str, SO101Calibration]:
    """Return the (port, calibration) of the arm whose pose is read in the chosen mode."""
    if args.mode == "leader":
        return resolve_arm(
            studio_export=args.studio_export,
            name=args.leader_robot,
            port=args.leader_port,
            calibration_path=args.leader_calibration,
        )
    return resolve_arm(
        studio_export=args.studio_export,
        name=args.robot,
        port=args.port,
        calibration_path=args.calibration,
        follower=True,
    )


def run_teaching(args: argparse.Namespace, read_port: str, read_cal: SO101Calibration) -> int:
    """Open the arm that is read (follower in hand mode, leader in leader mode) and save poses from it.

    In hand mode the follower is held with torque on when teaching ends, so it does not drop
    under gravity after this script exits (the driver's disconnect() only holds a follower
    that was connected with torque_on_disconnect True).
    """
    with claim_devices(device_ids_for(read_port), ports=(read_port,)):
        arm = open_passive(read_port, read_cal, baudrate=args.baudrate)
        try:
            if args.capture is not None:
                values = read_normalized(arm)
                write_pose(args.out, args.capture, values)
                print(f"saved {args.capture}: {format_pose(values)}")
                return 0
            saved = teach_loop(lambda: read_normalized(arm), args.out)
            print(f"saved {len(saved)} pose(s) to {args.out}: {', '.join(saved) or 'none'}")
            if args.out.exists() and is_placeholder(args.out):
                print(f"{args.out} still has placeholder: true; remove that key once every pose is taught")
            return 0
        finally:
            try:
                if args.mode == "hand":
                    hold_and_enable_torque(arm)
            finally:
                arm.disconnect()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.capture is not None and not NAME_RE.fullmatch(args.capture):
        print(f"invalid pose name {args.capture!r}", file=sys.stderr)
        return 4
    try:
        port, calibration = resolve(args)
        return run_teaching(args, port, calibration)
    except DeviceOwnedError as error:
        print(f"device busy: {error}", file=sys.stderr)
        return 3
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
