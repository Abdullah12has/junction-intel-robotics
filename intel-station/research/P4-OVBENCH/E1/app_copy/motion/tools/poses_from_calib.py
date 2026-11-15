"""Turn a calibration JSON into the poses YAML read by app/motion and app/loop.

Input (--calib): a JSON object in the driver's normalized unit (body joints [-100, 100], gripper [0, 100]):
  {"unit": "normalized", "home": [6], "spot_release": [6], "hover": [6],
   "blue_drop": [6], "green_drop": [6], "gripper": {"open": v, "closed": v}}
Joint order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
spot_release becomes spot_grasp; hover becomes spot_hover. Nothing else is invented:
lift_closed is the hover pose with the gripper closed.

Output (--out): poses YAML in the app/motion format with placeholder: true, and the sequences
spot_to_blue and spot_to_green: home -> spot_hover -> spot_grasp_open -> spot_grasp_closed (allow_stall)
-> lift_closed -> blue_drop_closed -> blue_drop_open -> home (and the green equivalent).
Speed is DEFAULT_MAX_SPEED (25.0 units/s), the motion code's default. dwell_s is 0.5 at the grasp and release points.

Exit codes: 0 written, 4 input or validation error (nothing written).
An existing --out with no placeholder: true (taught poses) is refused unless --force is given.

Example:
  python3 -I app/motion/tools/poses_from_calib.py --calib CALIB.json --out app/loop/poses.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import yaml

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
BODY_LIMIT = 100.0
GRIPPER_LOW, GRIPPER_HIGH = 0.0, 100.0
DEFAULT_SPEED = 25.0  # so101_motion.DEFAULT_MAX_SPEED
HARD_MAX_SPEED = 50.0  # so101_motion.HARD_MAX_SPEED
DWELL_S = 0.5
POSE_KEYS = ("home", "spot_release", "hover", "blue_drop", "green_drop", "gripper")


class CalibError(ValueError):
    """Input or validation failure. Nothing is written."""


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        msg = f"{label} must be a finite number, got {value!r}"
        raise CalibError(msg)
    return float(value)


def check_gripper(value: object, label: str) -> float:
    number = _number(value, label)
    if not GRIPPER_LOW <= number <= GRIPPER_HIGH:
        msg = f"{label} must lie in [0, 100], got {number}"
        raise CalibError(msg)
    return number


def check_pose(name: str, values: object) -> list[float]:
    if not isinstance(values, list) or len(values) != len(JOINTS):
        msg = f"{name} needs {len(JOINTS)} values in the order {', '.join(JOINTS)}"
        raise CalibError(msg)
    out = [_number(value, f"{name}.{joint}") for joint, value in zip(JOINTS, values, strict=True)]
    for joint, value in zip(JOINTS[:-1], out[:-1], strict=True):
        if abs(value) > BODY_LIMIT:
            msg = f"{name}.{joint} must lie in [-100, 100], got {value}"
            raise CalibError(msg)
    out[-1] = check_gripper(out[-1], f"{name}.gripper")
    return out


def load_calib(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"{path}: cannot read as JSON ({error})"
        raise CalibError(msg) from error
    if not isinstance(data, dict):
        msg = f"{path}: top level must be a JSON object"
        raise CalibError(msg)
    if data.get("unit") != "normalized":
        msg = f"{path}: unit must be 'normalized', got {data.get('unit')!r}; convert the calibration first"
        raise CalibError(msg)
    missing = [key for key in POSE_KEYS if key not in data]
    if missing:
        msg = f"{path}: missing {missing}; teach those poses before running this tool"
        raise CalibError(msg)
    return data


def build(calib: dict[str, object]) -> dict[str, list[float]]:
    grip = calib["gripper"]
    if not isinstance(grip, dict) or set(grip) != {"open", "closed"}:
        msg = 'gripper must be an object {"open": value, "closed": value}'
        raise CalibError(msg)
    open_value = check_gripper(grip["open"], "gripper.open")
    closed_value = check_gripper(grip["closed"], "gripper.closed")
    if open_value == closed_value:
        msg = "gripper open and closed values are equal; they must differ"
        raise CalibError(msg)
    home = check_pose("home", calib["home"])
    spot = check_pose("spot_release", calib["spot_release"])
    hover = check_pose("hover", calib["hover"])

    def with_gripper(pose: list[float], value: float) -> list[float]:
        return [*pose[:-1], value]

    poses = {
        "home": home,
        "spot_hover": with_gripper(hover, open_value),
        "spot_grasp_open": with_gripper(spot, open_value),
        "spot_grasp_closed": with_gripper(spot, closed_value),
        "lift_closed": with_gripper(hover, closed_value),
    }
    for side in ("blue", "green"):
        drop = check_pose(f"{side}_drop", calib[f"{side}_drop"])
        poses[f"{side}_drop_closed"] = with_gripper(drop, closed_value)
        poses[f"{side}_drop_open"] = with_gripper(drop, open_value)
    return poses


def steps(side: str) -> list[dict[str, object]]:
    return [
        {"pose": "home"},
        {"pose": "spot_hover"},
        {"pose": "spot_grasp_open", "dwell_s": DWELL_S},
        {"pose": "spot_grasp_closed", "dwell_s": DWELL_S, "allow_stall": True},
        {"pose": "lift_closed"},
        {"pose": f"{side}_drop_closed"},
        {"pose": f"{side}_drop_open", "dwell_s": DWELL_S},
        {"pose": "home"},
    ]


def _render_step(step: dict[str, object]) -> str:
    fields = [f"pose: {step['pose']}", f"speed: {DEFAULT_SPEED:.1f}"]
    if "dwell_s" in step:
        fields.append(f"dwell_s: {float(step['dwell_s']):.1f}")
    if step.get("allow_stall"):
        fields.append("allow_stall: true")
    return "    - {" + ", ".join(fields) + "}"


def render(poses: dict[str, list[float]], digest: str) -> str:
    lines = [
        f"# Generated by poses_from_calib.py from a calibration JSON (sha256 {digest}). Normalized units.",
        "# placeholder: true until Opus verifies the poses live; run_sequence.py refuses motion while it is set.",
        "placeholder: true",
        "unit: normalized",
        f"joint_order: [{', '.join(JOINTS)}]",
        "poses:",
    ]
    for name, values in poses.items():
        lines.append(f"  {name}: [{', '.join(f'{value:.2f}' for value in values)}]")
    lines.append("sequences:")
    for side in ("blue", "green"):
        lines.append(f"  spot_to_{side}:")
        lines.extend(_render_step(step) for step in steps(side))
    return "\n".join(lines) + "\n"


def validate_text(text: str) -> None:
    doc = yaml.safe_load(text)
    if doc.get("placeholder") is not True or doc.get("unit") != "normalized" or doc.get("joint_order") != list(JOINTS):
        msg = "generated header failed the check"
        raise CalibError(msg)
    poses = {name: check_pose(name, values) for name, values in doc["poses"].items()}
    for seq_name, items in doc["sequences"].items():
        for item in items:
            if item["pose"] not in poses:
                msg = f"sequence {seq_name} uses unknown pose {item['pose']}"
                raise CalibError(msg)
            if not 0.0 < float(item["speed"]) <= HARD_MAX_SPEED:
                msg = f"sequence {seq_name} speed {item['speed']} is outside (0, {HARD_MAX_SPEED:g}]"
                raise CalibError(msg)


def refuse_taught(out: Path, force: bool) -> None:
    if force or not out.exists():
        return
    existing = yaml.safe_load(out.read_text(encoding="utf-8"))
    if not isinstance(existing, dict) or existing.get("placeholder") is not True:
        msg = f"{out} holds taught poses (placeholder is not true); pass --force to replace them"
        raise CalibError(msg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the app/motion poses YAML from a calibration JSON.")
    parser.add_argument("--calib", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--force", action="store_true", help="replace an existing file with taught poses")
    args = parser.parse_args(argv)
    try:
        digest = hashlib.sha256(args.calib.read_bytes()).hexdigest()
        poses = build(load_calib(args.calib))
        text = render(poses, digest)
        validate_text(text)
        refuse_taught(args.out, args.force)
    except (CalibError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 4
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}: {len(poses)} poses, sequences spot_to_blue and spot_to_green, placeholder: true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
