"""Routing for loop_v2 (LOOP-CONTRACT section 8; I-52, OD-10, D-058, OD-11b).

Pure functions with no I/O: the scan target, triage (CONFIDENT_DEFECT straight to GREEN), fusion of the still-image deep
check, the fail-safe decision, the release side read from the shoulder pan, the verify record, and the scripted-mode rule
that skips the inspection pose only for the scanned brick that is actually picked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

GOOD = "GOOD"
DEFECTIVE = "DEFECTIVE"
UNCERTAIN = "UNCERTAIN"
CONFIDENT_DEFECT = "CONFIDENT_DEFECT"
GREEN_DIRECT = "GREEN_DIRECT"
INSPECT = "INSPECT"
BLUE = "BLUE"
GREEN = "GREEN"
NONE = "NONE"
PLATE_FOR_LABEL = {GOOD: BLUE, DEFECTIVE: GREEN, UNCERTAIN: GREEN}  # OD-10 (final, ledger 13:47Z)
SKILL_FOR_PLATE = {BLUE: "S2", GREEN: "S3"}  # LOOP-CONTRACT section 5 skill names
PAN_BLUE_ABOVE = 0.0  # release pan above this is the BLUE side (FACTS-LIVE 13:52Z: BLUE release pan +32.1)
PAN_GREEN_BELOW = -30.0  # release pan below this is the GREEN side (FACTS-LIVE 13:52Z: GREEN -51.8 to -57.3) [unclear: calibrate]
MATCH_TOL_PX = 40.0  # [unclear] distance that still counts as the same brick between two scans


@dataclass(frozen=True)
class ScanReport:
    """One DET2 scan: the brick dicts and the hand flags of both views. None means the scan did not report (refused)."""

    bricks: tuple[Mapping[str, Any], ...]
    hand_top: bool | None = None
    hand_close: bool | None = None


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def hands_in_view(hand_top: bool | None, hand_close: bool | None) -> list[str]:
    """Views that refuse motion: a hand seen, or no report at all (None). Fail-safe."""
    return [name for name, flag in (("top", hand_top), ("close", hand_close)) if flag is not False]


def centre_of(brick: Mapping[str, Any]) -> tuple[float, float] | None:
    centre = brick.get("centre_px")
    if not isinstance(centre, (list, tuple)) or len(centre) != 2:
        return None
    x, y = centre
    if not (finite_number(x) and finite_number(y)):
        return None
    return float(x), float(y)


def most_isolated(
    bricks: Sequence[Mapping[str, Any]], universe: Sequence[Mapping[str, Any]] | None = None
) -> Mapping[str, Any] | None:
    """The brick whose centre is farthest from every other brick of universe (default: bricks). The first one wins a tie."""
    pool = list(universe if universe is not None else bricks)
    best: Mapping[str, Any] | None = None
    best_gap = -1.0
    for brick in bricks:
        centre = centre_of(brick)
        if centre is None:
            continue
        gap = math.inf
        for other in pool:
            other_centre = None if other is brick else centre_of(other)
            if other_centre is not None:
                gap = min(gap, math.hypot(centre[0] - other_centre[0], centre[1] - other_centre[1]))
        if best is None or gap > best_gap:
            best, best_gap = brick, gap
    return best


def find_target(bricks: Sequence[Mapping[str, Any]], colour: str = "red") -> tuple[Mapping[str, Any] | None, str]:
    """The team's 'the red brick' (DET2 find_target), else the most isolated brick (OD-12). Returns (brick, how)."""
    reds = [brick for brick in bricks if str(brick.get("colour", "")).lower() == colour]
    if reds:
        red = most_isolated(reds, bricks)
        if red is not None:
            return red, colour
    fallback = most_isolated(bricks)
    if fallback is not None:
        return fallback, "most_isolated"
    return None, "none"


def triage(target: Mapping[str, Any]) -> tuple[str, list[str]]:
    """I-52: a CONFIDENT_DEFECT from any view goes straight to GREEN. Every other verdict, GOOD included, is inspected."""
    verdict = str(target.get("verdict", ""))
    if verdict == CONFIDENT_DEFECT:
        return GREEN_DIRECT, ["confident_defect", *(str(reason) for reason in target.get("reasons") or [])]
    return INSPECT, [f"verdict:{verdict or 'missing'}"]


def fuse(per_camera: Mapping[str, Mapping[str, Any]], margin: float = 0.1) -> tuple[str, list[str]]:
    """Fusion of the still-image deep check (D-058, LOOP-CONTRACT section 7).

    UNCERTAIN when a camera saw no brick. DEFECTIVE when any camera says DEFECTIVE. UNCERTAIN when a camera is unsure
    or a GOOD score sits within margin of its threshold. Otherwise GOOD.
    """
    if not per_camera:
        return UNCERTAIN, ["no_camera_result"]
    no_brick = [name for name, result in per_camera.items() if "no_brick" in (result.get("reasons") or [])]
    if no_brick:
        return UNCERTAIN, [f"{name}:no_brick" for name in no_brick]
    labels = {name: str(result.get("label", UNCERTAIN)) for name, result in per_camera.items()}
    defective = [name for name, label in labels.items() if label == DEFECTIVE]
    if defective:
        return DEFECTIVE, [f"{name}:defective" for name in defective]
    unsure = [name for name, label in labels.items() if label != GOOD]
    if unsure:
        return UNCERTAIN, [f"{name}:{labels[name].lower()}" for name in unsure]
    for name, result in per_camera.items():
        score, threshold = result.get("score"), result.get("threshold")
        if finite_number(score) and finite_number(threshold) and threshold > 0 and score >= threshold * (1.0 - margin):
            return UNCERTAIN, [f"{name}:near_threshold"]
    return GOOD, []


def decide(target_id: str | None, route: str, inspected: str | None) -> dict[str, Any]:
    """OD-10 with fail-safe routing: only an inspected GOOD reaches BLUE. Direct defects and unproven bricks go to GREEN."""
    if route == GREEN_DIRECT:
        plate, via = GREEN, "direct"
    elif inspected == GOOD:
        plate, via = BLUE, "inspected"
    elif inspected in (DEFECTIVE, UNCERTAIN):
        plate, via = GREEN, "inspected"
    else:
        plate, via = GREEN, "fail_safe"
    return {"target_id": target_id, "plate": plate, "rule": "OD-10", "via": via}


def release_side(pan: Any) -> str:
    """Plate row of the release, from the shoulder pan at release: pan above 0 is BLUE, below -30 is GREEN, else NONE."""
    if not finite_number(pan):
        return NONE
    if pan > PAN_BLUE_ABOVE:
        return BLUE
    if pan < PAN_GREEN_BELOW:
        return GREEN
    return NONE


def verify_result(target_id: str | None, plate_intended: str, pan_at_release: Any, frame_top: str | None) -> dict[str, Any]:
    side = release_side(pan_at_release)
    return {
        "target_id": target_id,
        "plate_intended": plate_intended,
        "release_side": side,
        "ok": side == plate_intended,
        "frame_top": frame_top,
    }


def match_target(
    bricks: Sequence[Mapping[str, Any]], target: Mapping[str, Any], scale: float = 1.0, tol_px: float = MATCH_TOL_PX
) -> Mapping[str, Any] | None:
    """The brick of another scan (the 1280x720 stills, scale 2.0) whose centre is the target's centre, or None."""
    centre = centre_of(target)
    if centre is None:
        return None
    best: Mapping[str, Any] | None = None
    best_distance = math.inf
    for brick in bricks:
        other = centre_of(brick)
        if other is None:
            continue
        distance = math.hypot(other[0] - centre[0] * scale, other[1] - centre[1] * scale)
        if distance < best_distance:
            best, best_distance = brick, distance
    return best if best_distance <= tol_px else None


def may_skip_inspection(
    route: str, scanned_centre: tuple[float, float] | None, picked_centre: tuple[float, float] | None,
    tol_px: float = MATCH_TOL_PX,
) -> bool:
    """Scripted mode (OD-11b): the inspection pose is skipped only for a CONFIDENT_DEFECT brick that is the one picked."""
    if route != GREEN_DIRECT or scanned_centre is None or picked_centre is None:
        return False
    return math.hypot(scanned_centre[0] - picked_centre[0], scanned_centre[1] - picked_centre[1]) <= tol_px
