"""ACT and HOME monitors for policy mode (LOOP-CONTRACT section 8). Pure, no I/O.

Samples are (t, state). t is the observation timestamp and only differences of t are used. state maps joint names
(for example 'gripper.pos') to values. grasp: gripper below 4 for 0.3 s after an approach. release: gripper at or above 6
after a grasp. return: the reference joints within 10 units of home and still for 1.0 s. timeout: 45 s from the first sample.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

GRIPPER = "gripper.pos"
PAN = "shoulder_pan.pos"
REF_JOINTS = ("shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos")  # rules reference.home_pose keys
APPROACH_UNITS = 3.0  # [unclear] movement from home that counts as an approach


def _value(state: Mapping[str, Any] | None, key: str) -> float | None:
    if not state:
        return None
    value = state.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


class HomeStill:
    """True once the reference joints are within tol of home and have moved no more than eps per sample for still_s."""

    def __init__(self, home: Mapping[str, float], *, tol: float = 10.0, still_s: float = 1.0, eps: float = 0.5) -> None:
        self.home = {key: float(home[key]) for key in REF_JOINTS}
        self.tol = tol
        self.still_s = still_s
        self.eps = eps
        self._prev: dict[str, float] | None = None
        self._still_since: float | None = None

    def feed(self, t: float, state: Mapping[str, Any] | None) -> bool:
        values = {key: _value(state, key) for key in REF_JOINTS}
        if any(value is None for value in values.values()):
            self._prev, self._still_since = None, None
            return False
        current = {key: float(values[key]) for key in REF_JOINTS}
        moved = 0.0 if self._prev is None else max(abs(current[k] - self._prev[k]) for k in REF_JOINTS)
        self._prev = current
        near = all(abs(current[k] - self.home[k]) <= self.tol for k in REF_JOINTS)
        if not near or moved > self.eps:
            self._still_since = None
            return False
        if self._still_since is None:
            self._still_since = t
        return t - self._still_since >= self.still_s


class ActMonitor:
    """Phases of one policy run: approach, carry (after grasp), return (after release), done or timed out."""

    def __init__(
        self,
        home: Mapping[str, float],
        *,
        home_tol: float = 10.0,
        still_s: float = 1.0,
        eps: float = 0.5,
        grasp_below: float = 4.0,
        grasp_hold_s: float = 0.3,
        release_at: float = 6.0,
        timeout_s: float = 45.0,
    ) -> None:
        self.home = {key: float(home[key]) for key in REF_JOINTS}
        self._home = HomeStill(home, tol=home_tol, still_s=still_s, eps=eps)
        self.grasp_below = grasp_below
        self.grasp_hold_s = grasp_hold_s
        self.release_at = release_at
        self.timeout_s = timeout_s
        self.phase = "approach"
        self.t0: float | None = None
        self.release_pan: float | None = None
        self.done = False
        self.timed_out = False
        self._below_since: float | None = None
        self._approached = False

    def _moved_from_home(self, state: Mapping[str, Any] | None) -> bool:
        for key in REF_JOINTS:
            value = _value(state, key)
            if value is not None and abs(value - self.home[key]) > APPROACH_UNITS:
                return True
        return False

    def feed(self, t: float, state: Mapping[str, Any] | None) -> list[tuple[str, dict[str, Any]]]:
        if self.done or self.timed_out:
            return []
        if self.t0 is None:
            self.t0 = t
        if t - self.t0 > self.timeout_s:
            self.timed_out = True
            return [("timeout", {"elapsed_s": round(t - self.t0, 3), "at_phase": self.phase})]
        events: list[tuple[str, dict[str, Any]]] = []
        gripper = _value(state, GRIPPER)
        if self.phase == "approach":
            if self._moved_from_home(state):
                self._approached = True
            if self._approached and gripper is not None and gripper < self.grasp_below:
                if self._below_since is None:
                    self._below_since = t
                if t - self._below_since >= self.grasp_hold_s:
                    self.phase = "carry"
                    events.append(("grasp", {"gripper": gripper}))
                    events.append(("carry", {}))
            else:
                self._below_since = None
        elif self.phase == "carry" and gripper is not None and gripper >= self.release_at:
            self.release_pan = _value(state, PAN)
            self.phase = "return"
            events.append(("release", {"pan": self.release_pan}))
            events.append(("return", {}))
        if self.phase == "return" and self._home.feed(t, state):
            self.done = True
            events.append(("done", {}))
        return events
