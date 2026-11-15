"""Policy-mode view state (LOOP-CONTRACT section 8, OD-12, I-52): scan, triage, decision, policy_run, verify.

Folds those events, and the policy skill configuration of run_start, into what the page shows. Pure: no file
access; the caller passes the URL builder for run-dir images. Every number here is computed from the events:
per-step latency is the time between two policy_run events of one cycle (their t), per-cycle latency is the
time from decision to verify, per-inspection latency is the latency_ms the inspection event carries.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any, Callable

CONFIDENT_DEFECT = "CONFIDENT_DEFECT"
ROUTE_TEXT = {"GREEN_DIRECT": "direct to GREEN (I-52)", "INSPECT": "inspected"}
BRICK_ROUTE_TEXT_DIRECT = "straight to GREEN (I-52)"
BRICK_ROUTE_TEXT_INSPECTED = "inspected"
VIA_TEXT = {"direct": "direct", "inspected": "inspected", "fail_safe": "fail-safe"}
PLATES = ("BLUE", "GREEN")
MAX_CYCLES = 500
MAX_SAMPLES = 5000
POLICY_KINDS = ("scan", "triage", "decision", "policy_run", "verify")


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def parse_t(text: Any) -> float | None:
    """Epoch seconds of an ISO 8601 UTC stamp such as 2026-11-14T12:00:00.000Z; None if unparseable."""
    if not isinstance(text, str) or not text:
        return None
    try:
        moment = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=dt.timezone.utc)
    return moment.timestamp()


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolation percentile (q in 0..100) of the finite values; None when there are none."""
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def latency_stats(values: list[float]) -> dict[str, Any]:
    """n, p50, p95 and max in milliseconds, rounded to 0.01 ms; the p50/p95 are None until a value exists."""
    xs = [v for v in values if math.isfinite(v)]

    def rounded(number: float | None) -> float | None:
        return None if number is None else round(number, 2)

    return {
        "n": len(xs),
        "p50": rounded(percentile(xs, 50)),
        "p95": rounded(percentile(xs, 95)),
        "max": rounded(max(xs)) if xs else None,
    }


def _brick(raw: Any, run_url: Callable[[Any], str | None]) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    verdict = (_text(raw.get("verdict")) or "").upper() or None
    centre = raw.get("centre_px")
    centre_px = None
    if isinstance(centre, (list, tuple)) and len(centre) == 2:
        pair = [_num(c) for c in centre]
        if pair[0] is not None and pair[1] is not None:
            centre_px = pair
    reasons = raw.get("reasons")
    crop_close = run_url(raw.get("crop_close"))
    crop_top = run_url(raw.get("crop_top"))
    direct = verdict == CONFIDENT_DEFECT
    return {
        "id": None if raw.get("id") is None else str(raw.get("id")),
        "colour": _text(raw.get("colour")),
        "centre_px": centre_px,
        "angle_deg": _num(raw.get("angle_deg")),
        "shape": _text(raw.get("shape")),
        "shape_conf": _num(raw.get("shape_conf")),
        "tape_score": _num(raw.get("tape_score")),
        "tape_conf": _num(raw.get("tape_conf")),
        "verdict": verdict,
        "confident": raw.get("confident") is True,
        "reasons": [str(r) for r in reasons] if isinstance(reasons, list) else [],
        "crop_url": crop_close or crop_top,
        "crop_top_url": crop_top,
        "crop_close_url": crop_close,
        "route": BRICK_ROUTE_TEXT_DIRECT if direct else BRICK_ROUTE_TEXT_INSPECTED,
        "direct_green": direct,
    }


class PolicyState:
    """Per-run policy view: latest scan, and per cycle the triage, inspection, decision, phases and verify."""

    def __init__(self, run_url: Callable[[Any], str | None]) -> None:
        self.run_url = run_url
        self.plate_cfg: dict[str, dict[str, Any]] = {}
        self.cycles: dict[int, dict[str, Any]] = {}
        self.last_scan: dict[str, Any] | None = None
        self.step_ms: list[float] = []
        self.cycle_ms: list[float] = []
        self.inspection_ms: list[float] = []

    # ----- configuration -------------------------------------------------------------------------

    def read_config(self, cfg: Any) -> None:
        """run_start.config: the policy emitter writes plates {BLUE|GREEN: skill}; section 9 names skills."""
        if not isinstance(cfg, dict):
            return
        plates = cfg.get("plates")
        if isinstance(plates, dict):
            for name, skill in plates.items():
                if isinstance(skill, dict) and str(name).upper() in PLATES:
                    self.plate_cfg[str(name).upper()] = {
                        "skill": _text(skill.get("skill")),
                        "model_id": _text(skill.get("model_id")),
                        "task": _text(skill.get("task")),
                        "backend": _text(skill.get("backend")),
                        "device": _text(skill.get("device")),
                    }

    def _device_for(self, plate: str | None) -> str | None:
        return (self.plate_cfg.get(plate or "") or {}).get("device")

    # ----- events --------------------------------------------------------------------------------

    def _slot(self, cycle: int) -> dict[str, Any]:
        if cycle not in self.cycles:
            self.cycles[cycle] = {"cycle": cycle, "triage": None, "inspection": None, "decision": None,
                                  "phases": [], "verify": None, "t": None}
            if len(self.cycles) > MAX_CYCLES:
                del self.cycles[min(self.cycles)]
        return self.cycles[cycle]

    @staticmethod
    def _cycle_of(ev: dict[str, Any], fallback: int) -> int:
        cycle = _int(ev.get("cycle"))
        return fallback if cycle is None else cycle

    def apply(self, kind: str, ev: dict[str, Any], t: str | None, fallback_cycle: int) -> None:
        cycle = self._cycle_of(ev, fallback_cycle)
        if kind == "scan":
            bricks = ev.get("bricks")
            parsed = [b for b in (_brick(raw, self.run_url) for raw in bricks) if b] if isinstance(bricks, list) else []
            self.last_scan = {
                "cycle": cycle,
                "t": t,
                "target_id": None if ev.get("target_id") is None else str(ev.get("target_id")),
                "target_how": _text(ev.get("target_how")),
                "bricks": parsed,
                "frame_top_url": self.run_url(ev.get("frame_top")),
                "frame_close_url": self.run_url(ev.get("frame_close")),
            }
        elif kind == "triage":
            route = (_text(ev.get("route")) or "").upper() or None
            self._slot(cycle)["triage"] = {
                "target_id": None if ev.get("target_id") is None else str(ev.get("target_id")),
                "route": route,
                "route_text": ROUTE_TEXT.get(route or "", route or "-"),
                "reasons": [str(r) for r in ev["reasons"]] if isinstance(ev.get("reasons"), list) else [],
                "t": t,
            }
        elif kind == "decision":
            plate = (_text(ev.get("plate")) or "").upper() or None
            via = _text(ev.get("via"))
            slot = self._slot(cycle)
            slot["decision"] = {
                "target_id": None if ev.get("target_id") is None else str(ev.get("target_id")),
                "plate": plate,
                "rule": _text(ev.get("rule")) or "-",
                "via": via,
                "path": VIA_TEXT.get(via or "", via or "-"),
                "t": t,
            }
            slot["t"] = t
        elif kind == "policy_run":
            self._add_phase(cycle, ev, t)
        elif kind == "verify":
            self._add_verify(cycle, ev, t)

    def _add_phase(self, cycle: int, ev: dict[str, Any], t: str | None) -> None:
        slot = self._slot(cycle)
        plate = (_text(ev.get("plate")) or "").upper() or None
        phases = slot["phases"]
        previous_t = parse_t(phases[-1]["t"]) if phases else None
        this_t = parse_t(t)
        step = None
        if previous_t is not None and this_t is not None and this_t >= previous_t:
            step = round((this_t - previous_t) * 1000.0, 2)
            self.step_ms.append(step)
            del self.step_ms[:-MAX_SAMPLES]
        phases.append({
            "phase": _text(ev.get("phase")) or "-",
            "t": t,
            "model_id": _text(ev.get("model_id")),
            "task_text": _text(ev.get("task_text")),
            "plate": plate,
            "device": _text(ev.get("device")) or self._device_for(plate),
            "step_ms": step,
            "message": _text(ev.get("message")),
        })

    def _add_verify(self, cycle: int, ev: dict[str, Any], t: str | None) -> None:
        slot = self._slot(cycle)
        intended = (_text(ev.get("plate_intended")) or "").upper() or None
        side = (_text(ev.get("release_side")) or "").upper() or None
        slot["verify"] = {
            "target_id": None if ev.get("target_id") is None else str(ev.get("target_id")),
            "plate_intended": intended,
            "release_side": side,
            "ok": ev.get("ok") is True,
            "frame_top_url": self.run_url(ev.get("frame_top")),
            "t": t,
        }
        decided = parse_t(slot["decision"]["t"]) if slot["decision"] else None
        verified = parse_t(t)
        if decided is not None and verified is not None and verified >= decided:
            self.cycle_ms.append(round((verified - decided) * 1000.0, 2))
            del self.cycle_ms[:-MAX_SAMPLES]

    def note_inspection(self, ev: dict[str, Any], t: str | None, fallback_cycle: int) -> dict[str, Any]:
        """Keep the inspection of the cycle (mode still_hires or wrist_hover, result, reasons, latency)."""
        cycle = self._cycle_of(ev, fallback_cycle)
        latency = _num(ev.get("latency_ms"))
        if latency is not None:
            self.inspection_ms.append(latency)
            del self.inspection_ms[:-MAX_SAMPLES]
        record = {
            "cycle": cycle,
            "mode": _text(ev.get("mode")),
            "result": (_text(ev.get("result")) or _text(ev.get("label")) or "").upper() or None,
            "reasons": [str(r) for r in ev["reasons"]] if isinstance(ev.get("reasons"), list) else [],
            "target_id": None if ev.get("target_id") is None else str(ev.get("target_id")),
            "device": _text(ev.get("device")),
            "latency_ms": latency,
            "t": t,
        }
        self._slot(cycle)["inspection"] = record
        return record

    # ----- view ----------------------------------------------------------------------------------

    def snapshot(self, cycle: int) -> dict[str, Any]:
        slots = list(self.cycles.values())
        routes = {"GREEN_DIRECT": 0, "INSPECT": 0}
        vias = {"direct": 0, "inspected": 0, "fail_safe": 0}
        plates = {"BLUE": 0, "GREEN": 0}
        verify_total = verify_ok = 0
        for slot in slots:
            if slot["triage"] and slot["triage"]["route"] in routes:
                routes[slot["triage"]["route"]] += 1
            if slot["decision"]:
                if slot["decision"]["via"] in vias:
                    vias[slot["decision"]["via"]] += 1
                if slot["decision"]["plate"] in plates:
                    plates[slot["decision"]["plate"]] += 1
            if slot["verify"]:
                verify_total += 1
                verify_ok += 1 if slot["verify"]["ok"] else 0
        current = self.cycles.get(cycle)
        phases = []
        if current:
            phases = [dict(p) for p in current["phases"]]
        return {
            "cycle": cycle,
            "scan": self.last_scan,
            "current": None if current is None else {
                "triage": current["triage"],
                "inspection": current["inspection"],
                "decision": current["decision"],
                "phases": phases,
                "verify": current["verify"],
            },
            "plates": {"blue": plates["BLUE"], "green": plates["GREEN"]},
            "routes": routes,
            "via": vias,
            "verify": {"total": verify_total, "ok": verify_ok,
                       "success_rate": None if verify_total == 0 else round(verify_ok / verify_total, 4)},
            "latency": {
                "step_ms": latency_stats(self.step_ms),
                "cycle_ms": latency_stats(self.cycle_ms),
                "inspection_ms": latency_stats(self.inspection_ms),
            },
            "config": {plate: dict(cfg) for plate, cfg in self.plate_cfg.items()},
        }
