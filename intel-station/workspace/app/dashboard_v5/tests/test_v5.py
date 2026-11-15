"""v5 tests: the policy reducer (scan, triage, decision, policy_run, verify, inspection latency), the intel
sources (telemetry tail and summary, bench.json rows and cache), JSON safety of the new fields, and the replay
tool (a schema-exact run folded through the reducer and the live session).

Run:  python3 -I -m pytest -q tests        (from the dashboard code folder)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

from dashboard import policy
from dashboard.intel import IntelSources, read_bench, read_tail, summarise_telemetry
from dashboard.policy import PolicyState, latency_stats, parse_t, percentile
from dashboard.server import LiveSession, RunLocator, RunReducer, _json_safe

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import replay_v5  # noqa: E402  (tools/: the synthetic run writer)

T = "2026-11-14T12:00:00.000Z"

REQUIRED_FIELDS = {
    "run_start": {"run_id", "mode", "config"},
    "state": {"state"},
    "scan": {"bricks", "target_id", "target_how", "hands", "frame_top", "frame_close"},
    "triage": {"target_id", "route", "reasons"},
    "inspection": {"target_id", "mode", "result", "label", "reasons", "score", "threshold", "cameras", "geometry",
                   "device", "latency_ms", "frame", "crop", "route"},
    "decision": {"target_id", "plate", "rule", "via"},
    "skill_start": {"skill", "model", "plate", "task_text"},
    "policy_run": {"phase", "model_id", "task_text", "plate", "target_id"},
    "skill_end": {"skill", "ok", "duration_s", "reason"},
    "verify": {"target_id", "plate_intended", "release_side", "ok", "frame_top"},
    "stats": {"cycles", "good", "defective", "uncertain", "mean_cycle_s"},
}


# ----- helpers -----------------------------------------------------------------------------------


def event(kind: str, **fields) -> dict:
    record = {"t": T, "mono": 1.0, "type": kind}
    record.update(fields)
    return record


def plates_config() -> dict:
    return {"plates": {
        "BLUE": {"skill": "S2", "model_id": "replay-act-blue-openvino", "task": "move to blue",
                 "backend": "openvino", "device": "GPU"},
        "GREEN": {"skill": "S3", "model_id": "replay-act-green-openvino", "task": "move to green",
                  "backend": "openvino", "device": "GPU"},
    }}


def url_for(rel):
    return None if not isinstance(rel, str) else "/run/current/" + rel


def policy_state() -> PolicyState:
    state = PolicyState(url_for)
    state.read_config(plates_config())
    return state


def brick(brick_id: str, verdict: str, **extra) -> dict:
    confident = verdict == "CONFIDENT_DEFECT"
    raw = {"id": brick_id, "colour": "white", "centre_px": [190.0, 300.0], "angle_deg": 1.5,
           "shape": "2x2", "shape_conf": 0.9, "tape_score": 0.1, "tape_conf": 0.8,
           "verdict": verdict, "confident": confident, "reasons": ["not_2x2"] if confident else [],
           "crop_top": "frames/c1_scan_top.jpg", "crop_close": "frames/c1_scan_close.jpg"}
    raw.update(extra)
    return raw


def sample(t: str, cpu, pkg=66.0, **extra) -> dict:
    row = {"t": t, "pkg_w": None, "temp_c": {"pkg": pkg, "core_max": 61.0}, "cpu_util": cpu,
           "gpu": {"freq_mhz": 900.0, "max_mhz": 2500, "freq_ratio": 0.36, "media_mhz": 0},
           "npu": {"busy": 0.1, "freq_mhz": 0}, "mem": {"used_pct": 25.0, "avail_mb": 47000.0},
           "src": "live sampler (test)"}
    row.update(extra)
    return row


def bench_doc(rows: list | None = None) -> dict:
    if rows is None:
        rows = [
            {"model": "act_389da0fd_fp16", "device": "GPU", "status": "ok",
             "latency_ms": {"p50": 12.5, "p95": 14.0}, "throughput_ips": 80.0,
             "telemetry_during_timed": {"pkg_temp_c_mean": 71.2, "cpu_util_mean": 0.31, "npu_busy_mean": 0.0},
             "energy_per_inference_j": None},
            {"model": "padim_b1_detector", "device": "NPU", "status": "error", "error": "compile failed",
             "latency_ms": {"p50": math.nan}},
        ]
    return {"status": "complete", "openvino": "test runtime", "warmup": 20, "timed": 200, "batch": 1,
            "power": {"pkg_w": "not measured: RAPL energy_uj is root-only on this laptop",
                      "energy_per_inference_j": None},
            "runs": rows}


# ----- policy: numbers ---------------------------------------------------------------------------


def test_percentile_interpolates_and_skips_non_finite_values():
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([1.0, math.nan, 3.0, math.inf], 50) == 2.0
    assert percentile([], 50) is None
    assert percentile([7.0], 95) == 7.0


def test_latency_stats_rounds_and_is_empty_without_samples():
    assert latency_stats([]) == {"n": 0, "p50": None, "p95": None, "max": None}
    assert latency_stats([10.0, 20.0, 30.0, math.nan]) == {"n": 3, "p50": 20.0, "p95": 29.0, "max": 30.0}


def test_parse_t_reads_utc_stamps_and_rejects_other_text():
    start = parse_t(T)
    assert parse_t("2026-11-14T12:00:01.500Z") - start == pytest.approx(1.5)
    assert parse_t("not a time") is None
    assert parse_t(None) is None


# ----- policy: the reducer -----------------------------------------------------------------------


def test_scan_routes_a_confident_defect_straight_to_green_and_keeps_the_rest_inspected():
    state = policy_state()
    state.apply("scan", {"cycle": 1, "bricks": [brick("b1", "GOOD"), brick("b3", "CONFIDENT_DEFECT")],
                         "target_id": "b1", "target_how": "nearest",
                         "frame_top": "frames/c1_scan_top.jpg", "frame_close": "frames/c1_scan_close.jpg"}, T, 1)
    scan = state.last_scan
    assert scan["target_id"] == "b1" and scan["target_how"] == "nearest"
    assert scan["frame_top_url"] == "/run/current/frames/c1_scan_top.jpg"
    assert [b["route"] for b in scan["bricks"]] == ["inspected", "straight to GREEN (I-52)"]
    assert [b["direct_green"] for b in scan["bricks"]] == [False, True]
    assert scan["bricks"][0]["crop_url"] == "/run/current/frames/c1_scan_close.jpg"


def test_step_latency_is_the_gap_between_policy_run_events_of_one_cycle():
    state = policy_state()
    state.apply("decision", {"cycle": 2, "target_id": "b1", "plate": "BLUE", "rule": "OD-10", "via": "inspected"},
                T, 2)
    for phase, when in [("start", "2026-11-14T12:00:00.000Z"), ("grasp", "2026-11-14T12:00:01.100Z"),
                        ("carry", "2026-11-14T12:00:02.500Z")]:
        state.apply("policy_run", {"cycle": 2, "phase": phase, "plate": "BLUE",
                                   "model_id": "replay-act-blue-openvino", "task_text": "move to blue",
                                   "target_id": "b1"}, when, 2)
    snap = state.snapshot(2)
    phases = snap["current"]["phases"]
    assert [p["step_ms"] for p in phases] == [None, 1100.0, 1400.0]
    assert [p["device"] for p in phases] == ["GPU", "GPU", "GPU"]  # the device comes from the plate config
    assert snap["latency"]["step_ms"]["n"] == 2
    assert snap["latency"]["step_ms"]["p50"] == 1250.0
    assert snap["latency"]["step_ms"]["p95"] == pytest.approx(1385.0)


def test_cycle_latency_runs_from_decision_to_verify():
    state = policy_state()
    state.apply("decision", {"cycle": 1, "target_id": "b1", "plate": "GREEN", "rule": "OD-10", "via": "direct"},
                "2026-11-14T12:00:00.000Z", 1)
    state.apply("verify", {"cycle": 1, "target_id": "b1", "plate_intended": "GREEN", "release_side": "GREEN",
                           "ok": True, "frame_top": "frames/c1_verify_top.jpg"}, "2026-11-14T12:00:05.200Z", 1)
    snap = state.snapshot(1)
    assert snap["latency"]["cycle_ms"] == {"n": 1, "p50": 5200.0, "p95": 5200.0, "max": 5200.0}
    assert snap["verify"] == {"total": 1, "ok": 1, "success_rate": 1.0}
    assert snap["current"]["verify"]["frame_top_url"] == "/run/current/frames/c1_verify_top.jpg"
    assert snap["current"]["verify"]["release_side"] == "GREEN"


def test_route_plate_and_verify_counts_cover_every_cycle():
    state = policy_state()
    cases = [("GREEN_DIRECT", "direct", "GREEN", True), ("INSPECT", "inspected", "BLUE", True),
             ("INSPECT", "fail_safe", "GREEN", False)]
    for cycle, (route, via, plate, ok) in enumerate(cases, start=1):
        state.apply("triage", {"cycle": cycle, "target_id": "b1", "route": route, "reasons": ["why"]}, T, cycle)
        state.apply("decision", {"cycle": cycle, "target_id": "b1", "plate": plate, "rule": "OD-10", "via": via},
                    T, cycle)
        state.apply("verify", {"cycle": cycle, "target_id": "b1", "plate_intended": plate,
                               "release_side": plate, "ok": ok}, T, cycle)
    snap = state.snapshot(3)
    assert snap["routes"] == {"GREEN_DIRECT": 1, "INSPECT": 2}
    assert snap["via"] == {"direct": 1, "inspected": 1, "fail_safe": 1}
    assert snap["plates"] == {"blue": 1, "green": 2}
    assert snap["verify"] == {"total": 3, "ok": 2, "success_rate": 0.6667}
    assert snap["current"]["triage"]["route_text"] == "inspected"


def test_inspection_latency_keeps_finite_numbers_only():
    state = policy_state()
    state.note_inspection({"cycle": 1, "mode": "still_hires", "result": "GOOD", "latency_ms": 11.0,
                           "device": "GPU"}, T, 1)
    state.note_inspection({"cycle": 2, "mode": "still_hires", "result": "DEFECTIVE", "latency_ms": math.nan,
                           "reasons": ["tape/appearance"]}, T, 2)
    state.note_inspection({"cycle": 3, "mode": "still_hires", "latency_ms": "12"}, T, 3)
    assert state.inspection_ms == [11.0]
    snap = state.snapshot(2)
    assert snap["latency"]["inspection_ms"] == {"n": 1, "p50": 11.0, "p95": 11.0, "max": 11.0}
    assert snap["current"]["inspection"]["result"] == "DEFECTIVE"
    assert snap["current"]["inspection"]["reasons"] == ["tape/appearance"]


def test_plate_config_is_read_from_run_start_and_absent_config_is_quiet():
    state = PolicyState(url_for)
    state.read_config({"mode": "policy"})  # no plates: nothing to show, no error
    assert state.snapshot(1)["config"] == {}
    state.read_config(plates_config())
    assert state.snapshot(1)["config"]["GREEN"]["model_id"] == "replay-act-green-openvino"


def test_cycle_records_are_bounded_to_the_newest_max_cycles():
    state = PolicyState(url_for)
    for cycle in range(1, policy.MAX_CYCLES + 11):
        state.apply("decision", {"plate": "BLUE", "via": "inspected", "rule": "OD-10"}, T, cycle)
    assert len(state.cycles) == policy.MAX_CYCLES
    assert min(state.cycles) == 11


def test_non_finite_numbers_in_policy_events_come_out_as_null():
    state = policy_state()
    state.apply("scan", {"bricks": [brick("b1", "GOOD", angle_deg=math.nan, shape_conf=math.inf,
                                          centre_px=[math.nan, 2.0])]}, T, 1)
    state.note_inspection({"cycle": 1, "mode": "still_hires", "result": "GOOD", "latency_ms": math.inf}, T, 1)
    body = json.loads(json.dumps(_json_safe(state.snapshot(1)), allow_nan=False))
    scan_brick = body["scan"]["bricks"][0]
    assert scan_brick["angle_deg"] is None and scan_brick["shape_conf"] is None and scan_brick["centre_px"] is None
    assert body["latency"]["inspection_ms"]["n"] == 0


# ----- reducer: one cycle, errors, JSON --------------------------------------------------------


def test_reducer_folds_one_policy_cycle_into_the_snapshot():
    red = RunReducer("run-policy")
    red.apply(event("run_start", mode="policy", config=plates_config()))
    red.apply(event("state", state="HOME", cycle=1))
    red.apply(event("state", state="SCAN", cycle=1))
    red.apply(event("scan", cycle=1, bricks=[brick("b1", "GOOD"), brick("b3", "CONFIDENT_DEFECT")],
                    target_id="b1", target_how="nearest", frame_top="frames/t.jpg", frame_close="frames/c.jpg"))
    assert red.stage() == "SCAN"
    red.apply(event("state", state="TRIAGE", cycle=1))
    red.apply(event("triage", cycle=1, target_id="b1", route="INSPECT", reasons=["not confident: inspect"]))
    cam_close = {"label": "GOOD", "score": 0.3, "threshold": 0.42, "frame": "frames/close.jpg",
                 "crop": "frames/close.jpg"}
    cam_top = {"label": "GOOD", "score": 0.5, "threshold": 0.42, "frame": "frames/top.jpg", "crop": "frames/top.jpg"}
    red.apply(event("inspection", cycle=1, target_id="b1", mode="still_hires", result="GOOD", label="GOOD",
                    reasons=[], score=0.61, threshold=0.42, device="GPU", latency_ms=9.5, route="S2",
                    frame="frames/s.jpg", cameras={"realsense 243722064316": cam_close,
                                                   "realsense 243622061421": cam_top}))
    red.apply(event("state", state="ACT", cycle=1))
    red.apply(event("decision", cycle=1, target_id="b1", plate="BLUE", rule="OD-10", via="inspected"))
    for phase in ("start", "grasp"):
        red.apply(event("policy_run", cycle=1, phase=phase, plate="BLUE", model_id="replay-act-blue-openvino",
                        task_text="move to blue", target_id="b1"))
    red.apply(event("state", state="VERIFY", cycle=1))
    red.apply(event("verify", cycle=1, target_id="b1", plate_intended="BLUE", release_side="BLUE", ok=True,
                    frame_top="frames/v.jpg"))
    red.apply(event("stats", cycles=1, good=1, defective=0, uncertain=0, mean_cycle_s=12.0))
    snap = red.snapshot()
    assert [c["view"] for c in snap["inspection"]["cameras"]] == ["close", "top"]
    assert snap["policy"]["current"]["decision"]["plate"] == "BLUE"
    assert snap["policy"]["current"]["decision"]["path"] == "inspected"
    assert [p["phase"] for p in snap["policy"]["current"]["phases"]] == ["start", "grasp"]
    assert snap["policy"]["current"]["verify"]["ok"] is True
    assert snap["policy"]["verify"] == {"total": 1, "ok": 1, "success_rate": 1.0}
    assert snap["policy"]["scan"]["target_id"] == "b1"
    assert snap["policy"]["latency"]["inspection_ms"]["p50"] == 9.5
    assert snap["counts"] == {"source": "stats", "cycles": 1, "good": 1, "defective": 0, "uncertain": 0,
                              "mean_cycle_s": 12.0}
    assert snap["backend"] == "OpenVINO"  # read from the plates of run_start
    assert snap["errors"] == 0


def test_error_events_are_counted_and_the_last_message_is_shown():
    red = RunReducer("run-err")
    red.apply(event("error", where="SORT", message="skill timeout after 60 s"))
    red.apply(event("error", where="ACT", message="gripper stalled"))
    snap = red.snapshot()
    assert snap["errors"] == 2
    assert snap["error"]["message"] == "gripper stalled"


def test_reducer_snapshot_is_json_safe_for_nan_and_infinity():
    red = RunReducer("run-nan")
    red.apply(event("run_start", mode="policy", config=plates_config()))
    red.apply(event("inspection", cycle=1, label="GOOD", score=math.nan, threshold=math.inf, latency_ms=math.nan,
                    reasons=[], geometry={"w": math.nan}, route="S2"))
    red.apply(event("stats", cycles=1, good=1, defective=0, uncertain=0, mean_cycle_s=math.nan))
    body = json.loads(json.dumps(_json_safe(red.snapshot()), allow_nan=False))
    assert body["inspection"]["score"] is None and body["inspection"]["latency_ms"] is None
    assert body["counts"]["mean_cycle_s"] is None


# ----- intel: telemetry --------------------------------------------------------------------------


def test_read_tail_keeps_only_whole_json_objects(tmp_path):
    path = tmp_path / "day.jsonl"
    path.write_text('{"a":1}\n{"b":2}\n{"c":3}\n', encoding="utf-8")
    assert read_tail(path, max_bytes=10) == [{"c": 3}]  # the first, cut line is dropped
    assert read_tail(path) == [{"a": 1}, {"b": 2}, {"c": 3}]
    path.write_text('{"a":1}\n[1,2]\n{"b":', encoding="utf-8")  # a non-object line and a torn last line
    assert read_tail(path) == [{"a": 1}]
    assert read_tail(tmp_path / "missing.jsonl") == []


def test_summarise_telemetry_means_the_last_minute_and_marks_a_stale_sample():
    now = parse_t("2026-11-14T12:01:00.000Z")
    rows = [sample("2026-11-14T12:00:30.000Z", 0.2), sample("2026-11-14T12:00:50.000Z", 0.4),
            sample("2026-11-14T12:00:59.000Z", 0.6)]
    fresh = summarise_telemetry(rows, now, "logs/telemetry/20261114.jsonl")
    assert fresh["available"] is True and fresh["stale"] is False and fresh["age_s"] == 1.0
    assert fresh["samples_in_window"] == 3
    assert fresh["cpu_util"] == {"latest": 0.6, "avg_60s": 0.4}
    assert fresh["src"] == "live sampler (test)"
    assert fresh["power"] == {"pkg_w": None, "available": False, "avg_60s_w": None}  # RAPL not readable: no guess
    assert fresh["gpu"]["freq_ratio"] == 0.36 and fresh["npu"]["busy"] == 0.1
    stale = summarise_telemetry(rows, now + 15.0, "f")
    assert stale["stale"] is True and stale["samples_in_window"] == 3


def test_summarise_telemetry_keeps_missing_and_non_finite_values_as_none():
    now = parse_t("2026-11-14T12:01:00.000Z")
    row = sample("2026-11-14T12:00:59.000Z", math.nan, pkg=math.inf, gpu={"freq_mhz": math.nan})
    out = summarise_telemetry([row], now, "f")
    assert out["cpu_util"]["latest"] is None
    assert out["temp_c"]["pkg"] is None and out["temp_c"]["max_all"] == 61.0
    assert out["gpu"]["freq_mhz"] is None and out["gpu"]["busy"] is None
    assert out["mem"] == {"used_pct": 25.0, "avail_mb": 47000.0}
    json.dumps(_json_safe(out), allow_nan=False)


def test_summarise_telemetry_without_samples_says_so():
    out = summarise_telemetry([], parse_t(T), "f")
    assert out["available"] is False and out["reason"] == "no telemetry samples yet"


# ----- intel: bench.json -------------------------------------------------------------------------


def test_read_bench_lists_each_row_with_its_status_and_source(tmp_path):
    path = tmp_path / "bench.json"
    path.write_text(json.dumps(bench_doc()), encoding="utf-8")
    out = read_bench(path)
    assert out["available"] is True and out["file"] == str(path)
    assert out["status"] == "complete" and out["openvino"] == "test runtime"
    assert out["energy_note"] == "not measured"
    assert out["power_note"].startswith("not measured: RAPL")
    ok, failed = out["rows"]
    assert (ok["model"], ok["device"], ok["status"]) == ("act_389da0fd_fp16", "GPU", "ok")
    assert ok["p50_ms"] == 12.5 and ok["p95_ms"] == 14.0 and ok["throughput_ips"] == 80.0
    assert ok["temp_c_mean"] == 71.2 and ok["npu_busy_mean"] == 0.0 and ok["error"] is None
    assert failed["status"] == "error" and failed["error"] == "compile failed"
    assert failed["p50_ms"] is None


def test_read_bench_reads_again_when_the_file_changes(tmp_path):
    path = tmp_path / "bench.json"
    path.write_text(json.dumps(bench_doc()), encoding="utf-8")
    assert len(read_bench(path)["rows"]) == 2
    path.write_text(json.dumps(bench_doc(bench_doc()["runs"][:1])), encoding="utf-8")
    assert len(read_bench(path)["rows"]) == 1


def test_read_bench_reports_a_missing_or_unreadable_file(tmp_path):
    missing = tmp_path / "none.json"
    assert read_bench(missing) == {"available": False, "file": str(missing), "reason": "bench.json not found"}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    out = read_bench(bad)
    assert out["available"] is False and out["reason"] == "bench.json unreadable"


def test_intel_sources_read_the_utc_day_file_and_the_bench(tmp_path):
    telemetry = tmp_path / "telemetry"
    telemetry.mkdir()
    now = parse_t("2026-11-14T23:59:30.000Z")
    (telemetry / "20261114.jsonl").write_text(json.dumps(sample("2026-11-14T23:59:29.000Z", 0.25)) + "\n",
                                              encoding="utf-8")
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps(bench_doc()), encoding="utf-8")
    snap = IntelSources(telemetry, bench).snapshot(now)
    assert snap["telemetry"]["file"] == str(telemetry / "20261114.jsonl")
    assert snap["telemetry"]["cpu_util"]["latest"] == 0.25
    assert snap["bench"]["available"] is True and len(snap["bench"]["rows"]) == 2
    json.dumps(_json_safe(snap), allow_nan=False)
    assert IntelSources(None, None).snapshot(now) == {"telemetry": None, "bench": None}


# ----- replay: a schema-exact run through the reducer and the live session ------------------------


@pytest.fixture(scope="module")
def replay(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("replay")
    replay_v5.build(out, cycles=5, run_id="replay-test", pace=0.0, seed=3)
    return out


def load_events(run_dir: Path) -> list[dict]:
    lines = (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_replay_events_carry_the_emitter_fields_and_stay_strict_json(replay):
    events = load_events(replay / "logs" / "loop" / "replay-test")
    assert events[0]["type"] == "run_start" and events[0]["mode"] == "policy"
    assert set(events[0]["config"]["plates"]) == {"BLUE", "GREEN"}
    assert events[0]["config"]["synthetic"] is True
    for ev in events:
        assert ev["t"].endswith("Z") and "cycle" in ev and isinstance(ev["mono"], float)
        json.dumps(ev, allow_nan=False)
        missing = REQUIRED_FIELDS.get(ev["type"], set()) - set(ev)
        assert not missing, (ev["type"], missing)
    counts: dict[str, int] = {}
    for ev in events:
        counts[ev["type"]] = counts.get(ev["type"], 0) + 1
    assert counts["policy_run"] == 30  # 5 cycles x (start, grasp, carry, release, return, done)
    assert counts["decision"] == counts["verify"] == counts["skill_end"] == 5
    assert counts["inspection"] == 4  # the four INSPECT cycles; the GREEN_DIRECT cycle is not inspected
    assert "run_end" not in counts  # the run stays open unless --ended


def test_replay_run_folds_to_five_verified_cycles(replay):
    run_dir = replay / "logs" / "loop" / "replay-test"
    red = RunReducer(run_dir.name)
    for ev in load_events(run_dir):
        red.apply(ev)
    snap = red.snapshot()
    assert snap["stage"] == "IDLE" and snap["ended"] is None
    assert snap["policy"]["verify"] == {"total": 5, "ok": 5, "success_rate": 1.0}
    assert snap["policy"]["routes"] == {"GREEN_DIRECT": 1, "INSPECT": 4}
    assert snap["policy"]["via"] == {"direct": 1, "inspected": 3, "fail_safe": 1}
    assert snap["policy"]["plates"] == {"blue": 2, "green": 3}
    assert snap["policy"]["latency"]["step_ms"]["n"] == 25
    assert abs(snap["policy"]["latency"]["step_ms"]["p50"] - 1200.0) <= 2.0  # synthetic step times, +-1 ms stamps
    assert snap["counts"]["source"] == "stats" and snap["counts"]["cycles"] == 5
    assert snap["backend"] == "OpenVINO" and snap["errors"] == 0
    assert snap["policy"]["config"]["GREEN"]["model_id"] == "replay-act-green-openvino"
    assert [b["route"] for b in snap["policy"]["scan"]["bricks"]] == [
        "inspected", "inspected", "straight to GREEN (I-52)"]
    json.dumps(_json_safe(snap), allow_nan=False)


def test_replay_telemetry_and_bench_are_labelled_synthetic(replay):
    files = sorted((replay / "logs" / "telemetry").glob("*.jsonl"))
    assert len(files) == 1
    samples = read_tail(files[0])
    assert samples and all(s["src"].startswith("replay synthetic") for s in samples)
    assert all(s["pkg_w"] is None for s in samples)
    summary = summarise_telemetry(samples, parse_t(samples[-1]["t"]) + 1.0, str(files[0]))
    assert summary["available"] is True and summary["stale"] is False
    assert summary["power"] == {"pkg_w": None, "available": False, "avg_60s_w": None}
    bench = read_bench(replay / "bench.json")
    assert bench["available"] is True and bench["status"] == "synthetic-replay"
    assert len(bench["rows"]) == 12 and {row["status"] for row in bench["rows"]} == {"ok"}
    assert bench["energy_note"] == "not measured"


def test_live_session_serves_the_replay_with_intel(replay):
    runs = replay / "logs" / "loop"
    session = LiveSession(RunLocator(runs), IntelSources(replay / "logs" / "telemetry", replay / "bench.json"))
    assert RunLocator(runs).current_dir().name == "replay-test"
    assert session.step() is True
    snap = session.snapshot()
    assert snap["run_id"] == "replay-test" and snap["stage"] == "IDLE"
    assert snap["policy"]["verify"]["ok"] == 5
    assert len(snap["intel"]["bench"]["rows"]) == 12
    assert snap["intel"]["telemetry"]["available"] is True
    json.dumps(_json_safe(snap), allow_nan=False)


def test_replay_with_the_ended_flag_writes_run_end(tmp_path):
    replay_v5.build(tmp_path, cycles=1, run_id="ended-run", pace=0.0, seed=1, ended=True)
    events = load_events(tmp_path / "logs" / "loop" / "ended-run")
    assert events[-1]["type"] == "run_end" and events[-1]["reason"] == "max_cycles"
    red = RunReducer("ended-run")
    for ev in events:
        red.apply(ev)
    snap = red.snapshot()
    assert snap["stage"] == "RUN_ENDED" and snap["ended"]["reason"] == "max_cycles"
