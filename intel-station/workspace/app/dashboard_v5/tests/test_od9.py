"""OD-9 additions (LOOP-CONTRACT section 7): stages INSPECT / HANDOVER / SORT to BLUE or GREEN,
one card per RealSense view with the fused verdict, handover, align and sort events, the live
preview source and backend from run_start, control ids that also clear orchestrator acks, and
tools/fake_run.py in both sort modes.

Run:  python3 -I -m pytest -q tests        (from the dashboard code folder)
"""

from __future__ import annotations

import json
from pathlib import Path

from dashboard.server import ControlWriter, RunReducer
from test_dashboard import T0, _run_fake, client, ev, runs, server  # noqa: F401 (fixtures)


def _records(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()]


def test_od9_stage_mapping_and_sort_plate():
    r = RunReducer("o1")
    r.apply({"type": "state", "state": "INSPECT", "cycle": 1})
    assert r.stage() == "INSPECT"
    r.apply({"type": "state", "state": "HANDOVER_OUT", "cycle": 1})
    assert r.stage() == "HANDOVER"
    r.apply({"type": "state", "state": "SORT", "cycle": 1})
    assert r.stage() == "SORT"  # no route known yet
    r.apply({"type": "sort_start", "route": "S3", "cycle": 1})
    assert r.stage() == "SORT_GREEN"
    r.apply({"type": "sort_start", "route": "S2", "cycle": 1})
    assert r.stage() == "SORT_BLUE"
    r.apply({"type": "state", "state": "HANDOVER_IN", "cycle": 1})
    assert r.stage() == "HANDOVER"
    r.apply({"type": "state", "state": "CYCLE_DONE", "cycle": 1})
    assert r.stage() == "CYCLE_DONE"
    r.apply({"type": "state", "state": "HOLD", "cycle": 1})
    assert r.stage() == "HOLD"


def test_per_camera_cards_and_fused_verdict():
    r = RunReducer("o2")
    r.apply({"type": "run_start", "mode": "dry-run", "config": {
        "live_source": "wrist camera USB2.0_CAM1 (live.jpg)", "detector": {"backend": "openvino", "device": "GPU"}}})
    assert r.live_source == "wrist camera USB2.0_CAM1 (live.jpg)"
    assert r.backend == "OpenVINO"
    r.apply({"type": "inspect_cameras", "cycle": 2, "cameras": [
        {"camera": "realsense 243622061421", "label": "defective", "score": 0.9, "threshold": 0.42,
         "reasons": ["not_2x2"], "frame": "frames/c002_top_pre.jpg", "crop": "frames/c002_top_crop.jpg"},
        {"camera": "realsense 243722064316", "label": "GOOD", "score": 0.1, "threshold": 0.42,
         "reasons": [], "frame": "frames/c002_close_pre.jpg", "crop": "frames/c002_close_crop.jpg"}]})
    r.apply({"type": "inspection", "cycle": 2, "label": "DEFECTIVE", "score": 0.9, "threshold": 0.42,
             "reasons": ["not_2x2"], "route": "S3", "device": "GPU", "latency_ms": 11, "t": T0})
    insp = r.inspection
    assert [c["view"] for c in insp["cameras"]] == ["close", "top"]  # close view first
    top = insp["cameras"][1]
    assert top["label"] == "DEFECTIVE" and top["score"] == 0.9 and top["threshold"] == 0.42
    assert top["crop_url"] == "/run/current/frames/c002_top_crop.jpg"
    assert insp["plate"] == "GREEN" and insp["safe"] is False and insp["backend"] == "OpenVINO"
    assert insp["frame_url"] == "/run/current/frames/c002_top_pre.jpg"  # no frame on the fused event: top view used
    assert insp["crop_url"] == "/run/current/frames/c002_top_crop.jpg"
    r.apply({"type": "inspection", "cycle": 3, "label": "GOOD", "score": 0.1, "threshold": 0.42,
             "route": "S2", "t": T0})
    assert r.inspection["cameras"] == []  # cameras of another cycle are not attached


def test_uncertain_fused_verdict_is_green_and_safe():
    r = RunReducer("o2b")
    r.apply({"type": "inspection", "cycle": 5, "label": "UNCERTAIN", "score": 0.43, "threshold": 0.42,
             "reasons": [], "route": "S3", "t": T0})
    assert r.inspection["plate"] == "GREEN" and r.inspection["safe"] is True


def test_handover_align_and_sort_fields_reach_the_snapshot():
    r = RunReducer("o3")
    r.apply({"type": "state", "state": "SORT", "cycle": 4})
    r.apply({"type": "sort_start", "route": "S2", "cycle": 4})
    r.apply({"type": "align", "iteration": 2, "dx": 1, "dy": -2, "correction": 0.1})
    r.apply({"type": "sort_end", "ok": True, "duration_s": 3.2})
    r.apply({"type": "handover", "phase": "in", "duration_s": 0.8, "model_kept": True})
    snap = r.snapshot()
    assert snap["sort"]["plate"] == "BLUE" and snap["sort"]["ok"] is True and snap["sort"]["active"] is False
    assert snap["align"]["iteration"] == 2 and snap["align"]["dx"] == 1 and snap["align"]["dy"] == -2
    assert snap["handover"]["phase"] == "in" and snap["handover"]["model_kept"] is True
    assert snap["handover"]["duration_s"] == 0.8


def test_control_id_skips_ids_the_orchestrator_already_acknowledged(tmp_path: Path):
    run = tmp_path / "run-x"
    run.mkdir()
    (run / "events.jsonl").write_text(ev("control", cmd="start", id=7), encoding="utf-8")
    (run / "control.json").write_text(json.dumps({"cmd": "start", "id": 3}), encoding="utf-8")
    payload = ControlWriter().issue(run, "next")
    assert payload == {"cmd": "next", "id": 8}
    assert json.loads((run / "control.json").read_text(encoding="utf-8")) == {"cmd": "next", "id": 8}
    assert sorted(p.name for p in run.iterdir()) == ["control.json", "events.jsonl"]  # no temp file left


def test_fake_run_writes_every_od9_event_in_scripted_sort(tmp_path: Path, runs: Path, server, client):  # noqa: F811
    run = _run_fake(tmp_path, runs, client, ["--cycles", "2", "--auto-next"], "od9-scripted")
    recs = _records(run)
    kinds = [r["type"] for r in recs]
    for kind in ("inspect_cameras", "inspection", "handover", "align", "sort_start", "sort_end"):
        assert kinds.count(kind) >= 2, kind
    states = [r["state"] for r in recs if r["type"] == "state"]
    for name in ("S1", "INSPECT", "HANDOVER_OUT", "SORT", "HANDOVER_IN", "CYCLE_DONE"):
        assert name in states, name
    assert [r["skill"] for r in recs if r["type"] == "skill_start"] == ["S1", "S1"]
    cams = [r for r in recs if r["type"] == "inspect_cameras"][0]["cameras"]
    assert [c["camera"] for c in cams] == ["realsense 243722064316", "realsense 243622061421"]
    for cam in cams:
        assert set(cam) >= {"camera", "label", "score", "threshold", "reasons", "frame", "crop"}
        assert (run / cam["crop"]).read_bytes()[:2] == b"\xff\xd8"
        assert (run / cam["frame"]).read_bytes()[:2] == b"\xff\xd8"
    assert recs[0]["config"]["live_source"] and recs[0]["config"]["detector"]["backend"] == "openvino"
    assert [r["route"] for r in recs if r["type"] == "sort_start"] == [
        r["route"] for r in recs if r["type"] == "inspection"]  # sorter follows the fused route


def test_fake_run_policy_sort_uses_the_team_skills(tmp_path: Path, runs: Path, server, client):  # noqa: F811
    run = _run_fake(tmp_path, runs, client, ["--cycles", "2", "--auto-next", "--sort-mode", "policy"],
                    "od9-policy")
    recs = _records(run)
    assert [r["skill"] for r in recs if r["type"] == "skill_start"] == ["S1", "S2", "S1", "S3"]
    assert not any(r["type"] in ("sort_start", "sort_end", "align") for r in recs)
