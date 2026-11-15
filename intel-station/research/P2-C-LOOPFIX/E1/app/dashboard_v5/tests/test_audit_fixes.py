"""Regression tests for the audit findings (A1, A2, A3) that the server side can see.

One group per finding; changelog.md maps each finding to its fix and to these tests.

Run:  python3 -I -m pytest -q tests        (from the dashboard code folder)
"""

from __future__ import annotations

import json

import pytest

from dashboard import server as srv
from dashboard.server import RunReducer
from test_dashboard import EVENTS, T0, Sse, append, client, ev, make_run, runs, server, set_current  # noqa: F401 (fixtures)

NAN_LINE = (
    '{"t":"' + T0 + '","mono":1.0,"type":"inspection","cycle":1,"label":"GOOD","score":0.1,'
    '"threshold":0.42,"reasons":[],"geometry":{"studs":NaN,"aspect":1.02,"deep":{"x":Infinity},'
    '"list":[1,NaN]},"device":"GPU","route":"S2"}\n'
)
NAN_GEOMETRY = {"studs": None, "aspect": 1.02, "deep": {"x": None}, "list": [1, None]}


def _refuse_constant(token: str):
    raise AssertionError(f"non-strict JSON token {token}")


def test_reducer_stores_non_finite_geometry_as_null():
    r = RunReducer("nan-unit")
    r.apply({"type": "inspection", "cycle": 1, "label": "GOOD", "score": 0.1, "threshold": 0.42, "reasons": [],
             "geometry": {"studs": float("nan"), "aspect": 1.02, "deep": {"x": float("inf")},
                          "list": [1, float("nan")]}})
    assert r.inspection["geometry"] == NAN_GEOMETRY
    assert r.history[-1]["geometry"] == NAN_GEOMETRY


# A1 F35 (defect 1), A2 row 35 / FIX-1, A3 facts 36-37 / fixes 1-2: NaN or Infinity in geometry


def test_json_safe_maps_non_finite_floats_to_none():
    value = {"a": float("nan"), "b": [1.5, float("-inf")], "c": {"d": True, "e": "x"}}
    assert srv._json_safe(value) == {"a": None, "b": [1.5, None], "c": {"d": True, "e": "x"}}


def test_non_finite_geometry_is_sanitised_and_the_stream_keeps_streaming(runs, server, client):  # noqa: F811
    run = make_run(runs, "run-nan", ev("run_start", mode="dry-run") + ev("state", state="INSPECT", cycle=1) + NAN_LINE)
    set_current(runs, "run-nan")
    sse = Sse(server.server_address[1])
    try:
        snap = sse.until(lambda s: s["inspection"] is not None)
        assert snap["inspection"]["geometry"] == NAN_GEOMETRY
        assert snap["history"][-1]["geometry"] == NAN_GEOMETRY
        append(run / EVENTS, ev("handover", phase="out", duration_s=0.8, model_kept=True))
        snap = sse.until(lambda s: s["handover"] is not None)  # the same connection keeps delivering
        assert snap["handover"]["phase"] == "out"
    finally:
        sse.close()


def test_api_state_is_strict_json_with_non_finite_values(runs, client):  # noqa: F811
    make_run(runs, "run-nan2", ev("run_start", mode="dry-run") + NAN_LINE)
    set_current(runs, "run-nan2")
    status, ctype, payload = client.request("GET", "/api/state")
    assert status == 200 and ctype.startswith("application/json")
    snap = json.loads(payload.decode("utf-8"), parse_constant=_refuse_constant)
    assert snap["inspection"]["geometry"] == NAN_GEOMETRY


@pytest.fixture()
def poisoned_snapshot(monkeypatch):
    """The next snapshots carry a NaN outside geometry, as a future reducer change could produce."""
    original = srv.LiveSession.snapshot

    def poisoned(self):
        snap = original(self)
        snap["counts"] = {**snap["counts"], "mean_cycle_s": float("nan")}
        return snap

    monkeypatch.setattr(srv.LiveSession, "snapshot", poisoned)


def test_api_state_stays_strict_if_a_snapshot_carries_nan(runs, client, poisoned_snapshot):  # noqa: F811
    make_run(runs, "run-nan4", ev("run_start", mode="dry-run"))
    set_current(runs, "run-nan4")
    status, _, payload = client.request("GET", "/api/state")
    assert status == 200
    snap = json.loads(payload.decode("utf-8"), parse_constant=_refuse_constant)
    assert snap["counts"]["mean_cycle_s"] is None


def test_sse_stays_strict_if_a_snapshot_carries_nan(runs, server, client, poisoned_snapshot):  # noqa: F811
    make_run(runs, "run-nan5", ev("run_start", mode="dry-run"))
    set_current(runs, "run-nan5")
    sse = Sse(server.server_address[1])
    try:
        snap = sse.until(lambda s: True)
    finally:
        sse.close()
    assert snap["counts"]["mean_cycle_s"] is None


# A1 F40/F51 (defect 3 part), A2 row 42 / FIX-4, A3 fact 41 / fix 4: image symlink escapes the run dir


def test_run_dir_image_symlink_to_outside_is_refused(tmp_path, runs, client):  # noqa: F811
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"\xff\xd8OUTSIDEBYTES")
    run = make_run(runs, "run-img", ev("run_start", mode="dry-run"))
    (run / "live.jpg").write_bytes(b"\xff\xd8inside")
    (run / "frames" / "escape.jpg").symlink_to(outside)
    (run / "frames" / "alias.jpg").symlink_to(run / "live.jpg")  # a link that stays inside is still served
    set_current(runs, "run-img")
    status, _, body = client.request("GET", "/run/current/frames/escape.jpg")
    assert status == 404 and b"OUTSIDEBYTES" not in body
    status, ctype, body = client.request("GET", "/run/current/frames/alias.jpg")
    assert status == 200 and ctype == "image/jpeg" and body == b"\xff\xd8inside"


# Static serving: a symlink in the static folder that points outside it must not be served


def test_static_symlink_escape_is_refused(tmp_path, monkeypatch, client):  # noqa: F811
    static = tmp_path / "static"
    static.mkdir()
    (static / "ok.js").write_text("console.log('ok');", encoding="utf-8")
    secret = tmp_path / "secret.js"
    secret.write_text("SECRETJS", encoding="utf-8")
    (static / "escape.js").symlink_to(secret)
    monkeypatch.setattr(srv, "STATIC_DIR", static)
    assert client.request("GET", "/static/ok.js")[0] == 200
    status, _, body = client.request("GET", "/static/escape.js")
    assert status == 404 and b"SECRETJS" not in body


# A1 F52 / M5a, A2 row 56 / FIX-5, A3 fact 21: plate from the label when no route is reported


@pytest.mark.parametrize("label,plate,safe", [("GOOD", "BLUE", False), ("DEFECTIVE", "GREEN", False),
                                              ("UNCERTAIN", "GREEN", True)])
def test_plate_follows_the_label_when_no_route_is_reported(label, plate, safe):
    r = RunReducer("fallback")
    r.apply({"type": "inspection", "cycle": 1, "label": label, "score": 0.1, "threshold": 0.42,
             "reasons": [], "t": T0})
    assert r.inspection["route"] is None
    assert r.inspection["plate"] == plate and r.inspection["safe"] is safe


# A1 F48 (defect 2), A2 row 53 / FIX-2, A3 detail check: an earlier cycle's sort, handover or align


def test_handover_align_and_sort_of_an_earlier_cycle_are_not_reported():
    r = RunReducer("stale")
    r.apply({"type": "state", "state": "SORT", "cycle": 3})
    r.apply({"type": "sort_start", "route": "S2"})
    r.apply({"type": "align", "iteration": 1, "dx": 2, "dy": 1, "correction": 0.1})
    r.apply({"type": "sort_end", "ok": True, "duration_s": 3.0})
    r.apply({"type": "handover", "phase": "in", "duration_s": 0.8, "model_kept": True})
    snap = r.snapshot()
    assert snap["sort"]["plate"] == "BLUE" and snap["align"]["iteration"] == 1 and snap["handover"]["phase"] == "in"
    r.apply({"type": "state", "state": "S1", "cycle": 4})
    snap = r.snapshot()
    assert snap["sort"] is None and snap["align"] is None and snap["handover"] is None
    r.apply({"type": "state", "state": "HANDOVER_OUT", "cycle": 4})
    r.apply({"type": "handover", "phase": "out", "duration_s": 0.9, "model_kept": True})
    snap = r.snapshot()
    assert snap["handover"]["phase"] == "out" and snap["handover"]["cycle"] == 4 and snap["sort"] is None
    r.apply({"type": "state", "state": "SORT", "cycle": 4})
    r.apply({"type": "sort_start", "route": "S3"})
    assert r.snapshot()["sort"]["plate"] == "GREEN" and r.stage() == "SORT_GREEN"


# A1 missing item (reconnect after a server restart): a new connection gets the current run again


def test_reconnect_gets_the_current_run_again(runs, server, client):  # noqa: F811
    make_run(runs, "run-rc", ev("run_start", mode="dry-run") + ev("state", state="S1", cycle=2))
    set_current(runs, "run-rc")
    port = server.server_address[1]
    first = Sse(port)
    try:
        assert first.until(lambda s: s["state"] == "S1")["run_id"] == "run-rc"
    finally:
        first.close()
    second = Sse(port)
    try:
        snap = second.until(lambda s: True)
    finally:
        second.close()
    assert snap["run_id"] == "run-rc" and snap["state"] == "S1" and snap["cycle"] == 2
