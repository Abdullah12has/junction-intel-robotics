"""Dashboard tests: SSE tailing (partial lines, bad lines, run switch), control.json
(atomic writes, increasing ids), path safety, run-dir-only file serving, and an
end-to-end run of tools/fake_run.py through the server.

Run:  python3 -I -m pytest -q tests        (from the dashboard code folder)
"""

from __future__ import annotations

import http.client
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import PIL  # noqa: F401 (fake_run needs Pillow: a missing Pillow fails these tests, it does not skip them)
import pytest

from dashboard import server as srv
from dashboard.server import ControlWriter, DashboardServer, EventTailer, RunReducer

ROOT = Path(__file__).resolve().parents[1]
FAKE_RUN = ROOT / "tools" / "fake_run.py"
EVENTS = "events.jsonl"
T0 = "2026-11-14T12:00:00.000Z"


# ----- helpers ---------------------------------------------------------------


def ev(kind: str, **fields) -> str:
    record = {"t": T0, "mono": 1.0, "type": kind}
    record.update(fields)
    return json.dumps(record, separators=(",", ":")) + "\n"


def make_run(runs: Path, name: str, text: str = "") -> Path:
    run = runs / name
    (run / "frames").mkdir(parents=True, exist_ok=True)
    (run / EVENTS).write_text(text, encoding="utf-8")
    return run


def append(path: Path, text: str) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(text)


def set_current(runs: Path, name: str) -> None:
    tmp = runs / "current.tmp-test"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(name, tmp)
    os.replace(tmp, runs / "current")


class Client:
    def __init__(self, port: int) -> None:
        self.port = port

    def request(self, method, path, body=None, headers=None, raw_body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        hdrs = {"Host": f"127.0.0.1:{self.port}"}
        hdrs.update(headers or {})
        if raw_body is not None:
            data = raw_body
        elif body is not None:
            data = json.dumps(body).encode("utf-8")
            hdrs.setdefault("Content-Type", "application/json")
        else:
            data = None
        conn.request(method, path, body=data, headers=hdrs)
        resp = conn.getresponse()
        payload = resp.read()
        ctype = resp.getheader("Content-Type")
        status = resp.status
        conn.close()
        return status, ctype, payload

    def post_control(self, cmd: str, origin: str | None = None):
        headers = {"Origin": origin} if origin else None
        status, _, payload = self.request("POST", "/api/control", body={"cmd": cmd}, headers=headers)
        return status, json.loads(payload)


class Sse:
    """Reads snapshot events from /api/events with a socket timeout, so a broken stream fails fast."""

    def __init__(self, port: int) -> None:
        self.conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        self.conn.request("GET", "/api/events", headers={"Host": f"127.0.0.1:{port}"})
        self.resp = self.conn.getresponse()
        assert self.resp.status == 200
        assert self.resp.getheader("Content-Type").startswith("text/event-stream")

    def next_snapshot(self) -> dict:
        event = None
        while True:
            raw = self.resp.fp.readline()
            if not raw:
                raise EOFError("event stream closed")
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and event == "snapshot":
                return json.loads(line.split(":", 1)[1])
            elif line == "":
                event = None

    def until(self, predicate, timeout: float = 6.0) -> dict:
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            last = self.next_snapshot()
            if predicate(last):
                return last
        raise AssertionError(f"no matching snapshot within {timeout}s; last={last}")

    def close(self) -> None:
        self.conn.close()


@pytest.fixture()
def runs(tmp_path: Path) -> Path:
    root = tmp_path / "loop"
    root.mkdir()
    return root


@pytest.fixture()
def server(runs: Path):
    srv_obj = DashboardServer(runs, "127.0.0.1", 0)
    thread = threading.Thread(target=srv_obj.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    yield srv_obj
    srv_obj.shutdown()
    srv_obj.server_close()


@pytest.fixture()
def client(server) -> Client:
    return Client(server.server_address[1])


# ----- EventTailer (unit) ----------------------------------------------------


def test_tailer_holds_partial_line_until_newline(tmp_path: Path):
    path = tmp_path / EVENTS
    path.write_text(ev("state", state="IDLE", cycle=0) + '{"type":"state","sta', encoding="utf-8")
    tailer = EventTailer(path)
    reset, events = tailer.poll()
    assert reset is False
    assert [e["state"] for e in events] == ["IDLE"]
    append(path, 'te":"S1","cycle":1}')  # complete JSON but no newline yet
    assert tailer.poll() == (False, [])
    append(path, "\n")
    _, events = tailer.poll()
    assert [e["state"] for e in events] == ["S1"]
    assert tailer.bad_lines == 0


def test_tailer_skips_and_counts_bad_lines(tmp_path: Path):
    path = tmp_path / EVENTS
    path.write_text("not json\n[1,2]\n\n" + ev("heartbeat"), encoding="utf-8")
    tailer = EventTailer(path)
    _, events = tailer.poll()
    assert [e["type"] for e in events] == ["heartbeat"]
    assert tailer.bad_lines == 2  # the blank line is ignored, not counted


def test_tailer_resets_when_file_is_replaced_or_truncated(tmp_path: Path):
    path = tmp_path / EVENTS
    path.write_text(ev("state", state="A") + ev("state", state="B"), encoding="utf-8")
    tailer = EventTailer(path)
    assert [e["state"] for e in tailer.poll()[1]] == ["A", "B"]

    new_file = tmp_path / "new.jsonl"
    new_file.write_text(ev("state", state="C"), encoding="utf-8")
    os.replace(new_file, path)  # new inode
    reset, events = tailer.poll()
    assert reset is True
    assert [e["state"] for e in events] == ["C"]

    path.write_text(ev("state", state="D"), encoding="utf-8")  # same inode, shorter content
    reset, events = tailer.poll()
    assert reset is True
    assert [e["state"] for e in events] == ["D"]


def test_tailer_missing_file_is_quiet(tmp_path: Path):
    assert EventTailer(tmp_path / "absent.jsonl").poll() == (False, [])


# ----- RunReducer (unit) -----------------------------------------------------


def test_reducer_stage_mapping_and_history_window():
    r = RunReducer("r1")
    assert r.stage() == "STARTING"
    r.apply({"type": "state", "state": "S1", "cycle": 1, "t": T0})
    assert r.stage() == "S1_PICK"
    for cycle in range(1, 11):
        r.apply({"type": "inspection", "cycle": cycle, "label": "GOOD", "score": 0.1,
                 "threshold": 0.42, "reasons": [], "device": "GPU", "latency_ms": 10,
                 "frame": f"frames/c{cycle:03d}_pre.jpg", "crop": f"frames/c{cycle:03d}_crop.jpg",
                 "route": "S2", "t": T0})
    assert r.stage() == "S1_PICK"  # inspection of cycle 10 is not the current cycle yet
    r.apply({"type": "state", "state": "S1", "cycle": 10})
    assert r.stage() == "INSPECT"
    assert len(r.history) == 8
    assert r.history[-1]["cycle"] == 10
    assert r.history[-1]["frame_url"] == "/run/current/frames/c010_pre.jpg"
    assert r.history[-1]["plate"] == "BLUE"
    assert r.counts()["source"] == "derived" and r.counts()["cycles"] == 10
    r.apply({"type": "state", "state": "DECIDE", "cycle": 10})
    assert r.stage() == "INSPECT"
    r.apply({"type": "state", "state": "HOLD", "cycle": 10})
    assert r.stage() == "HOLD"
    r.apply({"type": "state", "state": "STOPPED", "cycle": 10})
    assert r.stage() == "STOPPED"
    r.apply({"type": "run_end", "reason": "quit", "t": T0})
    assert r.stage() == "RUN_ENDED"


def test_reducer_verdict_plate_follows_route_and_safe_flag():
    r = RunReducer("r2")
    r.apply({"type": "inspection", "cycle": 1, "label": "UNCERTAIN", "score": 0.44,
             "threshold": 0.42, "reasons": ["tape/appearance"], "route": "S3", "t": T0,
             "frame": "frames/c001_pre.jpg", "crop": "../escape.jpg"})
    insp = r.inspection
    assert insp["plate"] == "GREEN" and insp["safe"] is True
    assert insp["crop_url"] is None  # a path that could escape the run dir is never turned into a URL
    assert insp["frame_url"] == "/run/current/frames/c001_pre.jpg"


def test_reducer_non_finite_and_wrong_types_are_dropped():
    r = RunReducer("r3")
    r.apply({"type": "inspection", "cycle": True, "label": "GOOD", "score": float("nan"),
             "threshold": "0.4", "reasons": "not-a-list", "geometry": [1], "latency_ms": None})
    insp = r.inspection
    assert insp["cycle"] is None and insp["score"] is None and insp["threshold"] is None
    assert insp["reasons"] == [] and insp["geometry"] == {} and insp["latency_ms"] is None


# ----- SSE over HTTP ---------------------------------------------------------


def test_sse_partial_lines_and_bad_lines_over_http(runs: Path, server, client: Client):
    run = make_run(runs, "run-a", ev("run_start", mode="dry-run") + ev("state", state="WAIT_BRICK", cycle=1)
                   + '{"t":"2026-11-14T12:00:02.000Z","type":"state","state":"S1"')
    set_current(runs, "run-a")
    sse = Sse(server.server_address[1])
    try:
        first = sse.until(lambda s: True)
        assert first["run_id"] == "run-a" and first["mode"] == "dry-run"
        assert first["state"] == "WAIT_BRICK" and first["events"] == 2  # partial last line is not delivered

        append(run / EVENTS, ',"cycle":1}\n')
        snap = sse.until(lambda s: s["state"] == "S1")
        assert snap["stage"] == "S1_PICK" and snap["cycle"] == 1 and snap["events"] == 3

        # a line split over three writes must come out as one event
        append(run / EVENTS, '{"t":"2026-11-14T12:00:03.000Z",')
        time.sleep(0.3)
        append(run / EVENTS, '"type":"state",')
        time.sleep(0.3)
        append(run / EVENTS, '"state":"S2","cycle":1}\n')
        snap = sse.until(lambda s: s["state"] == "S2")
        assert snap["control"]["bad_lines"] == 0

        append(run / EVENTS, "garbage line\n" + ev("state", state="WAIT_BRICK", cycle=2))
        snap = sse.until(lambda s: s["state"] == "WAIT_BRICK" and s["cycle"] == 2)
        assert snap["control"]["bad_lines"] == 1
    finally:
        sse.close()


def test_sse_follows_current_across_run_switch_and_no_run(runs: Path, server, client: Client):
    run_a = make_run(runs, "run-a", ev("run_start", mode="dry-run") + ev("state", state="S1", cycle=1))
    set_current(runs, "run-a")
    sse = Sse(server.server_address[1])
    try:
        sse.until(lambda s: s["run_id"] == "run-a" and s["state"] == "S1")

        run_b = make_run(runs, "run-b", ev("run_start", mode="live") + ev("state", state="WAIT_BRICK", cycle=5))
        set_current(runs, "run-b")
        snap = sse.until(lambda s: s["run_id"] == "run-b")
        assert snap["mode"] == "live" and snap["cycle"] == 5 and snap["history"] == [] and snap["inspection"] is None

        append(run_a / EVENTS, ev("state", state="S3", cycle=1))  # old run keeps writing: must not leak
        append(run_b / EVENTS, ev("state", state="S1", cycle=5))
        snap = sse.until(lambda s: s["state"] == "S1")
        assert snap["run_id"] == "run-b" and snap["stage"] == "S1_PICK"

        os.unlink(runs / "current")
        snap = sse.until(lambda s: s["has_run"] is False)
        assert snap["stage"] == "NO_RUN" and snap["run_id"] is None
    finally:
        sse.close()


def test_sse_resets_when_events_file_is_replaced(runs: Path, server, client: Client):
    run = make_run(runs, "run-r", ev("run_start", mode="dry-run") + ev("state", state="S1", cycle=7))
    set_current(runs, "run-r")
    sse = Sse(server.server_address[1])
    try:
        sse.until(lambda s: s["cycle"] == 7)
        fresh = run / "events.new"
        fresh.write_text(ev("run_start", mode="dry-run") + ev("state", state="IDLE", cycle=0), encoding="utf-8")
        os.replace(fresh, run / EVENTS)
        snap = sse.until(lambda s: s["state"] == "IDLE")
        assert snap["cycle"] == 0 and snap["events"] == 2
    finally:
        sse.close()


def test_api_state_one_shot_matches_stream(runs: Path, client: Client):
    make_run(runs, "run-s", ev("run_start", mode="dry-run") + ev("state", state="WAIT_BRICK", cycle=1)
             + ev("stats", cycles=4, good=2, defective=1, uncertain=1, mean_cycle_s=2.5))
    set_current(runs, "run-s")
    status, ctype, payload = client.request("GET", "/api/state")
    snap = json.loads(payload)
    assert status == 200 and ctype.startswith("application/json")
    assert snap["stage"] == "WAIT_BRICK" and snap["counts"] == {
        "source": "stats", "cycles": 4, "good": 2, "defective": 1, "uncertain": 1, "mean_cycle_s": 2.5}


# ----- control.json ----------------------------------------------------------


def test_control_ids_strictly_increase_and_survive_a_deleted_file(runs: Path, client: Client):
    run = make_run(runs, "run-c", ev("run_start", mode="dry-run"))
    set_current(runs, "run-c")
    ids = []
    for cmd in ("start", "next", "stop"):
        status, body = client.post_control(cmd)
        assert status == 200 and body["ok"] is True and body["run_id"] == "run-c"
        ids.append(body["id"])
    assert ids == [1, 2, 3]
    assert json.loads((run / "control.json").read_text(encoding="utf-8")) == {"cmd": "stop", "id": 3}

    (run / "control.json").write_text('{"cmd":"next","id":10}', encoding="utf-8")  # written by someone else
    status, body = client.post_control("next")
    assert body["id"] == 11

    os.unlink(run / "control.json")  # gone from disk: the server still never goes back
    status, body = client.post_control("start")
    assert body["id"] == 12


def test_control_json_is_never_seen_partial(runs: Path):
    run = make_run(runs, "run-p", ev("run_start", mode="dry-run"))
    writer = ControlWriter()
    seen: list[int] = []
    errors: list[str] = []
    done = threading.Event()

    def reader() -> None:
        while not done.is_set():
            try:
                data = json.loads((run / "control.json").read_bytes())
                seen.append(data["id"])
            except FileNotFoundError:
                continue
            except ValueError as exc:
                errors.append(str(exc))

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        for _ in range(400):
            writer.issue(run, "next")
    finally:
        done.set()
        thread.join()
    assert errors == []
    assert seen == sorted(seen)
    assert json.loads((run / "control.json").read_text(encoding="utf-8")) == {"cmd": "next", "id": 400}
    assert not [name for name in os.listdir(run) if name.endswith(".tmp")]


def test_control_writes_nothing_but_control_json(runs: Path, client: Client):
    run = make_run(runs, "run-w", ev("run_start", mode="dry-run") + ev("state", state="WAIT_BRICK", cycle=1))
    set_current(runs, "run-w")
    before = sorted(os.listdir(run))
    events_before = (run / EVENTS).read_bytes()
    for cmd in ("start", "stop", "quit"):
        assert client.post_control(cmd)[0] == 200
    assert sorted(os.listdir(run)) == sorted(before + ["control.json"])
    assert (run / EVENTS).read_bytes() == events_before


def test_control_refusals(runs: Path, server, client: Client):
    make_run(runs, "run-x", ev("run_start", mode="dry-run"))
    set_current(runs, "run-x")
    assert client.request("POST", "/api/control", body={"cmd": "reboot"})[0] == 400
    assert client.request("POST", "/api/control", body={"cmd": "start"},
                          headers={"Origin": "http://evil.example"})[0] == 403
    assert client.request("POST", "/api/control", raw_body=b'{"cmd":"stop"}',
                          headers={"Content-Type": "text/plain"})[0] == 415
    assert client.request("POST", "/api/control", raw_body=b"not json",
                          headers={"Content-Type": "application/json"})[0] == 400
    assert client.request("POST", "/api/control", raw_body=b"",
                          headers={"Content-Type": "application/json"})[0] == 400
    assert client.request("POST", "/api/control", raw_body=b"x" * 600,
                          headers={"Content-Type": "application/json"})[0] == 400
    assert client.request("POST", "/api/control", body={"cmd": "start"},
                          headers={"Host": "evil.example:1"})[0] == 403
    assert client.request("GET", "/api/control")[0] == 404
    os.unlink(runs / "current")
    status, body = client.post_control("start")
    assert status == 409 and body["ok"] is False


def test_only_loopback_hosts_may_bind(runs: Path):
    with pytest.raises(ValueError):
        DashboardServer(runs, "0.0.0.0", 0)


# ----- files and paths -------------------------------------------------------


def test_static_and_index(client: Client):
    status, ctype, body = client.request("GET", "/")
    assert status == 200 and ctype.startswith("text/html") and b'id="verdict"' in body
    status, ctype, body = client.request("GET", "/static/app.js")
    assert status == 200 and ctype.startswith("text/javascript") and b"sendCommand" in body
    assert client.request("GET", "/static/style.css")[1].startswith("text/css")
    assert client.request("GET", "/static/missing.js")[0] == 404


def test_only_run_dir_files_are_served(tmp_path: Path, runs: Path, client: Client):
    run_a = make_run(runs, "run-a", ev("run_start", mode="dry-run"))
    make_run(runs, "run-b", "b-events\n")
    (run_a / "frames" / "c001_pre.jpg").write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    (run_a / "live.jpg").write_bytes(b"\xff\xd8\xff\xe0live")
    (run_a / ".hidden").write_text("hidden", encoding="utf-8")
    set_current(runs, "run-a")

    status, ctype, body = client.request("GET", "/run/current/frames/c001_pre.jpg")
    assert status == 200 and ctype == "image/jpeg" and body.startswith(b"\xff\xd8")
    assert client.request("GET", "/run/current/live.jpg")[0] == 200
    assert client.request("GET", "/run/current/events.jsonl")[0] == 404  # only images are served
    assert client.request("GET", "/run/run-a/live.jpg")[0] == 404  # only the current run is served
    assert client.request("GET", "/run/run-b/live.jpg")[0] == 404
    assert client.request("GET", "/run/current/.hidden")[0] == 403
    assert client.request("GET", "/run/current/frames")[0] == 404  # directory
    assert client.request("GET", "/run/current/frames/")[0] == 403  # trailing slash
    assert client.request("GET", "/run/current")[0] == 404
    assert client.request("GET", "/run/nosuchrun/live.jpg")[0] == 404
    assert client.request("GET", "/run/current/missing.jpg")[0] == 404
    assert client.request("GET", "/run/current/../run-b/events.jsonl")[0] == 403


def test_path_traversal_is_refused(tmp_path: Path, runs: Path, client: Client):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET", encoding="utf-8")
    run = make_run(runs, "run-t", ev("run_start", mode="dry-run"))
    set_current(runs, "run-t")
    (run / "link.txt").symlink_to(secret)
    (run / "linkdir").symlink_to(tmp_path)
    attacks = [
        "/run/current/../../secret.txt",
        "/run/current/%2e%2e/%2e%2e/secret.txt",
        "/run/%2e%2e/secret.txt",
        "/run/current/..%2f..%2fsecret.txt",
        "/run/current/frames/..%2f..%2f..%2fsecret.txt",
        "/run/current/..%5c..%5csecret.txt",
        "/run/../run-t/events.jsonl",
        "/run/current/%00",
        "/run/current/events.jsonl%00.txt",
        "/static/../../secret.txt",
        "/static/..%2f..%2fsecret.txt",
        "/static/..%5c..%5csecret.txt",
        "/static/%2e%2e%2fsecret.txt",
        "//etc/passwd",
        "/run/current/link.txt",
        "/run/current/linkdir/secret.txt",
    ]
    for path in attacks:
        status, _, body = client.request("GET", path)
        assert status in (403, 404), (path, status)
        assert b"TOPSECRET" not in body, path
        assert b"root:" not in body, path


def test_run_alias_outside_runs_root_is_ignored(tmp_path: Path, runs: Path, client: Client):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / EVENTS).write_text(ev("state", state="S1", cycle=9), encoding="utf-8")
    os.symlink(outside, runs / "current")
    status, _, body = client.request("GET", "/api/state")
    assert json.loads(body)["has_run"] is False
    assert client.request("GET", "/run/current/events.jsonl")[0] == 404


def test_foreign_host_header_is_refused(client: Client):
    assert client.request("GET", "/", headers={"Host": "evil.example:80"})[0] == 403
    assert client.request("GET", "/api/state", headers={"Host": "evil.example:80"})[0] == 403


# ----- fake run end to end ---------------------------------------------------


def _wait_for(predicate, timeout: float, message: str):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError(message)


def _run_fake(tmp_path: Path, runs: Path, client: Client, extra: list[str], run_id: str):
    out = open(tmp_path / f"{run_id}.out", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-I", str(FAKE_RUN), "--runs", str(runs), "--speed", "40", "--seed", "3",
         "--run-id", run_id, *extra],
        stdout=out, stderr=subprocess.STDOUT,
    )
    try:
        _wait_for(lambda: (runs / "current").is_symlink(), 20, "fake run never set current")
        status, body = client.post_control("start")
        assert status == 200 and body["id"] == 1
        proc.wait(timeout=90)
    finally:
        if proc.poll() is None:
            proc.kill()
        out.close()
    assert proc.returncode == 0, (tmp_path / f"{run_id}.out").read_text(encoding="utf-8")[-800:]
    return runs / run_id


def test_fake_run_end_to_end_through_dashboard(tmp_path: Path, runs: Path, server, client: Client):
    run = _run_fake(tmp_path, runs, client, ["--cycles", "2", "--auto-next", "--partial-lines"], "e2e")
    lines = (run / EVENTS).read_text(encoding="utf-8").splitlines()
    kinds = [json.loads(line)["type"] for line in lines]  # every line is complete JSON
    assert kinds[0] == "run_start" and kinds[-1] == "run_end"
    assert kinds.count("inspection") == 2 and kinds.count("stats") == 2
    assert json.loads(lines[-1])["reason"] == "max_cycles"
    for name in ("live.jpg", "frames/c001_top_pre.jpg", "frames/c001_top_crop.jpg",
                 "frames/c001_close_pre.jpg", "frames/c002_top_pre.jpg"):
        assert (run / name).read_bytes()[:2] == b"\xff\xd8", name

    sse = Sse(server.server_address[1])
    try:
        snap = sse.until(lambda s: s["stage"] == "RUN_ENDED")
    finally:
        sse.close()
    assert snap["control"]["bad_lines"] == 0 and snap["events"] == len(lines)
    assert snap["counts"]["source"] == "stats" and snap["counts"]["cycles"] == 2
    assert snap["acked"]["cmd"] == "start" and snap["control"]["issued"] == {"cmd": "start", "id": 1}
    assert len(snap["history"]) == 2
    for insp in snap["history"]:
        status, ctype, body = client.request("GET", insp["frame_url"])
        assert status == 200 and ctype == "image/jpeg" and body[:2] == b"\xff\xd8"
        assert client.request("GET", insp["crop_url"])[0] == 200


def test_fake_run_fault_holds_then_quit(tmp_path: Path, runs: Path, server, client: Client):
    out = open(tmp_path / "fault.out", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-I", str(FAKE_RUN), "--runs", str(runs), "--speed", "40", "--seed", "3",
         "--run-id", "fault", "--auto-next", "--error-after", "1"],
        stdout=out, stderr=subprocess.STDOUT,
    )
    try:
        _wait_for(lambda: (runs / "current").is_symlink(), 20, "fake run never set current")
        assert client.post_control("start")[0] == 200
        sse = Sse(server.server_address[1])
        try:
            snap = sse.until(lambda s: s["stage"] == "STOPPED")
        finally:
            sse.close()
        assert snap["error"] is not None and snap["error"]["message"] == "skill timeout after 60 s"
        assert snap["counts"]["cycles"] == 1 and snap["acked"]["cmd"] == "start"
        status, body = client.post_control("quit")
        assert status == 200 and body["id"] == 2
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
        out.close()
    assert proc.returncode == 0
    last = json.loads((runs / "fault" / EVENTS).read_text(encoding="utf-8").splitlines()[-1])
    assert last == {**last, "type": "run_end", "reason": "quit"}
