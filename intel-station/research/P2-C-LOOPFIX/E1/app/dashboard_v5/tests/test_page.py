"""Browser tests: Playwright drives Chrome against a live dashboard server (a separate process).

They cover what the server tests cannot see: the OD-10 verdict colour and text (A1 M5b, A2 M6,
A3 M5), the STOP button path (A1 M6), the detail line never showing an earlier cycle (A1 F48),
the detail line on a 1366x768 screen (A1 F46), the server-offline badge (A1 F50) and that the
page makes no request off the loopback address.

Needs Playwright for Python and Chrome (DASHBOARD_CHROME, default /usr/bin/google-chrome).
A missing browser fails these tests; they are never skipped.

Run:  python3 -I -m pytest -q tests        (from the dashboard code folder)
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from test_dashboard import EVENTS, append, ev, make_run, runs, set_current  # noqa: F401 (fixture)

APP_DIR = Path(__file__).resolve().parents[1] / "app"
CHROME = os.environ.get("DASHBOARD_CHROME", "/usr/bin/google-chrome")
WAIT_MS = 15000


def base_events(state: str = "WAIT_BRICK", cycle: int = 1) -> str:
    return (ev("run_start", mode="dry-run",
               config={"live_source": "wrist camera USB2.0_CAM1 (live.jpg)",
                       "detector": {"backend": "openvino", "device": "GPU"}})
            + ev("state", state=state, cycle=cycle))


def inspection(cycle: int, label: str, route: str | None, reasons: list[str] | None = None) -> str:
    score = 0.12 if label == "GOOD" else 0.9
    return ev("inspection", cycle=cycle, label=label, score=score, threshold=0.42,
              reasons=reasons or [], device="GPU", latency_ms=11, route=route)


def wait_for(predicate, timeout: float, message: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError(message)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class LiveServer:
    """The dashboard as its own process, the way it runs on the laptop, so the test can stop it."""

    def __init__(self, runs_root: Path) -> None:
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}/"
        self.log = open(runs_root.parent / "server.log", "w", encoding="utf-8")
        env = dict(os.environ, PYTHONPATH=str(APP_DIR), PYTHONDONTWRITEBYTECODE="1")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "dashboard.server", "--runs", str(runs_root), "--port", str(self.port),
             "--host", "127.0.0.1"],
            cwd=str(APP_DIR), env=env, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 20
        while True:
            try:
                with urllib.request.urlopen(self.url + "api/state", timeout=2) as resp:
                    if resp.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() > deadline:
                self.close()
                raise AssertionError("dashboard server did not start")
            time.sleep(0.2)

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=10)

    def close(self) -> None:
        self.stop()
        self.log.close()


@pytest.fixture(scope="module")
def chromium():
    if not os.path.exists(CHROME):
        raise AssertionError(f"Chrome not found at {CHROME}; set DASHBOARD_CHROME")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=CHROME, headless=True, args=["--no-sandbox", "--disable-gpu"])
        yield browser
        browser.close()


class PageView:
    """One page in one browser tab: records uncaught errors, requests and POST bodies."""

    def __init__(self, browser, url: str, width: int, height: int) -> None:
        self.page = browser.new_page(viewport={"width": width, "height": height})
        self.errors: list[str] = []
        self.hosts: set[str] = set()
        self.posts: list[object] = []
        self.page.on("pageerror", lambda exc: self.errors.append(str(exc)))
        self.page.on("request", self._on_request)
        self.page.goto(url, wait_until="load")

    def _on_request(self, request) -> None:
        match = re.match(r"https?://([^/:]+)", request.url)
        if match:
            self.hosts.add(match.group(1))
        if request.method == "POST":
            self.posts.append(json.loads(request.post_data or "null"))

    def eval(self, js: str):
        return self.page.evaluate(js)

    def wait_until(self, js: str, timeout_ms: int = WAIT_MS) -> None:
        # Polled from Python: wait_for_function evaluates its predicate as a string, which the page's CSP forbids.
        deadline = time.monotonic() + timeout_ms / 1000
        while True:
            if self.eval(f"Boolean({js})"):
                return
            if time.monotonic() > deadline:
                raise AssertionError(f"page never reached: {js}; errors={self.errors}")
            time.sleep(0.1)

    def text(self, element_id: str) -> str:
        return self.eval(f"document.getElementById('{element_id}').textContent") or ""

    def has_class(self, element_id: str, cls: str) -> bool:
        return self.eval(f"document.getElementById('{element_id}').classList.contains('{cls}')")

    def background(self, element_id: str) -> tuple[int, int, int]:
        value = self.eval(f"getComputedStyle(document.getElementById('{element_id}')).backgroundColor")
        red, green, blue = (int(x) for x in re.findall(r"\d+", value)[:3])
        return red, green, blue

    def close(self) -> None:
        self.page.close()


def _page(chromium, server: LiveServer, width: int = 1920, height: int = 1080) -> PageView:
    return PageView(chromium, server.url, width, height)


def test_verdict_colour_and_text_follow_the_plate(chromium, tmp_path):
    runs_root = tmp_path / "loop"
    runs_root.mkdir()
    run = make_run(runs_root, "run-col", base_events() + inspection(1, "GOOD", "S2"))
    set_current(runs_root, "run-col")
    server = LiveServer(runs_root)
    page = _page(chromium, server)
    try:
        page.wait_until("document.getElementById('verdict-plate').textContent === 'BLUE PLATE'")
        r, g, b = page.background("verdict")
        assert page.has_class("verdict", "is-blue") and b > r and b > g
        assert page.text("verdict-safe").strip() == ""

        append(run / EVENTS, inspection(2, "DEFECTIVE", "S3", ["tape/appearance"]))
        page.wait_until("document.getElementById('verdict-plate').textContent === 'GREEN PLATE'")
        r, g, b = page.background("verdict")
        assert page.has_class("verdict", "is-green") and g > r and g > b
        assert page.text("verdict-safe").strip() == ""

        append(run / EVENTS, inspection(3, "UNCERTAIN", "S3", ["tape/appearance"]))
        page.wait_until("document.getElementById('verdict-safe').textContent.trim() === '(safe)'")
        assert page.text("verdict-plate") == "GREEN PLATE"
        assert page.has_class("verdict", "is-green")
        assert page.errors == []
        assert page.hosts == {"127.0.0.1"}  # the page reaches the loopback address only
    finally:
        page.close()
        server.close()


def test_stop_button_posts_stop_and_shows_the_ack(chromium, tmp_path):
    runs_root = tmp_path / "loop"
    runs_root.mkdir()
    run = make_run(runs_root, "run-stop", base_events())
    set_current(runs_root, "run-stop")
    server = LiveServer(runs_root)
    page = _page(chromium, server)
    try:
        page.wait_until("document.body.dataset.stage === 'WAIT_BRICK' && !document.getElementById('btn-stop').disabled")
        page.page.click("#btn-stop")
        wait_for(lambda: (run / "control.json").exists(), 10, "STOP wrote no control.json")
        assert json.loads((run / "control.json").read_text(encoding="utf-8")) == {"cmd": "stop", "id": 1}
        assert page.posts == [{"cmd": "stop"}]
        append(run / EVENTS, ev("control", cmd="stop", id=1) + ev("state", state="STOPPED", cycle=1))
        page.wait_until("document.body.dataset.stage === 'STOPPED'")
        page.wait_until("document.getElementById('cmd-state').textContent.includes('loop acked stop #1')")
        assert page.errors == []
    finally:
        page.close()
        server.close()


def test_detail_line_never_shows_an_earlier_cycles_sort(chromium, tmp_path):
    runs_root = tmp_path / "loop"
    runs_root.mkdir()
    run = make_run(runs_root, "run-stale", base_events("WAIT_BRICK", 3)
                   + ev("state", state="SORT", cycle=3) + ev("sort_start", route="S2")
                   + ev("sort_end", ok=True, duration_s=3.1) + inspection(3, "GOOD", "S2")
                   + ev("state", state="CYCLE_DONE", cycle=3))
    set_current(runs_root, "run-stale")
    server = LiveServer(runs_root)
    page = _page(chromium, server)
    try:
        page.wait_until("document.getElementById('detail').textContent.includes('sort to BLUE PLATE done')")
        append(run / EVENTS, ev("state", state="S1", cycle=4) + ev("state", state="INSPECT", cycle=4)
               + inspection(4, "UNCERTAIN", "S3", ["tape/appearance"]))
        page.wait_until("document.getElementById('verdict-plate').textContent === 'GREEN PLATE'")
        page.wait_until("document.getElementById('verdict-safe').textContent.trim() === '(safe)'")
        detail = page.text("detail")
        assert "BLUE" not in detail and "sort to" not in detail, detail
    finally:
        page.close()
        server.close()


LONG_ROUTE = "S3 spot to green plate with two align steps"  # no plate name: the route text is shown

@pytest.mark.parametrize("width,height,route", [(1366, 768, "S3"), (1920, 1080, "S3"), (1366, 768, LONG_ROUTE)])
def test_detail_line_is_not_clipped(chromium, tmp_path, width, height, route):
    runs_root = tmp_path / "loop"
    runs_root.mkdir()
    # The order fake_run and the orchestrator use: handover out, then sort_start, then the align steps.
    make_run(runs_root, "run-detail", base_events("WAIT_BRICK", 4)
             + ev("state", state="S1", cycle=4) + inspection(4, "UNCERTAIN", "S3", ["tape/appearance"])
             + ev("state", state="HANDOVER_OUT", cycle=4) + ev("handover", phase="out", duration_s=0.8, model_kept=True)
             + ev("state", state="SORT", cycle=4) + ev("sort_start", route=route)
             + ev("align", iteration=1, dx=6, dy=-3, correction=0.8)
             + ev("align", iteration=2, dx=1, dy=0, correction=0.1)
             + ev("sort_end", ok=True, duration_s=3.2))
    set_current(runs_root, "run-detail")
    server = LiveServer(runs_root)
    page = _page(chromium, server, width, height)
    try:
        page.wait_until("document.getElementById('detail').textContent.includes('done')")
        assert 'sort to' in page.text('detail'), page.text('detail')
        fit = page.eval("""(() => {
          const d = document.getElementById('detail');
          return {
            vertical: d.scrollHeight <= d.clientHeight + 1,
            horizontal: d.scrollWidth <= d.clientWidth + 1,
            page: document.documentElement.scrollHeight <= window.innerHeight + 1,
          };
        })()""")
        assert fit == {"vertical": True, "horizontal": True, "page": True}, (width, height, route, fit)
    finally:
        page.close()
        server.close()


def test_server_offline_badge_after_the_server_stops(chromium, tmp_path):
    runs_root = tmp_path / "loop"
    runs_root.mkdir()
    make_run(runs_root, "run-off", base_events())
    set_current(runs_root, "run-off")
    server = LiveServer(runs_root)
    page = _page(chromium, server, 1366, 768)
    try:
        page.wait_until("document.getElementById('stage').textContent === 'WAIT FOR BRICK'")
        server.stop()
        page.wait_until("document.getElementById('link-badge').textContent === 'server offline'", timeout_ms=20000)
        assert page.has_class("link-badge", "is-bad")
    finally:
        page.close()
        server.close()


def test_stop_label_fits_when_an_error_is_shown_at_1366x768(chromium, tmp_path):
    # With the error alert on screen the STOP row is short; its label must still sit inside the button.
    runs_root = tmp_path / "loop"
    runs_root.mkdir()
    make_run(runs_root, "run-err", base_events("WAIT_BRICK", 2) + inspection(1, "GOOD", "S2")
             + ev("state", state="HANDOVER_OUT", cycle=2) + ev("handover", phase="out", duration_s=1.0, model_kept=True)
             + ev("state", state="SORT", cycle=2) + ev("sort_start", route="S3")
             + ev("error", where="SORT", message="skill timeout after 60 s")
             + ev("state", state="HOLD", cycle=2) + ev("state", state="STOPPED", cycle=2))
    set_current(runs_root, "run-err")
    server = LiveServer(runs_root)
    page = _page(chromium, server, 1366, 768)
    try:
        page.wait_until("document.body.dataset.stage === 'STOPPED' && !document.getElementById('alert').hidden")
        fit = page.eval("(() => { const b = document.getElementById('btn-stop');"
                        " return {label: b.scrollHeight <= b.clientHeight + 1, width: b.scrollWidth <= b.clientWidth + 1}; })()")
        assert fit == {"label": True, "width": True}, fit
    finally:
        page.close()
        server.close()
