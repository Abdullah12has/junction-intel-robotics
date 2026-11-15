"""v5 browser tests: the scan, policy and intel panels, video mode, and the layout at 1920x1080 and 1366x768.

A synthetic replay run (tools/replay_v5.py) is served by a dashboard process of its own, the way the laptop runs
it, and Chrome drives it through Playwright. The clipping check is capture_v5.overflow_report, the same check that
writes the screenshot reports, so a test and a screenshot report agree.

Needs Playwright for Python and Chrome (DASHBOARD_CHROME, default /usr/bin/google-chrome). A missing browser fails
these tests; they are never skipped.

Run:  python3 -I -m pytest -q tests        (from the dashboard code folder)
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
sys.path.insert(0, str(ROOT / "tools"))

import replay_v5  # noqa: E402  (tools/: the synthetic run writer)
from capture_v5 import overflow_report  # noqa: E402  (tools/: the clipping check the screenshots use)

from test_page import PageView, chromium  # noqa: E402,F401  (fixture: one Chrome for this module)

SIZES = [(1920, 1080), (1366, 768)]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def replay_out(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("v5-page")
    replay_v5.build(out, cycles=5, run_id="replay-page", pace=0.0, seed=5)
    return out


class V5Server:
    """The dashboard as its own process, with the replay's runs, telemetry folder and bench.json."""

    def __init__(self, out: Path) -> None:
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}/"
        self.log = open(out / "server-v5.log", "w", encoding="utf-8")
        env = dict(os.environ, PYTHONPATH=str(APP_DIR), PYTHONDONTWRITEBYTECODE="1")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "dashboard.server", "--runs", str(out / "logs" / "loop"),
             "--telemetry-dir", str(out / "logs" / "telemetry"), "--bench", str(out / "bench.json"),
             "--port", str(self.port), "--host", "127.0.0.1"],
            cwd=str(APP_DIR), env=env, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 20
        while True:
            try:
                with urllib.request.urlopen(self.url + "api/state", timeout=2) as resp:
                    if resp.status == 200:
                        return
            except OSError:
                pass
            if time.monotonic() > deadline:
                self.close()
                raise AssertionError("dashboard server did not start")
            time.sleep(0.2)

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=10)
        self.log.close()


def wait_idle(page: PageView) -> None:
    page.wait_until("document.body.dataset.stage === 'IDLE' && document.querySelectorAll('#bench tbody tr').length === 12")


@pytest.mark.parametrize("video", [False, True], ids=["normal", "video"])
@pytest.mark.parametrize("width,height", SIZES)
def test_scan_policy_and_intel_show_the_run_without_clipping(chromium, replay_out, width, height, video):
    server = V5Server(replay_out)
    page = PageView(chromium, server.url + ("?video=1" if video else ""), width, height)
    try:
        wait_idle(page)
        checks = page.eval("""() => ({
          bricks: document.querySelectorAll('#bricks .brick').length,
          target: document.querySelectorAll('#bricks .brick.is-target').length,
          routes: [...document.querySelectorAll('#bricks .brick-route')].map(n => n.textContent),
          steps: [...document.querySelectorAll('#timeline .step-name')].map(n => n.textContent),
          decision: document.getElementById('decision-plate').textContent,
          verify: document.getElementById('verify').textContent,
          bench: document.querySelectorAll('#bench tbody tr').length,
          video: document.body.classList.contains('video'),
          debugShown: [...document.querySelectorAll('.debug')].filter(n => n.offsetParent !== null).length,
          stopLabelFits: (() => { const b = document.getElementById('btn-stop');
            return b.scrollHeight <= b.clientHeight + 1 && b.scrollWidth <= b.clientWidth + 1; })(),
          clippedCards: [...document.querySelectorAll('#cams .cam, #cams .frame-card')].filter(c => !c.hidden).flatMap(c => {
            const box = c.getBoundingClientRect();
            return [...c.querySelectorAll('*')].filter(n => {
              const r = n.getBoundingClientRect();
              return n.offsetParent !== null && r.height > 0 && r.bottom > box.bottom + 1;
            }).map(n => n.className || n.tagName);
          }),
        })""")
        assert checks["bricks"] == 3 and checks["target"] == 1
        assert checks["routes"] == ["inspected", "inspected", "straight to GREEN (I-52)"]
        assert checks["steps"] == ["start", "grasp", "carry", "release", "return", "done"]
        assert checks["decision"] == "BLUE PLATE"
        assert checks["verify"] == "verify OK: intended BLUE, released BLUE"
        assert checks["bench"] == 12
        assert checks["clippedCards"] == [], checks["clippedCards"]  # reason chips stay inside their card
        assert checks["stopLabelFits"] is True  # the STOP label sits inside its button
        assert checks["video"] is video
        if video:
            assert checks["debugShown"] == 0
        clipped = {name: found["outside"] for name, found in overflow_report(page.page).items()
                   if isinstance(found, dict) and found["outside"]}
        assert clipped == {}, (width, height, video, clipped)
        assert page.errors == []
        assert page.hosts == {"127.0.0.1"}  # the page reaches the loopback address only
    finally:
        page.close()
        server.close()


def test_video_button_switches_the_mode_and_the_url(chromium, replay_out):
    server = V5Server(replay_out)
    page = PageView(chromium, server.url, 1366, 768)
    try:
        wait_idle(page)
        assert page.eval("document.body.classList.contains('video')") is False
        page.page.click("#btn-video")
        page.wait_until("document.body.classList.contains('video')")
        assert page.eval("document.getElementById('btn-video').getAttribute('aria-pressed')") == "true"
        assert page.eval("location.search") == "?video=1"
        page.page.click("#btn-video")
        page.wait_until("!document.body.classList.contains('video')")
        assert page.eval("location.search") == ""
        assert page.errors == []
    finally:
        page.close()
        server.close()
