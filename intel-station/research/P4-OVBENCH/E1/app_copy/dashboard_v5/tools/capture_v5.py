"""Screenshots of every v5 panel at both sizes, plus a clipping report (no robot, no Studio, no camera).

Serves --runs (a run tree written by tools/replay_v5.py) on 127.0.0.1 in this process, drives Chrome with
Playwright, and writes under --out:
  full_<W>x<H>[_video].png          the whole page
  panel_<W>x<H>_<panel>.png         one per panel: camera, verdict, side, scan, policy, intel, history, topbar
  capture_report_<W>x<H>[_video].json   per panel: box, scroll overflow (clipped content), stage, text checks

Examples:
  python3 -I tools/capture_v5.py --runs /tmp/P2-B-DASH2-E1/demo/logs/loop --telemetry-dir /tmp/P2-B-DASH2-E1/demo/logs/telemetry --bench /tmp/P2-B-DASH2-E1/demo/bench.json --out screenshots
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import socket
import sys
import threading
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from playwright.sync_api import sync_playwright  # noqa: E402

from dashboard.server import DashboardServer  # noqa: E402

PANELS = {
    "camera": ".camera",
    "verdict": ".verdict-panel",
    "side": ".side",
    "scan": ".scan-panel",
    "policy": ".policy-panel",
    "intel": ".intel-panel",
    "history": ".history",
    "topbar": ".topbar",
}
SIZES = [(1920, 1080), (1366, 768)]
CHROME = "/usr/bin/google-chrome"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def overflow_report(page) -> dict:
    """Panels whose content is taller or wider than their box: text cut by overflow hidden."""
    return page.evaluate("""() => {
      const out = {};
      for (const sel of ['.camera', '.verdict-panel', '.side', '.scan-panel', '.policy-panel',
                         '.intel-panel', '.history', '.topbar']) {
        const node = document.querySelector(sel);
        if (!node) { out[sel] = 'missing'; continue; }
        const box = node.getBoundingClientRect();
        const clipped = [];
        for (const child of node.querySelectorAll('*')) {
          if (child.closest('[hidden]') || child.offsetParent === null) continue;
          if (child.closest('.bench-wrap')) continue;  // scrolls inside its own box by design
          const r = child.getBoundingClientRect();
          if (r.width === 0 || r.height === 0) continue;
          if (r.bottom > box.bottom + 2 || r.right > box.right + 2) {
            clipped.push((child.className && child.className.baseVal === undefined ? child.className : child.tagName)
              + ' ' + Math.round(r.bottom - box.bottom) + 'px below / ' + Math.round(r.right - box.right) + 'px right');
          }
          if (clipped.length >= 6) break;
        }
        out[sel] = { box: [Math.round(box.width), Math.round(box.height)], outside: clipped };
      }
      return out;
    }""")


def keep_heartbeat(runs: Path, stop: threading.Event) -> None:
    """Appends a heartbeat each second to the current replay run, so the page reads 'loop live' while it is captured."""
    events = (runs / "current").resolve() / "events.jsonl"
    while not stop.is_set():
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int(time.time() * 1000) % 1000:03d}Z"
        with open(events, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"t": stamp, "mono": round(time.monotonic(), 3), "type": "heartbeat",
                                     "cycle": 0}) + "\n")
        stop.wait(1.0)


def keep_telemetry(telemetry: Path, stop: threading.Event) -> None:
    """Appends one synthetic sample a second in the sampler's shape (src says so), so the intel panel stays fresh."""
    rng = random.Random(3)
    while not stop.is_set():
        now = dt.datetime.now(dt.timezone.utc)
        row = {
            "t": now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z",
            "epoch": round(now.timestamp(), 3), "pkg_w": None,
            "temp_c": {"pkg": round(rng.uniform(64.0, 84.0), 2), "coretemp_pkg": 0.0, "nvme": 36.8, "core_max": 0.0},
            "cpu_util": round(rng.uniform(0.15, 0.55), 4),
            "gpu": {"freq_mhz": round(rng.uniform(300, 1900), 0), "max_mhz": 2500,
                    "freq_ratio": round(rng.uniform(0.1, 0.75), 3), "media_mhz": 0},
            "npu": {"busy": round(rng.uniform(0.0, 0.35), 4), "freq_mhz": 0},
            "mem": {"used_pct": round(rng.uniform(24.0, 27.0), 2), "avail_mb": 47000.0},
            "bat": {"status": "Full", "power_w": 0.0, "capacity": 100}, "ac_online": 1,
            "src": "replay synthetic (not measured); pkg_w:none",
        }
        with open(telemetry / f"{now:%Y%m%d}.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        stop.wait(1.0)


def capture(runs: Path, telemetry: Path | None, bench: Path | None, out: Path, width: int, height: int,
            video: bool, wait_s: float) -> dict:
    port = free_port()
    server = DashboardServer(runs, "127.0.0.1", port, telemetry, bench)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    beat_stop = threading.Event()
    threading.Thread(target=keep_heartbeat, args=(runs, beat_stop), daemon=True).start()
    if telemetry is not None:
        threading.Thread(target=keep_telemetry, args=(telemetry, beat_stop), daemon=True).start()
    suffix = f"{width}x{height}" + ("_video" if video else "")
    report: dict = {"size": [width, height], "video": video, "port": port, "panels": {}}
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=CHROME, headless=True,
                                         args=["--no-sandbox", "--disable-gpu"])
            page = browser.new_page(viewport={"width": width, "height": height})
            url = f"http://127.0.0.1:{port}/" + ("?video=1" if video else "")
            page.goto(url)
            deadline = time.monotonic() + wait_s
            while time.monotonic() < deadline:
                ready = page.evaluate("""() => {
                  const txt = (id) => (document.getElementById(id) || {}).textContent || '';
                  return txt('scan-meta') !== 'no scan yet' && txt('policy-meta') !== ''
                    && document.getElementById('intel-live').children.length > 0
                    && document.getElementById('bench').querySelectorAll('tr').length > 1
                    && txt('link-badge') === 'loop live';
                }""")
                if ready:
                    break
                time.sleep(0.3)
            time.sleep(0.8)  # let the 250 ms tick and the live preview settle
            report["stage"] = page.text_content("#stage")
            report["link"] = page.text_content("#link-badge")
            report["ready"] = bool(ready)
            page.screenshot(path=str(out / f"full_{suffix}.png"))
            for name, selector in PANELS.items():
                page.locator(selector).first.screenshot(path=str(out / f"panel_{suffix}_{name}.png"))
            report["overflow"] = overflow_report(page)
            report["checks"] = page.evaluate("""() => ({
              scan_bricks: document.querySelectorAll('#bricks .brick').length,
              target_marked: document.querySelectorAll('#bricks .brick.is-target').length,
              direct_marked: [...document.querySelectorAll('#bricks .brick-route')].map(n => n.textContent),
              timeline_steps: [...document.querySelectorAll('#timeline .step-name')].map(n => n.textContent),
              decision: document.getElementById('decision-plate').textContent,
              verify: document.getElementById('verify').textContent,
              frame_visible: !document.getElementById('frame-card').hidden,
              bench_rows: document.querySelectorAll('#bench tbody tr').length,
              video_on: document.body.classList.contains('video'),
              debug_visible: [...document.querySelectorAll('.debug')].filter(n => n.offsetParent !== null).length,
            })""")
            browser.close()
    finally:
        beat_stop.set()
        server.shutdown()
        server.server_close()
    (out / f"capture_report_{suffix}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Screenshots of the v5 dashboard panels at both sizes.")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--telemetry-dir", type=Path, default=None)
    parser.add_argument("--bench", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wait", type=float, default=20.0)
    parser.add_argument("--sizes", default="1920x1080,1366x768")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    sizes = [tuple(int(v) for v in s.split("x")) for s in args.sizes.split(",")]
    for width, height in sizes:
        for video in (False, True):
            result = capture(args.runs, args.telemetry_dir, args.bench, args.out, width, height, video, args.wait)
            print(f"{width}x{height} video={video} stage={result['stage']} link={result['link']} "
                  f"ready={result['ready']} checks={json.dumps(result['checks'])[:300]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
