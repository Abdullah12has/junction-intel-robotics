"""Drive the dashboard in headless Chrome against tools/fake_run.py and save a screenshot of every view.

One viewport and one scenario per call. Starts fake_run and the dashboard server on a free loopback
port, presses Start, saves a PNG the first time each view is on screen, then (replay) presses STOP.
Writes <out>/capture_report_<WxH>_<scenario>.json with page checks (page errors, overflow, broken
images, camera cards) and the list of views that were missed.

Scenarios:
  replay  scripted demo: WAIT_BRICK, S1, INSPECT, GOOD (BLUE), DEFECTIVE tape and not 2x2 (GREEN),
          UNCERTAIN (GREEN, safe), HANDOVER, SORT to BLUE and GREEN, CYCLE_DONE with 4 tiles, STOPPED
  fault   --error-after 1: the sort fails after a handover, the page shows the error alert at STOPPED
  policy  --sort-mode policy: the team S2 and S3 skills (stages S2_BLUE and S3_GREEN)

Usage (from the dashboard code folder):
  python3 -I tools/capture_screenshots.py --width 1920 --height 1080 --out screenshots \\
      --work /tmp/P2-B-DASH-F1/shots --scenario replay
Needs Playwright for Python and Chrome (default /usr/bin/google-chrome).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
APP = HERE / "app"
FAKE = HERE / "tools" / "fake_run.py"

SCENARIOS = {
    "replay": {"args": [], "wanted": ["wait_brick", "s1_pick", "inspect", "good_blue", "handover",
                                       "sort_SORT_BLUE", "defective_green", "reason_tape", "uncertain_safe",
                                       "sort_SORT_GREEN", "reason_not_2x2", "cycle_done"], "stop": True},
    "fault": {"args": ["--error-after", "1"], "wanted": ["error_stopped"], "stop": False},
    "policy": {"args": ["--sort-mode", "policy"], "wanted": ["s2_blue", "s3_green"], "stop": True},
}

STATE_JS = """() => ({
  stage: document.body.dataset.stage || '',
  stageText: document.getElementById('stage').textContent,
  verdict: document.getElementById('verdict-plate').textContent,
  safe: document.getElementById('verdict-safe').textContent,
  tiles: document.querySelectorAll('.tile:not(.empty)').length,
  cmd: document.getElementById('cmd-state').textContent,
  cams: document.querySelectorAll('.cam').length,
  chips: [...document.querySelectorAll('.chip')].map((c) => c.textContent),
  alert: !document.getElementById('alert').hidden,
  broken: [...document.images].filter((i) => !i.hidden && i.complete && i.naturalWidth === 0 && i.getAttribute('src')).length,
  overflowX: document.documentElement.scrollWidth > window.innerWidth,
  overflowY: document.documentElement.scrollHeight > window.innerHeight,
  liveSource: document.getElementById('live-source').textContent,
})"""


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_http(url: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except OSError:
            time.sleep(0.2)
    return False


def hits_for(st: dict) -> list[str]:
    found = []
    if st["stage"] == "WAIT_BRICK":
        found.append("wait_brick")
    if st["stage"] == "S1_PICK":
        found.append("s1_pick")
    if st["stage"] == "INSPECT":
        found.append("inspect")
    if st["stage"] == "HANDOVER":
        found.append("handover")
    if st["stage"] in ("SORT_BLUE", "SORT_GREEN"):
        found.append(f"sort_{st['stage']}")
    if st["stage"] in ("S2_BLUE", "S3_GREEN"):
        found.append(st["stage"].lower())
    if st["verdict"] == "BLUE PLATE":
        found.append("good_blue")
    if st["verdict"] == "GREEN PLATE" and not st["safe"].strip():
        found.append("defective_green")
    if "(safe)" in st["safe"]:
        found.append("uncertain_safe")
    if "tape" in st["chips"]:
        found.append("reason_tape")
    if "not 2x2" in st["chips"]:
        found.append("reason_not_2x2")
    if st["stage"] == "CYCLE_DONE" and st["tiles"] >= 4:
        found.append("cycle_done")
    if st["stage"] == "STOPPED" and st["alert"]:
        found.append("error_stopped")
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Headless Chrome screenshots of the dashboard driven by fake_run.")
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True, help="scratch root for the fake run and logs")
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default="replay")
    ap.add_argument("--speed", type=float, default=0.8)
    ap.add_argument("--chrome", default="/usr/bin/google-chrome")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args(argv)

    from playwright.sync_api import sync_playwright

    scenario = SCENARIOS[args.scenario]
    tag = f"{args.width}x{args.height}"
    runs = args.work / f"{tag}_{args.scenario}" / "logs" / "loop"
    runs.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(APP))
    log = open(args.work / f"{tag}_{args.scenario}.fake.log", "w", encoding="utf-8")
    fake = subprocess.Popen(
        [sys.executable, "-I", str(FAKE), "--runs", str(runs), "--run-id", "shots", "--speed", str(args.speed),
         "--auto-next", "--auto-gap", "2", "--seed", "7", *scenario["args"]],
        stdout=log, stderr=subprocess.STDOUT, env=env)
    port = free_port()
    srv = subprocess.Popen(
        [sys.executable, "-m", "dashboard.server", "--runs", str(runs), "--port", str(port), "--host", "127.0.0.1"],
        cwd=str(APP), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    report: dict = {"viewport": tag, "scenario": args.scenario, "url": f"http://127.0.0.1:{port}/",
                    "shots": {}, "checks": {}, "errors": []}
    try:
        if not wait_http(f"http://127.0.0.1:{port}/", 20):
            raise SystemExit("dashboard server did not start")
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=args.chrome, headless=True,
                                         args=["--no-sandbox", "--disable-gpu"])
            page = browser.new_page(viewport={"width": args.width, "height": args.height})
            page.on("pageerror", lambda exc: report["errors"].append(f"pageerror: {exc}"))
            page.on("console", lambda msg: report["errors"].append(f"console.{msg.type}: {msg.text}")
                    if msg.type == "error" and "Failed to load resource" not in msg.text else None)
            page.goto(report["url"], wait_until="load")
            page.wait_for_selector("#btn-start:not([disabled])", timeout=30000)
            page.click("#btn-start")
            wanted = set(scenario["wanted"])
            deadline = time.time() + args.timeout
            while wanted and time.time() < deadline:
                st = page.evaluate(STATE_JS)
                checks = report["checks"]
                checks["overflowX_seen"] = checks.get("overflowX_seen", False) or st["overflowX"]
                checks["overflowY_seen"] = checks.get("overflowY_seen", False) or st["overflowY"]
                checks["broken_images_max"] = max(checks.get("broken_images_max", 0), st["broken"])
                checks["cams_max"] = max(checks.get("cams_max", 0), st["cams"])
                checks["live_source"] = st["liveSource"]
                for name in hits_for(st):
                    if name in wanted:
                        wanted.discard(name)
                        shot = f"dashboard_{tag}_{name}.png"
                        page.screenshot(path=str(args.out / shot))
                        report["shots"][name] = {"file": shot, "stage": st["stageText"],
                                                 "verdict": (st["verdict"] + " " + st["safe"]).strip(),
                                                 "detail": page.text_content("#detail") or "",
                                                 "cmd": st["cmd"], "cams": st["cams"], "chips": st["chips"]}
                time.sleep(0.2)
            if scenario["stop"]:
                page.click("#btn-stop")
                stop_deadline = time.time() + 20
                while time.time() < stop_deadline:
                    st = page.evaluate(STATE_JS)
                    if st["stage"] in ("STOPPED", "HOLD"):
                        break
                    time.sleep(0.2)
                time.sleep(0.5)
                st = page.evaluate(STATE_JS)
                shot = (f"dashboard_{tag}_stopped.png" if args.scenario == "replay"
                        else f"dashboard_{tag}_{args.scenario}_stopped.png")
                page.screenshot(path=str(args.out / shot))
                report["shots"]["stopped"] = {"file": shot, "stage": st["stageText"], "cmd": st["cmd"],
                                              "detail": page.text_content("#detail") or ""}
            report["checks"]["missing_shots"] = sorted(wanted)
            browser.close()
    finally:
        srv.terminate()
        fake.terminate()
        try:
            fake.wait(timeout=10)
        except subprocess.TimeoutExpired:
            fake.kill()
        srv.wait(timeout=10)
        log.close()
    suffix = "" if args.scenario == "replay" else f"_{args.scenario}"
    (args.out / f"capture_report_{tag}{suffix}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"viewport": tag, "scenario": args.scenario, "shots": sorted(report["shots"]),
                      "missing": report["checks"].get("missing_shots"), "errors": report["errors"][:5]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
