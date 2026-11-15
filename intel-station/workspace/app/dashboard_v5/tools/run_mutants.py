"""Mutation check for the dashboard suite (kill table for changelog.md and tests.md).

Each mutant changes one behaviour in a scratch copy of app/, tests/ and static files, then runs the
full pytest suite there. A mutant is KILLED when the suite fails on it and SURVIVED when it passes.
The unmutated copy must pass first (BASELINE).

Usage (from the dashboard code folder):
  python3 -I tools/run_mutants.py --out ../mutants.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
SERVER = "app/dashboard/server.py"
APPJS = "app/dashboard/static/app.js"
CSS = "app/dashboard/static/style.css"

MUTANTS = [
    {"id": "K1", "behaviour": "control id honours orchestrator acks (A2 M2)", "file": SERVER,
     "old": "self._last_id.get(str(run_dir), 0), max_ack_id(run_dir))",
     "new": "self._last_id.get(str(run_dir), 0), 0)"},
    {"id": "K2", "behaviour": "control id remembers ids it issued (A1 M2b)", "file": SERVER,
     "old": "self._last_id.get(str(run_dir), 0), max_ack_id(run_dir))",
     "new": "max_ack_id(run_dir))"},
    {"id": "K3", "behaviour": "SSE holds partial last lines (A1 M3, A2 M1, A3 M2)", "file": SERVER,
     "old": '*lines, self.pending = (self.pending + chunk).split(b"\\n")',
     "new": 'lines = (self.pending + chunk).split(b"\\n")\n        self.pending = b""'},
    {"id": "K4", "behaviour": "SSE switches reducer on run switch (A1 M4, A3 M3)", "file": SERVER,
     "old": "self.reducer = RunReducer(current.name if current is not None else None)",
     "new": "self.reducer = self.reducer"},
    {"id": "K5", "behaviour": "control.json written atomically (A2 M3)", "file": SERVER,
     "old": "        os.replace(tmp_name, path)",
     "new": '        with open(path, "wb") as dst, open(tmp_name, "rb") as src:\n'
            '            dst.write(src.read())\n'
            '        os.unlink(tmp_name)'},
    {"id": "K6", "behaviour": "run-dir image containment (A1 M1, A2 M4, A3 M4)", "file": SERVER,
     "old": "if not _inside(target, run_dir) or not target.is_file()",
     "new": "if not target.is_file()"},
    {"id": "K7", "behaviour": "static files refuse symlink escapes (new)", "file": SERVER,
     "old": "if target.parent != root or not target.is_file():",
     "new": "if not target.is_file():"},
    {"id": "K8", "behaviour": "plate follows the label without a route: UNCERTAIN to BLUE (A1 M5a, A2 M5)",
     "file": SERVER,
     "old": '    return "BLUE" if label == "GOOD" else "GREEN"',
     "new": '    return "BLUE" if label in ("GOOD", "UNCERTAIN") else "GREEN"'},
    {"id": "K9", "behaviour": "non-finite geometry sanitised to null (A1 F35, A2 row 35, A3 fact 36)", "file": SERVER,
     "old": '"geometry": _json_safe(ev.get("geometry")) if isinstance(ev.get("geometry"), dict) else {},',
     "new": '"geometry": ev.get("geometry") if isinstance(ev.get("geometry"), dict) else {},'},
    {"id": "K10", "behaviour": "JSON replies sanitised and strict (A3 fact 37)", "file": SERVER,
     "old": 'body = json.dumps(_json_safe(obj), allow_nan=False) + "\\n"',
     "new": 'body = json.dumps(obj) + "\\n"'},
    {"id": "K11", "behaviour": "sort shown only for its own cycle (A1 F48, A2 row 53)", "file": SERVER,
     "old": '"sort": self._current(self.sort),',
     "new": '"sort": self.sort,'},
    {"id": "K12", "behaviour": "OD-10 page colour follows the plate (A1 M5b, A3 M5)", "file": APPJS,
     "old": "const blue = insp.plate === 'BLUE';",
     "new": "const blue = insp.plate === 'GREEN';"},
    {"id": "K13", "behaviour": "OD-10 UNCERTAIN text says (safe) (A2 M6)", "file": APPJS,
     "old": "el.verdictSafe.textContent = insp.safe ? ' (safe)' : '';",
     "new": "el.verdictSafe.textContent = '';"},
    {"id": "K14", "behaviour": "STOP button sends stop (A1 M6)", "file": APPJS,
     "old": "el.btnStop.addEventListener('click', () => sendCommand('stop'));",
     "new": "el.btnStop.addEventListener('click', () => sendCommand('next'));"},
    {"id": "K15", "behaviour": "server offline after no snapshot for 5 s (A1 F50, A2 N-3, A3 N3)", "file": APPJS,
     "old": "if (es.readyState !== EventSource.OPEN && quietMs > SERVER_DOWN_MS) {",
     "new": "if (es.readyState === EventSource.CLOSED) {"},
     {"id": "K16", "behaviour": "detail line wraps instead of being cut at 1366x768 (A1 F46, A2 row 52)", "file": CSS,
     "old": "  white-space: normal;\n  overflow: hidden;\n}",
     "new": "  white-space: nowrap;\n  overflow: hidden;\n}"},
    {"id": "K17", "behaviour": "SSE switches run only from no run (A1 M4)", "file": SERVER,
     "old": "        if current != self.run_dir:",
     "new": "        if self.run_dir is None:"},
    {"id": "K18", "behaviour": "SSE frames sanitised before encoding (new)", "file": SERVER,
     "old": 'data = json.dumps(_json_safe(obj), separators=(",", ":"), allow_nan=False)',
     "new": 'data = json.dumps(obj, separators=(",", ":"), allow_nan=False)'},
    {"id": "K19", "behaviour": "handover shown only for its own cycle (A1 F48)", "file": SERVER,
     "old": '"handover": self._current(self.handover),',
     "new": '"handover": self.handover,'},
    {"id": "K20", "behaviour": "align shown only for its own cycle (A1 F48)", "file": SERVER,
     "old": '"align": self._current(self.align),',
     "new": '"align": self.align,'},
]


def _apply(mutant: dict, root: Path) -> None:
    path = root / mutant["file"]
    text = path.read_text(encoding="utf-8")
    old, new = mutant["old"], mutant["new"]
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{mutant['id']}: expected one match, found {count}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def _suite(root: Path) -> dict:
    proc = subprocess.run([sys.executable, "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
                          cwd=str(root), capture_output=True, text=True, timeout=900)
    out = proc.stdout + proc.stderr
    failed = sorted(set(re.findall(r"^FAILED (\S+)", out, flags=re.MULTILINE)))
    summary = (re.findall(r"^=+ (.*) =+$|^(\d+ (?:passed|failed).*)$", out, flags=re.MULTILINE) or [("", "")])[-1]
    return {"returncode": proc.returncode, "summary": " ".join(s for s in summary if s).strip(),
            "failed": [f.split("::", 1)[-1] for f in failed]}


def run_one(mutant: dict | None) -> dict:
    scratch = Path(tempfile.mkdtemp(prefix="p2-b-dash-f1-mut-"))
    try:
        root = scratch / "code"
        shutil.copytree(CODE, root, ignore=shutil.ignore_patterns("screenshots", "__pycache__", ".pytest_cache"))
        if mutant is not None:
            _apply(mutant, root)
        result = _suite(root)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if mutant is None:
        return {"id": "BASELINE", "behaviour": "unmutated candidate copy", **result,
                "result": "PASS" if result["returncode"] == 0 else "FAIL"}
    return {"id": mutant["id"], "behaviour": mutant["behaviour"], "file": mutant["file"], **result,
            "result": "KILLED" if result["returncode"] != 0 else "SURVIVED"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args(argv)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = [run_one(None)] + list(pool.map(run_one, MUTANTS))
    survived = [r["id"] for r in rows[1:] if r["result"] == "SURVIVED"]
    report = {"baseline": rows[0]["result"], "mutants": len(MUTANTS), "killed": len(MUTANTS) - len(survived),
              "survived": survived, "rows": rows}
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for row in rows:
        print(f"{row['id']:9} {row['result']:9} {row['summary']:28} {row['behaviour']}")
    return 0 if rows[0]["result"] == "PASS" and not survived else 1


if __name__ == "__main__":
    sys.exit(main())
