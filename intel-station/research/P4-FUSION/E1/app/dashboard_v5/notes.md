# Dashboard notes (fixer F1 of P2-B-DASH)

## Run
- Server: `python -m dashboard.server --runs <R>/logs/loop --port 8090 --host 127.0.0.1` (PYTHONPATH = app folder).
- Demo without the robot: `python3 -I tools/fake_run.py --runs <dir> --auto-next` (`--sort-mode policy`, `--error-after N`).
- Tests: `python3 -I -m pytest -q tests` (51 tests). Browser tests need Playwright for Python and Chrome at `DASHBOARD_CHROME` (default `/usr/bin/google-chrome`); a missing browser fails them, it does not skip them.
- Mutation check: `python3 -I tools/run_mutants.py --out <file>.json` (kill table in tests.md).
- Screenshots: `python3 -I tools/capture_screenshots.py --width W --height H --out screenshots --work <scratch> --scenario replay|fault|policy`.

## Decisions (for Opus to confirm)
- Pre-grasp frame (A1 F49, A2 row 55, A3 N5): not shown in the inspection panel. OD-9 asks for per-camera crops; `frame_url` is still computed in the reducer and not rendered.
- I-52 triage (CONFIDENT_DEFECT straight to GREEN, no inspection; defined in ledger/INSTRUCTIONS.md M38 and tasks/P2-B-DET2.md): NOT built. LOOP-CONTRACT sections 5 and 7 have no triage field, and the loop does not emit one. Pin the event and field in the contract first; the page then needs one chip per verdict.
- Control writes: a transient `.control-*.tmp` is created in the run dir before `os.replace` (A3 N1, A1 note). It is gone after each write and the page never reads it. Kept, because an in-place write could be read half-written.
- Removed unused code: `RunLocator.resolve_run_dir`, `el.liveEmpty`, `import re`.

## Contract asks (not code; the server reads these names today)
- `inspect_cameras`: `{cycle, cameras: [{camera, label, score, threshold, reasons, frame, crop}]}`. Without `cycle` the per-camera cards are dropped (server.py `_cameras_for`).
- `run_start.config`: `live_source`, `detector: {backend, device}`, `skills: {<S>: {backend, device}}`. Without them the page shows the defaults ("wrist camera (live.jpg)", no backend name).
- `handover`, `align`, `sort_start`/`sort_end` carry no cycle. The server takes the cycle from the latest `state` event, so every `state` event must carry the current `cycle`.

## Behaviour notes
- Counts update at CYCLE_DONE, so UNCERTAIN (safe) can read 0 for about 1.5 s while the verdict already shows it (A2 N-4). Unchanged.
- "server offline" shows when the stream is not OPEN and no snapshot arrived for 5 s. An open stream with a silent loop shows "loop silent N s"; keepalive lines are comments and do not refresh the snapshot clock.
- After a server restart the page reconnects by itself (retry 2 s) and gets the current run in full (test_reconnect_gets_the_current_run_again).
- The page's CSP forbids string evaluation, so the browser tests poll with `page.evaluate` (Playwright's `wait_for_function` with a string fails under that CSP).
- Screenshots come from the synthetic fake_run (no robot, no Studio, no camera), replayed at speed 0.8.
- Sort, handover and align are reported for the current cycle only (server `_current`); the detail line lists the sort first, then handover, then align.
