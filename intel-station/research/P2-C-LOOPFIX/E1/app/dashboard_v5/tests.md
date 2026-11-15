# Tests (P2-B-DASH, fixer F1)

Note for v5 (P2-B-DASH2 E1): this file records the v4 run (51 passed) and its mutation check. The v5 counts are in `../tests.md` (83 passed). The mutation check has not been re-run on v5.

Run from this folder: `python3 -I -m pytest -q tests`. Last run: 51 passed in 18.59s (baseline of the mutation check, PASS).

## Files

| File | Tests | Covers |
|---|---|---|
| `tests/test_dashboard.py` | 23 (unchanged) | SSE tailing (partial lines, bad lines, run switch, replaced file), control.json (atomic, increasing ids), refusals, path safety, run-dir-only images, fake_run end to end. Pillow is imported, not skipped. |
| `tests/test_od9.py` | 7 (unchanged) | OD-9 stages, per-camera cards and fused verdict, handover/align/sort fields, control ids vs acks, fake_run in both sort modes. |
| `tests/test_audit_fixes.py` | 13 (new) | NaN/Infinity in geometry and in replies (reducer, `/api/state`, SSE); image symlink escape; static symlink escape; plate from the label without a route; stale handover/align/sort; reconnect after a dropped stream. |
| `tests/test_page.py` | 8 items (new, Playwright + Chrome) | OD-10 verdict colour and text; STOP posts `{"cmd":"stop"}` and shows the ack; stale detail never shown; detail line not clipped at 1366x768 (two routes) and 1920x1080; STOP label fits with an error shown at 1366x768; server-offline badge; page talks to 127.0.0.1 only. |

Total: 30 original + 21 new = 51 passed. The browser tests need Chrome (`DASHBOARD_CHROME`, default `/usr/bin/google-chrome`) and Playwright for Python; they fail, not skip, without them.

## Which new tests fail on the original candidate

Checked on a scratch copy of `candidates/code/dashboard` (not modified): 12 of the 21 new tests fail there, one per runtime defect (NaN stream and replies, stale detail, 1366x768 detail line clipping, STOP label clipping, server-offline badge). The other 9 pass on the original; they cover correct behaviour that had no test, and the mutation check below shows they catch regressions.

## Mutation check (kill table)

`python3 -I tools/run_mutants.py --out <file>.json` copies the code to a scratch folder, changes one behaviour, and runs the whole suite. Full results: `../mutants.json`.

Baseline (unmutated): PASS, 51 passed in 18.59s.
Killed: 20 of 20. Survived: none.

| Mutant | Behaviour (audit reference) | File | Result | Pytest | First failing tests |
|---|---|---|---|---|---|
| K1 | control id honours orchestrator acks (A2 M2) | `server.py` | KILLED | 1 failed, 50 passed in 20.22s | test_control_id_skips_ids_the_orchestrator_already_acknowledged |
| K2 | control id remembers ids it issued (A1 M2b) | `server.py` | KILLED | 1 failed, 50 passed in 20.25s | test_control_ids_strictly_increase_and_survive_a_deleted_file |
| K3 | SSE holds partial last lines (A1 M3, A2 M1, A3 M2) | `server.py` | KILLED | 2 failed, 49 passed, 1 warning in 22.87s | test_sse_partial_lines_and_bad_lines_over_http, test_tailer_holds_partial_line_until_newline |
| K4 | SSE switches reducer on run switch (A1 M4, A3 M3) | `server.py` | KILLED | 9 failed, 42 passed, 1 warning in 72.58s (0:01:12) | test_reconnect_gets_the_current_run_again, test_api_state_one_shot_matches_stream, test_fake_run_end_to_end_through_dashboard (+6 more) |
| K5 | control.json written atomically (A2 M3) | `server.py` | KILLED | 1 failed, 50 passed in 19.43s | test_control_json_is_never_seen_partial |
| K6 | run-dir image containment (A1 M1, A2 M4, A3 M4) | `server.py` | KILLED | 1 failed, 50 passed in 18.88s | test_run_dir_image_symlink_to_outside_is_refused |
| K7 | static files refuse symlink escapes (new) | `server.py` | KILLED | 1 failed, 50 passed in 19.14s | test_static_symlink_escape_is_refused |
| K8 | plate follows the label without a route: UNCERTAIN to BLUE (A1 M5a, A2 M5) | `server.py` | KILLED | 1 failed, 50 passed in 18.61s | test_plate_follows_the_label_when_no_route_is_reported |
| K9 | non-finite geometry sanitised to null (A1 F35, A2 row 35, A3 fact 36) | `server.py` | KILLED | 1 failed, 50 passed in 19.13s | test_reducer_stores_non_finite_geometry_as_null |
| K10 | JSON replies sanitised and strict (A3 fact 37) | `server.py` | KILLED | 1 failed, 50 passed in 18.95s | test_api_state_stays_strict_if_a_snapshot_carries_nan |
| K11 | sort shown only for its own cycle (A1 F48, A2 row 53) | `server.py` | KILLED | 2 failed, 49 passed in 19.08s | test_handover_align_and_sort_of_an_earlier_cycle_are_not_reported, test_detail_line_never_shows_an_earlier_cycles_sort |
| K12 | OD-10 page colour follows the plate (A1 M5b, A3 M5) | `app.js` | KILLED | 1 failed, 50 passed in 18.55s | test_verdict_colour_and_text_follow_the_plate |
| K13 | OD-10 UNCERTAIN text says (safe) (A2 M6) | `app.js` | KILLED | 2 failed, 49 passed in 48.49s | test_detail_line_never_shows_an_earlier_cycles_sort, test_verdict_colour_and_text_follow_the_plate |
| K14 | STOP button sends stop (A1 M6) | `app.js` | KILLED | 1 failed, 50 passed in 18.32s | test_stop_button_posts_stop_and_shows_the_ack |
| K15 | server offline after no snapshot for 5 s (A1 F50, A2 N-3, A3 N3) | `app.js` | KILLED | 1 failed, 50 passed in 33.33s | test_server_offline_badge_after_the_server_stops |
| K16 | detail line wraps instead of being cut at 1366x768 (A1 F46, A2 row 52) | `style.css` | KILLED | 1 failed, 50 passed in 18.96s | test_detail_line_is_not_clipped |
| K17 | SSE switches run only from no run (A1 M4) | `server.py` | KILLED | 1 failed, 50 passed, 1 warning in 23.67s | test_sse_follows_current_across_run_switch_and_no_run |
| K18 | SSE frames sanitised before encoding (new) | `server.py` | KILLED | 1 failed, 50 passed in 18.74s | test_sse_stays_strict_if_a_snapshot_carries_nan |
| K19 | handover shown only for its own cycle (A1 F48) | `server.py` | KILLED | 1 failed, 50 passed in 19.53s | test_handover_align_and_sort_of_an_earlier_cycle_are_not_reported |
| K20 | align shown only for its own cycle (A1 F48) | `server.py` | KILLED | 1 failed, 50 passed in 19.17s | test_handover_align_and_sort_of_an_earlier_cycle_are_not_reported |

Audit mutants and where they are killed: A1 M1 = K6; A1 M2a = K1; A1 M2b = K2; A1 M3 = K3; A1 M4 = K17 (and K4); A1 M5a = K8; A1 M5b = K12; A1 M6 = K14; A2 M1 = K3; A2 M2 = K1; A2 M3 = K5; A2 M4 = K6; A2 M5 = K8; A2 M6 = K13; A3 M1 = K1/K2; A3 M2 = K3; A3 M3 = K4; A3 M4 = K6; A3 M5 = K12.
