# P2-C-LOOPFIX, E1: CHANGES

Patch: `loopfix.patch` (7 files under `tools/`, made with diff -u against R/app, applies with `patch -p1`).
R/app HEAD when the patch was made: bbb3d75 (the copy was taken at 8afa94e). The six target files in R/app still match the baseline hashes in `sha256_before.txt`.

## What
1. Heartbeat. `handdet.write_heartbeat` writes the epoch time to `R/run/hand_watch_alive` through a temp file and `os.replace`. `handdet.heartbeat_fresh` is true when the age is 0 to 3 s. `hand_watch.main` writes it on every pass (about 4 per second). `sort_loop.watchdog_up` gates `hands_seen`: a missing, unreadable or stale heartbeat counts as not clear and the loop waits. It logs "hand watchdog not running" once per stale period.
2. Give-up. When `recover()` fails 40 times it writes `needs_operator` (utc, episode, last failure, reason) in the loop's run folder and logs "needs operator: <reason>". `sort_any.main` does not start the cycle loop when the start-up or the carry-over `recover()` returns False.
3. Start-up gate. `sort_any.main` and `sort_loop.main` call `wait_clear("start-up")` before the first move. This goes beyond the spec's explicit list.

No threshold, perception rule, speed or motion parameter was changed.

## Decisions for Opus
- Heartbeat is in `R/run`, the shared runtime folder next to `hand_watch.pid`. I read the spec's "<run dir>" as that folder.
- `needs_operator` is in the loop's run folder, `logs/sort/<run>/`. This is my reading of "<run dir>" for the loop.
- While the watchdog is down the loop waits. The 40-attempt limit applies only to `recover()`.
- The give-up path in `sort_loop`'s own recovery branch is unchanged.
- `wait_for_brick()` in `sort_loop` keeps its hard-coded `R/run` paths.
- Restart note: the running `hand_watch.py` writes no heartbeat until it is restarted with this code, so the loop will not move until then. Restart it through the normal resume path. Nothing was restarted here.
- Stale text: `docs/PICK-ANYWHERE.md` Safety section, and `tests/README.md` Files list (it omits `test_loopfix.py`). Not edited.
- R/app has a new file `simgate/watch_sim_v2.sh` that is not in the copy. It is unrelated, and the patch does not touch it.

## How tested
- Command: `nice -n 19 R/envs/data-env/bin/python -I -B run_tests.py`, run in the copy, with TMPDIR inside the E1 folder. The first runs used /tmp, which is recorded as a deviation in the laptop log.
- Before: 11 PASS (`test_before.txt`). After: 21 PASS (`test_after.txt`, the same run as `runs/stability_1.txt`). Five consecutive runs: 21 PASS each (`runs/stability_1..5.txt`).
- Patched R/app: a copy of R/app with the patch applied gives 21 PASS (`runs/verify_patched.txt`).
- Patch check: `patch --dry-run -p1` in R/app exits 0. The applied copy matches the working copy in every tools file.
- Mutation checks (`runs/M1..M4.txt`), each on a scratch copy: M1 removes the heartbeat gate from `hands_seen`, giving 4 FAIL, including the `wait_for_bricks` test. M2 disables `write_needs_operator`, giving 1 FAIL. M3 removes the start-up return, giving 1 FAIL. M4 disables the heartbeat write, giving 1 FAIL. The control run is 21 PASS.
- Side effects: no test files appear in R/run (no hand_watch_alive, estop or needs_operator), and no t_* run folders appear in logs/sort. R/run/sort_stop was already there and was not touched.
