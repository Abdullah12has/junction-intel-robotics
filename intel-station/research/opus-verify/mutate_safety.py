"""Mutation check: undo each safety fix in a temp copy of R/app and confirm the safety tests fail."""
import os
import shutil
import subprocess
import tempfile

APP = "/home/intel/intel-junction-2026/app"
PY = "/home/intel/intel-junction-2026/envs/data-env/bin/python"
MUTANTS = {
    "F-01 no re-anchor": ("tools/motion_server.py", "            self.target = np.asarray(held, dtype=np.float64)\n", "            self.target = q\n"),
    "F-47 no clamp": ("tools/sort_any.py", "    use = min(use, fk_tip + tip_final - MIN_TIP_MM)", "    use = use"),
    "F-48 sweep": ("tools/sort_any.py", "                    raise ga.Stop(\"lift before the scan move failed (IK): not sweeping from a low pose\")\n", "                    break\n"),
    "F-37 loop stale": ("tools/sort_loop.py", "                return True  # a dead recorder shows an old frame forever: not clear\n", "                pass\n"),
    "F-37 watchdog stale": ("tools/hand_watch.py", "            if frames_live():\n                handdet.write_heartbeat()\n", "            handdet.write_heartbeat()\n"),
    "watchdog gate": ("tools/motion_server.py", " or not watchdog_alive()\n", "\n"),
}
for name, (rel, old, new) in MUTANTS.items():
    tmp = tempfile.mkdtemp(prefix="mut_")
    try:
        shutil.copytree(APP, os.path.join(tmp, "app"), ignore=shutil.ignore_patterns(".git", "__pycache__", "anom3"))
        p = os.path.join(tmp, "app", rel)
        s = open(p).read()
        assert s.count(old) == 1, (name, s.count(old))
        open(p, "w").write(s.replace(old, new))
        r = subprocess.run([PY, "-I", "-B", os.path.join(tmp, "app", "tools", "tests", "run_tests.py")],
                           capture_output=True, text=True, timeout=300)
        fails = [ln for ln in r.stdout.splitlines() if ln.startswith(("FAIL", "ERROR"))]
        print(f"{name:22s} -> {'CAUGHT' if fails else 'NOT CAUGHT'}: {[f.split()[1] for f in fails]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
