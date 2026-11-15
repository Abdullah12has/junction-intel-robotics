"""Code-review safety fixes before the robot runs again (P3-CODE-REVIEW F-01, F-02, F-06, F-37, F-47, F-48)."""
import sys

T = "/home/intel/intel-junction-2026/app/tools"


def edit(name, old, new, count=1):
    p = f"{T}/{name}"
    s = open(p).read()
    n = s.count(old)
    if n != count:
        sys.exit(f"{name}: expected {count} of {old[:70]!r}, found {n}")
    open(p, "w").write(s.replace(old, new))
    print(f"{name}: ok {old[:60]!r}")


edit("sort_loop.py", "WATCHDOG_DOWN = False  # True", "FRAME_MAX_AGE = 1.0  # s: an older live frame is not evidence of a clear workspace\nWATCHDOG_DOWN = False  # True")

# F-47: the final descent never takes the jaw tips below 3 mm above the mat by FK, whatever the size estimate says.
edit("sort_any.py",
     "    EP[\"grasp_use_mm\" + (\"_retry\" if retry else \"\")] = round(use, 1)\n",
     "    use = min(use, fk_tip + tip_final - MIN_TIP_MM)  # jaws end at tip_final + fk_tip - use: never below MIN_TIP_MM\n"
     "    EP[\"grasp_use_mm\" + (\"_retry\" if retry else \"\")] = round(use, 1)\n")
edit("sort_any.py", "FAILED_XY = []", "MIN_TIP_MM = 3.0  # lowest jaw-tip height above the mat (FK) any descent may reach\nFAILED_XY = []")

# F-48: never sweep from a low pose to the scan pose: lift in steps, and stop if the lift fails.
edit("sort_any.py",
     "    if p[2] < kin.MAT_Z + 45:  # low: go straight up first\n        q = pa.move_rel(st[\"present\"][:5], g, [0, 0, kin.MAT_Z + 75 - p[2]], speed=CFG[\"hover\"])\n",
     "    if p[2] < kin.MAT_Z + 45:  # low: go straight up first; a failed lift must not turn into a low sweep\n"
     "        up = kin.MAT_Z + 75 - p[2]\n"
     "        q = pa.move_rel(st[\"present\"][:5], g, [0, 0, up], speed=CFG[\"hover\"])\n"
     "        if q is None:\n"
     "            q = st[\"present\"][:5]\n"
     "            for _ in range(int(math.ceil(up / 20.0))):\n"
     "                q = pa.move_rel(q, g, [0, 0, min(20.0, up)], speed=CFG[\"hover\"])\n"
     "                if q is None:\n"
     "                    raise ga.Stop(\"lift before the scan move failed (IK): not sweeping from a low pose\")\n")
print("done")
