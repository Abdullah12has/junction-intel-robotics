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


# F-01 + F-02: an interrupted move re-anchors the target at the present pose (rel/grip build on target);
# report the residual of the compensation loop. Motion server also refuses to move without a live hand watchdog.
edit("motion_server.py", 'ACTIVE = f"{R}/run/motion_active"\n',
     'ACTIVE = f"{R}/run/motion_active"\nHEARTBEAT = f"{R}/run/hand_watch_alive"  # written by hand_watch.py on every pass with live frames\n'
     'HEARTBEAT_MAX_AGE = 3.0\n')
edit("motion_server.py", "    def estop(self):\n        return os.path.exists(ESTOP) or STOP[\"flag\"]\n",
     "    def estop(self):\n        return os.path.exists(ESTOP) or STOP[\"flag\"] or not watchdog_alive()\n")
edit("motion_server.py", "class Server:\n",
     "def watchdog_alive():\n"
     "    \"\"\"True while hand_watch.py's heartbeat is at most HEARTBEAT_MAX_AGE s old: no move without a live watchdog.\"\"\"\n"
     "    try:\n"
     "        with open(HEARTBEAT) as f:\n"
     "            t = float(f.read())\n"
     "    except (OSError, ValueError):\n"
     "        return False\n"
     "    return 0.0 <= time.time() - t <= HEARTBEAT_MAX_AGE\n\n\n"
     "class Server:\n")
edit("motion_server.py", "        if self.estop():\n            self.status = \"estop\"\n            return \"refused: estop present\"\n",
     "        if self.estop():\n            self.status = \"estop\"\n"
     "            return \"refused: estop present\" if os.path.exists(ESTOP) else \"refused: hand watchdog not running\"\n")
edit("motion_server.py",
     "        self.target = q\n        if ok:\n            self.status = \"ok\"\n        return f\"comp_iters={its if comp else 0}\"\n",
     "        if ok:\n            self.target = q\n            self.status = \"ok\"\n"
     "        else:  # interrupted: re-anchor at the present pose so the next rel or grip starts from where the arm is\n"
     "            held = self.pl.current()\n"
     "            held[GRIP] = self.goal[GRIP]\n"
     "            self.target = np.asarray(held, dtype=np.float64)\n"
     "        resid = float(np.max(np.abs((q - self.pl.current())[:GRIP]))) if ok else float(\"nan\")\n"
     "        return f\"comp_iters={its if comp else 0} resid={resid:.2f}\"\n")

# F-06 + F-37: the watchdog survives errors and writes its heartbeat only while the top and close frames are live.
edit("hand_watch.py", "STATE = f\"{RUN_ROOT}/mserver/state.json\"\n",
     "STATE = f\"{RUN_ROOT}/mserver/state.json\"\nFRAME_MAX_AGE = 1.0  # s: an older top or close frame means a dead recorder, so no heartbeat\n")
edit("hand_watch.py",
     "def main():\n    log = open(f\"{LOG_ROOT}/monitor/hand_watch.log\", \"a\")\n    last, hits = {}, {}\n    while True:\n        handdet.write_heartbeat()\n        active",
     "def frames_live(now=None):\n"
     "    \"\"\"True when the top and close live frames are at most FRAME_MAX_AGE s old.\"\"\"\n"
     "    now = time.time() if now is None else now\n"
     "    for view in (\"top\", \"close\"):\n"
     "        try:\n"
     "            if now - os.path.getmtime(f\"{RUN_ROOT}/live_{view}.jpg\") > FRAME_MAX_AGE:\n"
     "                return False\n"
     "        except OSError:\n"
     "            return False\n"
     "    return True\n\n\n"
     "def main():\n    log = open(f\"{LOG_ROOT}/monitor/hand_watch.log\", \"a\")\n    last, hits = {}, {}\n    while True:\n"
     "        try:\n"
     "            watch_once(log, last, hits)\n"
     "            if frames_live():\n"
     "                handdet.write_heartbeat()\n"
     "        except Exception as e:  # noqa: BLE001  (keep watching; no heartbeat on a failed pass)\n"
     "            log.write(f\"{time.strftime('%H:%M:%SZ', time.gmtime())} WATCHDOG ERROR {e!r}\\n\")\n"
     "            log.flush()\n"
     "        time.sleep(0.25)\n\n\n"
     "def watch_once(log, last, hits):\n"
     "    if True:\n"
     "        active")
edit("hand_watch.py",
     "                cv2.imwrite(f\"{LOG_ROOT}/monitor/estop_{view}_{int(time.time())}.jpg\", img)\n        time.sleep(0.25)\n",
     "                cv2.imwrite(f\"{LOG_ROOT}/monitor/estop_{view}_{int(time.time())}.jpg\", img)\n")

# F-37 in the loop's own clear check: a stale live frame is not clear.
edit("sort_loop.py",
     "    for view in (\"top\", \"close\"):\n        img = cv2.imread(f\"{RUNTIME_DIR}/live_{view}.jpg\")\n",
     "    for view in (\"top\", \"close\"):\n"
     "        p = f\"{RUNTIME_DIR}/live_{view}.jpg\"\n"
     "        try:\n"
     "            if time.time() - os.path.getmtime(p) > FRAME_MAX_AGE:\n"
     "                return True  # a dead recorder shows an old frame forever: not clear\n"
     "        except OSError:\n"
     "            return True\n"
     "        img = cv2.imread(p)\n")
edit("sort_loop.py", "WATCHDOG_DOWN = False", "FRAME_MAX_AGE = 1.0  # s\nWATCHDOG_DOWN = False")

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
