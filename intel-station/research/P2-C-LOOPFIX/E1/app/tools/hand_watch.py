"""Opus hand watchdog (I-58; rules shared with the sort loop since 18:43Z).

Every 0.25 s: read the newest run/live_{top,close,wrist}.jpg. While run/motion_active exists, a hand seen on enough
consecutive frames writes run/estop (the motion server holds the arm and keeps squeezing a held brick) and SIGINTs
any run_sequence.py. Every hand frame is logged to logs/monitor/hand_watch.log; the frame that stops the arm is saved
as logs/monitor/estop_<view>_<epoch>.jpg.

The detector lives in handdet.py and is the SAME code the loop's clear check (sort_loop.hands_seen) uses, so the
watchdog and the loop can never disagree about whether the workspace is clear (cause of the 18:03Z tight loop).
The wrist view is used only while the gripper is open and the jaws are > 90 mm above the mat: a white or cream
brick in or under the gripper looks skin-toned to the wrist camera.
Every pass writes the heartbeat run/hand_watch_alive (handdet.write_heartbeat); the sort loop treats a missing or
stale heartbeat (over 3 s) as not clear.

Run:  nohup envs/data-env/bin/python -u -B app/tools/hand_watch.py >> logs/monitor/hand_watch.out 2>&1 &
      echo $! > run/hand_watch.pid      (stop: kill $(cat run/hand_watch.pid))
"""
import json
import os
import subprocess
import sys
import time

import cv2

R = "/home/intel/intel-junction-2026"
sys.path.insert(0, f"{R}/app/tools")
import handdet  # noqa: E402

try:
    import kin  # noqa: E402
except Exception:  # noqa: BLE001  (no kinematics -> the wrist view is never used)
    kin = None

STATE = f"{R}/run/mserver/state.json"
WRIST_MIN_HEIGHT_MM = 90.0


def _state():
    try:
        return json.load(open(STATE))
    except Exception:  # noqa: BLE001
        return None


def holding(st) -> bool:
    """Gripper commanded closed: a brick may be between the jaws."""
    return bool(st) and st["target"][5] < 20


def wrist_usable(st) -> bool:
    """Wrist view only with the gripper open and the jaws high above the mat (transit / scan)."""
    if not st or kin is None or holding(st):
        return False
    try:
        p, _ = kin.fk(st["present"][:5])
    except Exception:  # noqa: BLE001
        return False
    return p[2] > kin.MAT_Z + WRIST_MIN_HEIGHT_MM


def main():
    log = open(f"{R}/logs/monitor/hand_watch.log", "a")
    last, hits = {}, {}
    while True:
        handdet.write_heartbeat()
        active = os.path.exists(f"{R}/run/motion_active")
        st = _state()
        hold = holding(st)
        views = ("top", "close", "wrist") if wrist_usable(st) else ("top", "close")
        for view in views:
            p = f"{R}/run/live_{view}.jpg"
            try:
                mt = os.path.getmtime(p)
            except OSError:
                continue
            if last.get(view) == mt:
                continue
            last[view] = mt
            img = cv2.imread(p)
            if img is None:  # frame still being written
                continue
            hand, f = handdet.is_hand(img, view, hold)
            hits[view] = hits.get(view, 0) + 1 if hand else 0
            if not hand:
                continue
            msg = (f"{time.strftime('%H:%M:%SZ', time.gmtime())} HAND? view={view} blob={f['area']}px "
                   f"edge={f['edge'] or '-'} hsv={f['hue']}/{f['sat']}/{f['val']} active={active} hits={hits[view]}")
            log.write(msg + "\n")
            log.flush()
            if active and hits[view] >= handdet.PERSIST[view]:
                open(f"{R}/run/estop", "w").write(msg + "\n")
                subprocess.run(["pkill", "-INT", "-f", "run_sequence.py"], check=False)
                cv2.imwrite(f"{R}/logs/monitor/estop_{view}_{int(time.time())}.jpg", img)
        time.sleep(0.25)


if __name__ == "__main__":
    main()
