"""Opus robot-slot tool: bring the follower home from where it holds, along a recorded collision-free path.

Reads the present pose, finds the nearest waypoint on the live-validated recorded returns
(app/motion/gm/gm_fast.yaml, blue_wp* and green_wp*, plate -> home order), moves there, then follows
the rest of that path to home at 15 u/s (tolerance 6). The gripper keeps its present value.
Refuses (holds, no motion) if the nearest waypoint is more than MAX_JUMP units away on any joint.
--dry prints the plan only. --torque-off releases torque at the end (only at home).
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/intel/intel-junction-2026/app/motion")
import so101_motion as m  # noqa: E402

R = "/home/intel/intel-junction-2026"
MAX_JUMP = 40.0
doc = json.load(open(f"{R}/app/motion/gm/gm_fast.yaml"))
paths = {}
for name in ("blue", "green"):
    keys = sorted((k for k in doc["poses"] if k.startswith(f"{name}_wp")), key=lambda k: int(k.split("wp")[1]))
    paths[name] = [np.array(doc["poses"][k][:5], dtype=float) for k in keys]

dry = "--dry" in sys.argv
port, cal = m.resolve_arm(studio_export=f"{R}/app/motion/project_robots.json", name="SO101 Follower", follower=True)
driver = m.open_follower_held(port, cal)
try:
    player = m.MotionPlayer(driver, max_speed=15.0, tolerance=6.0)
    cur = np.array(player.current(), dtype=float)
    grip = float(cur[5])
    best = min(((np.abs(wp - cur[:5]).max(), name, i) for name, wps in paths.items() for i, wp in enumerate(wps)))
    jump, name, start = best
    print(f"present {np.round(cur, 1).tolist()}; nearest {name}_wp{start} at {jump:.1f} units")
    if jump > MAX_JUMP:
        print(f"REFUSED: nearest recorded waypoint is {jump:.1f} > {MAX_JUMP} units away; holding")
    else:
        plan = paths[name][start:]
        print(f"plan: {len(plan)} waypoints {name}_wp{start}..{name}_wp{len(paths[name]) - 1}")
        if not dry:
            for wp in plan:
                res = player.move_to(np.concatenate([wp, [grip]]))
                print(f"  -> {np.round(wp, 1).tolist()} arrived={res.arrived} err={res.max_error:.2f}")
            print("at home")
            if "--torque-off" in sys.argv:
                driver.disconnect()
                d2 = m.open_passive(port, cal)
                d2.disconnect()
                print("torque off")
finally:
    try:
        driver.disconnect()
    except Exception:
        pass
