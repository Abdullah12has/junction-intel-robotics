#!/usr/bin/env python3
"""Opus I-59 servo step: move to an absolute joint target (slow, watchdog armed), then measure with depth.

  servo_step.py NAME pan sl el wf roll grip [--speed 5] [--no-move] [--jaw-box ..] [--split-x ..] [--brick-box ..]

1. adds pose+sequence NAME to app/motion/gm/servo.yaml (JSON) and runs run_sequence with run/motion_active set
2. grabs close-camera depth (424x240, 30 frames) and runs rs_jaws (tips/brick in mat frame)
3. copies the live top/close/wrist colour frames next to the depth
Everything lands in logs/depth/servo/<HHMMSS>_<NAME>/ and one summary line is appended to logs/depth/servo/steps.tsv
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

R = "/home/intel/intel-junction-2026"
PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D040939-if00"
PY_DATA = f"{R}/envs/data-env/bin/python"
PY_MOT = f"{R}/envs/motion-env/bin/python"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("q", nargs=6, type=float)
    ap.add_argument("--speed", type=float, default=5)
    ap.add_argument("--no-move", action="store_true")
    ap.add_argument("--jaw-box", default="140,0,270,135")
    ap.add_argument("--split-x", default="215")
    ap.add_argument("--brick-box", default="190,95,265,165")
    ap.add_argument("--jaw-min-h", default="24")
    a = ap.parse_args()
    out = f"{R}/logs/depth/servo/{time.strftime('%H%M%S', time.gmtime())}_{a.name}"
    os.makedirs(out, exist_ok=True)
    line = {"t": time.strftime("%H:%M:%SZ", time.gmtime()), "name": a.name, "q": a.q}
    if not a.no_move and os.path.exists(f"{R}/run/mserver/pid"):
        if os.path.exists(f"{R}/run/estop"):
            sys.exit("run/estop present - refusing to move")
        r = subprocess.run([PY_DATA, "-B", "mcmd.py", json.dumps({"op": "move", "q": a.q, "speed": a.speed, "comp": True})],
                           cwd=f"{R}/app/tools", capture_output=True, text=True, timeout=90)
        line["move"] = r.stdout.strip()[-400:]
        print("MOVE", line["move"])
        time.sleep(0.5)
    elif not a.no_move:
        if os.path.exists(f"{R}/run/estop"):
            sys.exit("run/estop present - refusing to move")
        p = f"{R}/app/motion/gm/servo.yaml"
        d = json.load(open(p))
        d["poses"][a.name] = a.q
        d["sequences"][a.name] = [{"pose": a.name, "speed": a.speed}]
        json.dump(d, open(p, "w"), indent=1)
        open(f"{R}/run/motion_active", "w").close()
        try:
            r = subprocess.run([PY_MOT, "run_sequence.py", "--poses", "gm/servo.yaml", "--sequence", a.name,
                                "--studio-export", "project_robots.json", "--robot", "SO101 Follower", "--port", PORT,
                                "--max-speed", str(a.speed), "--tolerance", "6", "--confirm-motion"],
                               cwd=f"{R}/app/motion", capture_output=True, text=True, timeout=60)
        finally:
            os.remove(f"{R}/run/motion_active")
        steps = [l for l in r.stdout.splitlines() if l.startswith("step") or "stall" in l or "rror" in l]
        line["move"] = steps[-1] if steps else (r.stdout + r.stderr)[-300:]
        print("MOVE", line["move"])
        if os.path.exists(f"{R}/run/estop"):
            line["estop"] = True
            print("ESTOP present after move")
        time.sleep(0.8)
    for cam in ("top", "close", "wrist"):
        try:
            shutil.copy(f"{R}/run/live_{cam}.jpg", f"{out}/live_{cam}.jpg")
        except OSError:
            pass
    g = subprocess.run([PY_DATA, "-B", "rs_grab.py", "--cam", "close", "--out", out, "--mode", "424x240@15", "--frames", "24",
                        "--skip", "12"], cwd=f"{R}/app/tools", capture_output=True, text=True, timeout=60)
    if g.returncode != 0:
        print("GRAB FAIL", g.stderr[-300:])
    j = subprocess.run([PY_DATA, "-B", "rs_jaws.py", "--dir", out, "--jaw-box", a.jaw_box, "--split-x", a.split_x,
                        "--brick-box", a.brick_box, "--jaw-min-h", a.jaw_min_h], cwd=f"{R}/app/tools",
                       capture_output=True, text=True, timeout=60)
    subprocess.run([PY_DATA, "-B", "rs_height.py", "--dir", out, "--cam", "close"], cwd=f"{R}/app/tools",
                   capture_output=True, text=True, timeout=60)
    try:
        res = json.load(open(f"{out}/close_jaws.json"))
        lj, rj = res.get("left_jaw") or {}, res.get("right_jaw") or {}
        line.update({"brick_top": res["brick"]["top_h_p75"], "L_tip": lj.get("tip_mat"), "L_hmin": lj.get("tip_h_min"),
                     "R_tip": rj.get("tip_mat"), "R_hmin": rj.get("tip_h_min"), "dir": out})
    except Exception as e:
        line["jaws_err"] = str(e) + j.stderr[-200:]
    print(json.dumps(line))
    with open(f"{R}/logs/depth/servo/steps.tsv", "a") as f:
        f.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()
