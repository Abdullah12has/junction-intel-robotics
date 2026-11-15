#!/usr/bin/env python3
"""Opus I-59: estimate the jaw-tip Jacobian at a base pose (precise motion server + close depth + wrist cam).
usage: jprobe.py '[pan,sl,el,wf,roll,grip]' [--d 3] [--joints sl,el,wf,pan,roll]
Moves base, base + d on each joint (back to base between probes) and prints mm per unit for the L and R tips."""
import argparse
import json
import os
import shutil
import subprocess
import time

R = "/home/intel/intel-junction-2026"
PY = f"{R}/envs/data-env/bin/python"
T = f"{R}/app/tools"
IDX = {"pan": 0, "sl": 1, "el": 2, "wf": 3, "roll": 4}


def mv(q, speed=5):
    r = subprocess.run([PY, "-B", "mcmd.py", json.dumps({"op": "move", "q": q, "speed": speed, "comp": True})],
                       cwd=T, capture_output=True, text=True, timeout=90)
    st = json.loads(r.stdout.strip().splitlines()[-1])
    if st.get("status") != "ok":
        raise SystemExit(f"move failed: {r.stdout[-300:]}")
    return st


def meas(out):
    os.makedirs(out, exist_ok=True)
    time.sleep(0.6)
    for cam in ("top", "close", "wrist"):
        try:
            shutil.copy(f"{R}/run/live_{cam}.jpg", f"{out}/live_{cam}.jpg")
        except OSError:
            pass
    subprocess.run([PY, "-B", "rs_grab.py", "--cam", "close", "--out", out, "--mode", "424x240@15", "--frames", "20",
                    "--skip", "10", "--no-ir"], cwd=T, capture_output=True, text=True, timeout=60)
    j = subprocess.run([PY, "-B", "rs_jaws.py", "--dir", out], cwd=T, capture_output=True, text=True, timeout=60)
    wb = subprocess.run([PY, "-B", "wrist_brick.py", f"{out}/live_wrist.jpg"], cwd=T, capture_output=True, text=True)
    res = json.loads(j.stdout)
    try:
        res["wrist"] = json.loads(wb.stdout)
    except Exception:
        res["wrist"] = None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--d", type=float, default=3.0)
    ap.add_argument("--joints", default="sl,el,wf,pan")
    a = ap.parse_args()
    base = json.loads(a.base)
    root = f"{R}/logs/depth/jprobe/{time.strftime('%H%M%S', time.gmtime())}"
    st = mv(base)
    res = {"base": meas(f"{root}/base")}
    res["base"]["present"] = st["present"]
    for jn in a.joints.split(","):
        q = list(base)
        q[IDX[jn]] += a.d
        st = mv(q)
        res[jn] = meas(f"{root}/{jn}")
        res[jn]["present"] = st["present"]
    mv(base)
    J = {}
    for jn in a.joints.split(","):
        J[jn] = {}
        for tip in ("left_jaw", "right_jaw"):
            b, p = res["base"].get(tip), res[jn].get(tip)
            if b and p:
                J[jn][tip] = [round((p["tip_mat"][k] - b["tip_mat"][k]) / a.d, 2) for k in range(3)]
                if b.get("ref_mat") and p.get("ref_mat"):
                    J[jn][tip + "_ref"] = [round((p["ref_mat"][k] - b["ref_mat"][k]) / a.d, 2) for k in range(3)]
        wb, wp = res["base"].get("wrist") or {}, res[jn].get("wrist") or {}
        if wb.get("brick") and wp.get("brick"):
            J[jn]["wrist_px"] = [round((wp["brick"]["centre"][0] - wb["brick"]["centre"][0]) / a.d, 1),
                                 round((wp["brick"]["centre"][1] - wb["brick"]["centre"][1]) / a.d, 1),
                                 round((wp["brick"]["edge_yaw_deg"] - wb["brick"]["edge_yaw_deg"]) / a.d, 2)]
    out = {"base_q": base, "d": a.d, "J_per_unit": J, "base": res["base"], "dir": root}
    json.dump({"J": J, "res": res, "base_q": base}, open(f"{root}/jprobe.json", "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
