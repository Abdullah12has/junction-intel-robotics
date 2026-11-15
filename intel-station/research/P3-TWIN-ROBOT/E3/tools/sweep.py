#!/usr/bin/env python3
"""E3 sweep: gripper position gain (kp, kv scaled with kp) x contact condim, on patched copies of so101_twin_v2.xml.
Jaw collision geoms get friction 0.35 (ABS, scene value) and the sweep condim. Each variant: nominal, empty close,
and 10 seeds. Writes outputs/sweep_results.json. Run with: sim-env/bin/python -I -B sweep.py"""
import json
import os
import re
import sys
import tempfile
import time

R = "/home/intel/intel-junction-2026"
sys.path.insert(0, f"{R}/sim/twin/tests")
import test_grasp as tg  # noqa: E402

V2 = f"{R}/sim/twin/robot/so101_twin_v2.xml"
OUT = f"{R}/work/P3-TWIN-ROBOT/E3/outputs/sweep_results.json"
KP_KV_RATIO = 0.00154278912 / 0.5
KPS = [0.5, 5.0, 20.0]
CONDIMS = [3, 4, 6]
MU = 0.35
TORS, ROLL = 0.005, 0.0001


def patch(kp, condim):
    txt = open(V2).read()
    kv = kp * KP_KV_RATIO
    txt, n1 = re.subn(r'(<position name="act_gripper"[^>]*?)kp="[^"]*" kv="[^"]*"',
                      lambda m: f'{m.group(1)}kp="{kp}" kv="{kv}"', txt)
    extra = f' condim="{condim}" friction="{MU} {TORS} {ROLL}"'
    # jaw collision geoms: group 3 with contype 1 (gripper_link and moving jaw); one pass only
    txt, n2 = re.subn(r'(<geom [^>]*?group="3"[^>]*?)(/>)',
                      lambda m: m.group(1) + extra + m.group(2) if 'contype="1"' in m.group(1) else m.group(0), txt)
    assert n1 == 1, n1
    assert n2 >= 2, n2
    path = os.path.join(tempfile.mkdtemp(prefix="twin_sweep_"), "cand.xml")
    open(path, "w").write(txt)
    return path, n2


def main():
    results = []
    for kp in KPS:
        for condim in CONDIMS:
            t0 = time.time()
            path, nj = patch(kp, condim)
            tg.SRC = path
            nom = tg.run_sequence()
            emp = tg.run_sequence(brick=False)
            seeds = []
            for s in range(10):
                r = tg.run_sequence(*tg.seed_case(s))
                seeds.append({"seed": s, "held": r["held"], "lift_mm": r.get("brick_lift_mm"),
                              "reading_lift": r["reading_after_lift"], "forces": r.get("lift_contacts")})
            row = {"kp": kp, "kv": kp * KP_KV_RATIO, "condim": condim, "jaw_geoms_patched": nj,
                   "nominal_held": nom["held"], "nominal_lift_mm": nom.get("brick_lift_mm"),
                   "nominal_reading_closed": nom["reading_after_close"],
                   "nominal_reading_lift": nom["reading_after_lift"], "nominal_lift_forces": nom.get("lift_contacts"),
                   "empty_reading": emp["reading_after_close"], "seeds_held": sum(x["held"] for x in seeds),
                   "seeds": seeds, "wall_s": round(time.time() - t0, 1)}
            results.append(row)
            print(json.dumps({k: row[k] for k in ("kp", "condim", "nominal_held", "nominal_lift_mm",
                                                  "nominal_reading_lift", "empty_reading", "seeds_held",
                                                  "wall_s")}), flush=True)
            os.makedirs(os.path.dirname(OUT), exist_ok=True)
            json.dump(results, open(OUT, "w"), indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
