"""Apply B3 to sort_any.py: the jaw check retries when a stop interrupts it (instead of keeping a stale servo target);
a moved wrist camera (servo target changed > 6 px) archives the scan pairs, which were measured with the old camera
pose, so the scan map re-learns from new servo-verified pairs; every new pair records the servo target in use.
Also archives the current pairs now (the 21:30:56Z re-fit moved the target 403.1 -> 376.5) keeping only pairs
recorded after that re-fit. Exact anchors."""
import json
import os
import py_compile
import shutil
import sys
import time

R = "/home/intel/intel-junction-2026"
S = f"{R}/app/tools/sort_any.py"
s = open(S).read()
orig = s


def once(old, new):
    global s
    n = s.count(old)
    if n != 1:
        sys.exit(f"ABORT: anchor found {n} times: {old[:90]!r}")
    s = s.replace(old, new)


once('''    try:
        rc, d = run_jaw_check(move=True)
    except Exception as e:  # noqa: BLE001
        log(f"jaw check ({where}) could not run ({e}); target stays {ga.TARGET_PX}")
        return True, {}
    if rc == 3:''', '''    rc, d = None, {}
    for attempt in range(3):  # a stop (hand in view) during the two rolls makes the check inconclusive: wait, retry
        try:
            rc, d = run_jaw_check(move=True)
        except Exception as e:  # noqa: BLE001
            rc, d = None, {"error": repr(e)}
        if rc in (0, 3):
            break
        log(f"jaw check ({where}) inconclusive ({str(d.get('error'))[-160:]}); retry {attempt + 1}/3 after a clear workspace")
        sl.wait_clear("jaw check")
    if rc == 3:''')
once('''        old = ga.TARGET_PX
        apply_jaw_calibration()
        log(f"jaw check ({where}): tips {d.get('jaw_tips')} gap change {d.get('gap_change_px')} px, common shift "
            f"{d.get('common_shift_px')} px, servo target {old} -> {ga.TARGET_PX}")
''', '''        old = ga.TARGET_PX
        apply_jaw_calibration()
        log(f"jaw check ({where}): tips {d.get('jaw_tips')} gap change {d.get('gap_change_px')} px, common shift "
            f"{d.get('common_shift_px')} px, servo target {old} -> {ga.TARGET_PX}")
        if os.path.exists(SCAN_PAIRS) and max(abs(ga.TARGET_PX[0] - old[0]), abs(ga.TARGET_PX[1] - old[1])) > 6:
            dst = SCAN_PAIRS.replace(".jsonl", time.strftime("_before_%Y%m%dT%H%M%SZ.jsonl", time.gmtime()))
            os.replace(SCAN_PAIRS, dst)
            log(f"wrist camera moved: scan pairs from the old camera pose archived ({os.path.basename(dst)}); "
                "the scan map re-learns from new servo-verified pairs")
''')
once('''        f.write(json.dumps({"px": [float(px[0]), float(px[1])], "xy": [round(float(ph[0]), 1), round(float(ph[1]), 1)],
                            "t": time.strftime("%H:%M:%SZ", time.gmtime()), "run": os.path.basename(RUN),
                            "ep": EP.get("episode")}) + "\\n")''', '''        f.write(json.dumps({"px": [float(px[0]), float(px[1])], "xy": [round(float(ph[0]), 1), round(float(ph[1]), 1)],
                            "t": time.strftime("%H:%M:%SZ", time.gmtime()), "run": os.path.basename(RUN),
                            "ep": EP.get("episode"), "target_px": [float(v) for v in ga.TARGET_PX]}) + "\\n")''')

bak = f"{R}/run/bak_sort_any_{time.strftime('%H%M', time.gmtime())}c.py"
shutil.copy(S, bak)
tmp = S + ".tmp"
open(tmp, "w").write(s)
py_compile.compile(tmp, doraise=True)
os.replace(tmp, S)
print("patched", S, "backup", bak)

# one-off: archive pairs measured before the 21:30:56Z camera re-fit, keep the ones after it
P = f"{R}/app/tools/scan_pairs.jsonl"
rows = [json.loads(x) for x in open(P) if x.strip()]
keep = [r for r in rows if r.get("run") == "any_2127" and r.get("t", "") >= "21:30:56Z"]
arch = P.replace(".jsonl", "_before_20261114T213056Z.jsonl")
shutil.copy(P, arch)
tmp = P + ".tmp"
with open(tmp, "w") as f:
    for r in keep:
        r["target_px"] = [376.5, 266.5]
        f.write(json.dumps(r) + "\n")
os.replace(tmp, P)
print(f"scan pairs: {len(rows)} archived to {os.path.basename(arch)}, kept {len(keep)}")
