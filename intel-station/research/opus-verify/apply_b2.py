"""Apply B2: jaw check judges the jaws by the vector between the two tips (a loose or bent fixed jaw changes it, a
knocked wrist camera shifts both tips together), re-fits mask and servo target on a common shift; sort_any runs the
two-roll check at the scan pose at start-up and every 5 cycles, and loads the new mask and target. Exact anchors."""
import os
import py_compile
import shutil
import sys
import time

R = "/home/intel/intel-junction-2026"
T = f"{R}/app/tools"


def patch(path, edits):
    s = open(path).read()
    orig = s
    for old, new in edits:
        n = s.count(old)
        if n != 1:
            sys.exit(f"ABORT {os.path.basename(path)}: anchor found {n} times: {old[:90]!r}")
        s = s.replace(old, new)
    if s == orig:
        sys.exit(f"ABORT {path}: nothing changed")
    bak = f"{R}/run/bak_{os.path.basename(path)[:-3]}_{time.strftime('%H%M', time.gmtime())}.py"
    shutil.copy(path, bak)
    tmp = path + ".tmp"
    open(tmp, "w").write(s)
    py_compile.compile(tmp, doraise=True)
    os.replace(tmp, path)
    print("patched", path, "backup", bak)


patch(f"{T}/jaw_check.py", [
    ('"""Opus I-60 jaw check + static jaw mask (robust to bricks under the jaws).\n'
     'At the hover with the gripper open (25): frames at two wrist rolls; the jaws stay fixed in the wrist image while the\n'
     'background rotates, so red pixels present in BOTH frames are the jaws. Compares the jaw tips with the reference\n'
     '(16:42Z: left 235.7,320  right 544.6,329). Writes wrist_jaw_mask.png and jaw_calib.json when the jaws are within\n'
     'tolerance, else exits 3 (jaw moved: loose fixed jaw or bent).\n'
     'usage: jaw_check.py [--tol 18] [--no-move]"""',
     '"""Jaw check and static jaw mask for the wrist camera (robust to bricks under the jaws).\n'
     'At the current pose with the gripper open (25): frames at two wrist rolls; the jaws stay fixed in the wrist image\n'
     'while the background rotates, so red pixels present in BOTH frames are the jaws. The vector between the two tips is\n'
     'compared with the reference geometry: a loose or bent fixed jaw changes it (exit 3), a knocked wrist camera only\n'
     'shifts both tips together. When the jaws are rigid, writes wrist_jaw_mask.png and jaw_calib.json (servo target =\n'
     'gap centre + 3 px, y 268 + the common vertical shift).\n'
     'usage: jaw_check.py [--tol 15] [--no-move]"""'),
    ('REF = [[235.7, 320], [544.6, 329]]\n',
     'REF = [[235.7, 320], [544.6, 329]]  # tips with the fixed jaw tight (wrist camera, gripper 25)\n'),
    ('    ap.add_argument("--tol", type=float, default=18.0)\n',
     '    ap.add_argument("--tol", type=float, default=15.0, help="px change of the tip-to-tip vector")\n'),
    ('    out.update({"shift_px": [round(x, 1) for x in d], "gap_centre_x": round(gx, 1), "target_px": [round(gx + 3, 1), 268.0]})\n',
     '    gap = (tips[1][0] - tips[0][0], tips[1][1] - tips[0][1])\n'
     '    gap0 = (REF[1][0] - REF[0][0], REF[1][1] - REF[0][1])\n'
     '    dg = float(np.hypot(gap[0] - gap0[0], gap[1] - gap0[1]))\n'
     '    common = [round((tips[0][0] - REF[0][0] + tips[1][0] - REF[1][0]) / 2, 1),\n'
     '              round((tips[0][1] - REF[0][1] + tips[1][1] - REF[1][1]) / 2, 1)]\n'
     '    out.update({"shift_px": [round(x, 1) for x in d], "gap_change_px": round(dg, 1), "common_shift_px": common,\n'
     '                "gap_centre_x": round(gx, 1), "target_px": [round(gx + 3, 1), round(268.0 + common[1], 1)]})\n'),
    ('        out["ok"] = d[0] <= a.tol\n',
     '        out["ok"] = dg <= a.tol\n'),
    ('    if d[0] > a.tol:\n'
     '        out["error"] = f"fixed jaw moved {d[0]:.0f}px from the reference: check/tighten the jaw"\n'
     '        print(json.dumps(out))\n'
     '        return 3\n',
     '    if dg > a.tol:\n'
     '        out["ok"] = False\n'
     '        out["error"] = f"the jaws moved {dg:.0f}px relative to each other (fixed jaw loose or bent): tighten the fixed jaw"\n'
     '        print(json.dumps(out))\n'
     '        return 3\n'
     '    out["ok"] = True\n'),
])

S = f"{T}/sort_any.py"
s = open(S).read()
i0 = s.index("def periodic_jaw_check(k, every=5):\n")
i1 = s.index("def cycle(k):\n")
NEW_PJC = '''def run_jaw_check(move=True):
    """jaw_check.py at the current pose (the scan pose, gripper open). Returns (exit code, result dict)."""
    args = [f"{R}/envs/data-env/bin/python", "-B", "jaw_check.py"] + ([] if move else ["--no-move"])
    r = subprocess.run(args, cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True, timeout=60)
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip().startswith("{")]
    return r.returncode, (json.loads(lines[-1]) if lines else {"error": (r.stderr or r.stdout)[-300:]})


def apply_jaw_calibration():
    """Use the jaw mask and servo target that jaw_check.py has just written."""
    tools = os.path.dirname(os.path.abspath(__file__))
    calib = json.load(open(f"{tools}/jaw_calib.json"))
    ga.TARGET_PX = (float(calib["target_px"][0]), float(calib["target_px"][1]))
    jm = cv2.imread(f"{tools}/wrist_jaw_mask.png", cv2.IMREAD_GRAYSCALE)
    if jm is not None:
        ga.JAW_MASK = jm > 0


def jaw_check_now(where):
    """Two-roll jaw check at the scan pose. Rigid jaws (a knocked camera moves both tips together): new mask and servo
    target. Jaws moved relative to each other (loose or bent fixed jaw): returns False. Inconclusive: keeps the target."""
    try:
        rc, d = run_jaw_check(move=True)
    except Exception as e:  # noqa: BLE001
        log(f"jaw check ({where}) could not run ({e}); target stays {ga.TARGET_PX}")
        return True, {}
    if rc == 3:
        log(f"JAW CHECK FAILED ({where}): {d.get('error')} tips {d.get('jaw_tips')}")
        return False, d
    if rc == 0:
        old = ga.TARGET_PX
        apply_jaw_calibration()
        log(f"jaw check ({where}): tips {d.get('jaw_tips')} gap change {d.get('gap_change_px')} px, common shift "
            f"{d.get('common_shift_px')} px, servo target {old} -> {ga.TARGET_PX}")
    else:
        log(f"jaw check ({where}) inconclusive: {d.get('error')}; target stays {ga.TARGET_PX}")
    return True, d


def periodic_jaw_check(k, every=5):
    """Every `every` cycles (at the scan pose, before the scan): the start-up jaw check again. A loose or bent fixed
    jaw pauses the loop with a clear message (the fixed jaw works loose when it touches the ground)."""
    if k % every:
        return
    ok, d = jaw_check_now(f"cycle {k}")
    EP["jaw_check"] = d
    if not ok:
        open(f"{ga.RUN_ROOT}/sort_pause", "w").write(
            f"{time.strftime('%H:%M:%SZ', time.gmtime())} jaw check failed: {d.get('error')}\\n")
        log("pausing: tighten the fixed jaw, then delete run/sort_pause")


'''
s = s[:i0] + NEW_PJC + s[i1:]
old_start = ('    jc = subprocess.run(["true"], capture_output=True, text=True) if ga.state()["target"][5] < 20 else '
             'subprocess.run([f"{R}/envs/data-env/bin/python", "-B", "jaw_check.py", "--no-move"], '
             'cwd=os.path.dirname(os.path.abspath(__file__)),\n'
             '                        capture_output=True, text=True)\n'
             '    log(f"jaw check: {jc.stdout.strip()[-200:]}")\n')
if s.count(old_start) != 1:
    sys.exit("ABORT sort_any: start-up jaw check anchor not unique")
s = s.replace(old_start, "")
old_go = ('    try:\n        go_scan(False)\n    except ga.Stop:\n        if not recover():\n'
          '            return  # start-up recover() gave up: the cycle loop does not start\n    k = a.start\n')
if s.count(old_go) != 1:
    sys.exit("ABORT sort_any: go_scan anchor not unique")
s = s.replace(old_go, ('    try:\n        go_scan(False)\n    except ga.Stop:\n        if not recover():\n'
                       '            return  # start-up recover() gave up: the cycle loop does not start\n'
                       '    ok, d = jaw_check_now("start-up")  # at the scan pose with the gripper open\n'
                       '    if not ok:\n'
                       '        write_needs_operator(f"jaw check: {d.get(\'error\')}")\n'
                       '        return\n'
                       '    k = a.start\n'))
bak = f"{R}/run/bak_sort_any_{time.strftime('%H%M', time.gmtime())}b.py"
shutil.copy(S, bak)
tmp = S + ".tmp"
open(tmp, "w").write(s)
py_compile.compile(tmp, doraise=True)
os.replace(tmp, S)
print("patched", S, "backup", bak)
