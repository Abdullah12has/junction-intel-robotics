"""Apply B5 to jaw_check.py: three wrist rolls (0, +25, -25, clamped to +-95) and the median of the three pairwise
jaw measurements (one contaminated frame no longer decides); the gripper position is recorded; tolerance bands from
the measured repeatability (21:30-21:39Z at the scan pose: gap change 8.8, 18.0, 11.1 px): <= 15 ok, 15-25 drift
warning (target still re-fitted, loop continues), > 25 jaw loose or bent (exit 3). Exact anchors."""
import os
import py_compile
import shutil
import sys
import time

R = "/home/intel/intel-junction-2026"
P = f"{R}/app/tools/jaw_check.py"
s = open(P).read()


def once(old, new):
    global s
    n = s.count(old)
    if n != 1:
        sys.exit(f"ABORT: anchor found {n} times: {old[:90]!r}")
    s = s.replace(old, new)


once('''    ap.add_argument("--tol", type=float, default=15.0, help="px change of the tip-to-tip vector")
''', '''    ap.add_argument("--tol", type=float, default=15.0, help="px change of the tip-to-tip vector still fully ok")
    ap.add_argument("--fail", type=float, default=25.0, help="px change that means a loose or bent jaw")
''')
once('''    if not a.no_move:
        ga.move(q, speed=8, comp=True)
        for dr in (0.0, 30.0):
            qq = list(q)
            qq[4] = round(q[4] + dr, 2)
            ga.move(qq, speed=15, comp=False if dr else True)
            time.sleep(0.4)
            frames.append(ga.fresh("wrist", time.time() + 0.2))
        ga.move(q, speed=15, comp=True)
    else:
        frames.append(ga.fresh("wrist", time.time()))
    m = frames[0] if len(frames) == 1 else None
    r = redmask(frames[0])
    for f in frames[1:]:
        r = r & redmask(f)
    r = cv2.morphologyEx(r, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    keep, tips = jaws(r, frames[0].shape[0])
    out = {"t": time.strftime("%H:%M:%SZ", time.gmtime()), "jaw_tips": tips, "ref": REF}
''', '''    if not a.no_move:
        ga.move(q, speed=8, comp=True)
        for dr in (0.0, 25.0, -25.0):
            qq = list(q)
            qq[4] = round(max(-95.0, min(95.0, q[4] + dr)), 2)
            ga.move(qq, speed=15, comp=False if dr else True)
            time.sleep(0.4)
            frames.append(ga.fresh("wrist", time.time() + 0.2))
        ga.move(q, speed=15, comp=True)
    else:
        frames.append(ga.fresh("wrist", time.time()))
    grip = round(float(ga.state()["present"][5]), 2)
    pairs = [(0, 1), (0, 2), (1, 2)] if len(frames) == 3 else [(0, 0)]
    meas = []
    for i, j in pairs:  # every pair of rolls gives one measurement; the median ignores one contaminated frame
        r = cv2.morphologyEx(redmask(frames[i]) & redmask(frames[j]), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        k_, t_ = jaws(r, frames[0].shape[0])
        if len(t_) == 2:
            meas.append((k_, t_))
    if meas:
        med = [[float(np.median([m[1][i][0] for m in meas])), float(np.median([m[1][i][1] for m in meas]))] for i in range(2)]
        keep = meas[int(np.argmin([abs(m[1][0][0] - med[0][0]) + abs(m[1][1][0] - med[1][0]) for m in meas]))][0]
        tips = [[round(med[i][0], 1), int(round(med[i][1]))] for i in range(2)]
    else:
        keep, tips = None, []
    out = {"t": time.strftime("%H:%M:%SZ", time.gmtime()), "jaw_tips": tips, "ref": REF, "gripper": grip,
           "n_measurements": len(meas), "spread_px": round(float(max(abs(m[1][0][0] - tips[0][0]) + abs(m[1][1][0] - tips[1][0])
                                                                   for m in meas)), 1) if meas else None}
''')
once('''    if dg > a.tol:
        out["ok"] = False
        out["error"] = f"the jaws moved {dg:.0f}px relative to each other (fixed jaw loose or bent): tighten the fixed jaw"
        print(json.dumps(out))
        return 3
    out["ok"] = True
''', '''    if dg > a.fail:
        out["ok"] = False
        out["error"] = f"the jaws moved {dg:.0f}px relative to each other (fixed jaw loose or bent): tighten the fixed jaw"
        print(json.dumps(out))
        return 3
    if dg > a.tol:
        out["warning"] = f"jaw gap drifting ({dg:.0f}px from the tight reference): tighten the fixed jaw at the next stop"
    out["ok"] = True
''')
bak = f"{R}/run/bak_jaw_check_{time.strftime('%H%M', time.gmtime())}b.py"
shutil.copy(P, bak)
tmp = P + ".tmp"
open(tmp, "w").write(s)
py_compile.compile(tmp, doraise=True)
os.replace(tmp, P)
print("patched", P, "backup", bak)

S = f"{R}/app/tools/sort_any.py"
t = open(S).read()
old = '''        log(f"jaw check ({where}): tips {d.get('jaw_tips')} gap change {d.get('gap_change_px')} px, common shift "
            f"{d.get('common_shift_px')} px, servo target {old} -> {ga.TARGET_PX}")
'''
new = '''        log(f"jaw check ({where}): tips {d.get('jaw_tips')} (n {d.get('n_measurements')}, spread {d.get('spread_px')} px, "
            f"gripper {d.get('gripper')}) gap change {d.get('gap_change_px')} px, common shift {d.get('common_shift_px')} px, "
            f"servo target {old} -> {ga.TARGET_PX}" + (f"; WARNING {d['warning']}" if d.get("warning") else ""))
'''
if t.count(old) != 1:
    sys.exit("ABORT: sort_any log anchor")
t = t.replace(old, new)
bak2 = f"{R}/run/bak_sort_any_{time.strftime('%H%M', time.gmtime())}e.py"
shutil.copy(S, bak2)
open(S + ".tmp", "w").write(t)
py_compile.compile(S + ".tmp", doraise=True)
os.replace(S + ".tmp", S)
print("patched", S, "backup", bak2)
