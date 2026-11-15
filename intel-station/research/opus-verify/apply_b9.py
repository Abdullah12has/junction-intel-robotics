"""Apply B9:
1. pick_any.servo learns how the wrist image really moves for a commanded step (dp ~= C @ M @ d, M = the calibrated
   Jacobian from 17:28Z, C a 2x2 correction fitted by ridge least squares on the last 40 informative steps, kept in
   run/servo_corr.json). After the wrist camera was knocked the calibrated M points the steps in a slightly wrong
   direction: the servo oscillated (+-15 px for 6 iterations, any_2127 ep 11) and stopped short of tolerance.
2. sort_any scan map: the homography is used only inside the area its pairs cover (+40 px); elsewhere the similarity
   model (leave-one-out: pairs near the bottom of the scan image were 40-54 mm off with the homography, which
   extrapolates; the bricks there failed to be found at hover, any_2127 ep 9/10/11).
Exact anchors; adds tests."""
import os
import py_compile
import re
import shutil
import sys
import time

R = "/home/intel/intel-junction-2026"
T = f"{R}/app/tools"


def patch(path, edits, tag):
    s = open(path).read()
    for old, new in edits:
        n = s.count(old)
        if n != 1:
            sys.exit(f"ABORT {os.path.basename(path)}: anchor found {n} times: {old[:90]!r}")
        s = s.replace(old, new)
    bak = f"{R}/run/bak_{os.path.basename(path)[:-3]}_{time.strftime('%H%M', time.gmtime())}{tag}.py"
    shutil.copy(path, bak)
    open(path + ".tmp", "w").write(s)
    py_compile.compile(path + ".tmp", doraise=True)
    os.replace(path + ".tmp", path)
    print("patched", path, "backup", bak)


patch(f"{T}/pick_any.py", [
    ('''def servo(q, g, start_px, tol=7.0, max_it=6, log=print, want=None):
    """Bring the tracked brick (colour class `want`) to TARGET at the current height. Returns (q, blob, converged)."""
    px = np.array(start_px, float)
    b = None
    for it in range(max_it):
        bl = blobs(frame(), min_area=SERVO_MIN_AREA)
        b = find(bl, px, 140, want)
        if b is None:
            return q, None, False
        e = target() - np.array(b["centre"])
        log(f"servo it{it}: {b.get('colour')} brick {b['centre']} err {e.round(1).tolist()} short {b['short']}")
        if np.linalg.norm(e) <= tol:
            return q, b, True
        d = 0.65 * np.linalg.solve(m_base(q), e)  # damped: the 17:32Z servo overshot at gain 1
        n = np.linalg.norm(d)
        if n > 30:
            d *= 30 / n
        qn = move_rel(q, g, [d[0], d[1], 0.0], speed=SERVO_SPEED)
        if qn is None:
            qn = move_rel(q, g, [d[0] / 2, d[1] / 2, 0.0], speed=SERVO_SPEED)
            if qn is None:
                return q, b, False
        q = qn
        px = np.array(b["centre"], float) + 0.65 * e  # expected position after the damped step (tracking continuity)
''', '''SERVO_CORR_FILE = os.path.join(ga.RUN_ROOT, "servo_corr.json")
SERVO_CORR = {"C": np.eye(2), "samples": [], "loaded": False}


def servo_corr():
    """2x2 correction C of the calibrated wrist Jacobian (image motion seen = C @ M @ step), learned online."""
    if not SERVO_CORR["loaded"]:
        SERVO_CORR["loaded"] = True
        try:
            d = json.load(open(SERVO_CORR_FILE))
            SERVO_CORR["C"] = np.array(d["C"], float)
            SERVO_CORR["samples"] = [(np.array(u, float), np.array(p, float)) for u, p in d["samples"]][-40:]
        except (OSError, ValueError, KeyError):
            pass
    return SERVO_CORR["C"]


def learn_servo_corr(u, dp, ridge=200.0):
    """u: image motion the calibrated Jacobian predicts for the step taken; dp: image motion seen. Keeps the last 40
    steps that moved the brick at least 8 px and agree with the current model within 40 px (a bigger miss is a
    tracking jump, not the camera), and refits C by ridge least squares pulled toward the identity."""
    C = servo_corr()
    if np.linalg.norm(u) < 8 or np.linalg.norm(dp - C @ u) > 40:
        return C
    SERVO_CORR["samples"] = (SERVO_CORR["samples"] + [(np.asarray(u, float), np.asarray(dp, float))])[-40:]
    U = np.array([s[0] for s in SERVO_CORR["samples"]])
    D = np.array([s[1] for s in SERVO_CORR["samples"]])
    C = np.linalg.solve(U.T @ U + ridge * np.eye(2), U.T @ D + ridge * np.eye(2)).T
    SERVO_CORR["C"] = C
    try:
        tmp = SERVO_CORR_FILE + ".tmp"
        json.dump({"C": C.tolist(), "samples": [[a.tolist(), b.tolist()] for a, b in SERVO_CORR["samples"]],
                   "t": time.strftime("%H:%M:%SZ", time.gmtime())}, open(tmp, "w"))
        os.replace(tmp, SERVO_CORR_FILE)
    except OSError:
        pass
    return C


def servo(q, g, start_px, tol=7.0, max_it=6, log=print, want=None):
    """Bring the tracked brick (colour class `want`) to TARGET at the current height. Returns (q, blob, converged).
    Each step also teaches the Jacobian correction (learn_servo_corr), so a knocked camera is followed."""
    px = np.array(start_px, float)
    b = None
    prev = None  # (predicted image motion of the last step, brick centre before it)
    for it in range(max_it):
        bl = blobs(frame(), min_area=SERVO_MIN_AREA)
        b = find(bl, px, 140, want)
        if b is None:
            return q, None, False
        if prev is not None:
            learn_servo_corr(prev[0], np.array(b["centre"], float) - prev[1])
        e = target() - np.array(b["centre"])
        log(f"servo it{it}: {b.get('colour')} brick {b['centre']} err {e.round(1).tolist()} short {b['short']}")
        if np.linalg.norm(e) <= tol:
            return q, b, True
        M = m_base(q)
        d = 0.65 * np.linalg.solve(servo_corr() @ M, e)  # damped: the 17:32Z servo overshot at gain 1
        n = np.linalg.norm(d)
        if n > 30:
            d *= 30 / n
        qn = move_rel(q, g, [d[0], d[1], 0.0], speed=SERVO_SPEED)
        if qn is None:
            d = d / 2
            qn = move_rel(q, g, [d[0], d[1], 0.0], speed=SERVO_SPEED)
            if qn is None:
                return q, b, False
        q = qn
        prev = (M @ d, np.array(b["centre"], float))
        px = np.array(b["centre"], float) + servo_corr() @ M @ d  # expected position after the step
'''),
], "c")

S = f"{T}/sort_any.py"
patch(S, [
    ('''SCAN_H = None
''', '''SCAN_H = None
SCAN_HULL = None  # scan-image area covered by the pairs the homography was fitted on
'''),
    ('''    global SCAN_C, SCAN_H
''', '''    global SCAN_C, SCAN_H, SCAN_HULL
'''),
    ('''    SCAN_H = None
    if len(PX) >= 15:''', '''    SCAN_H, SCAN_HULL = None, None
    if len(PX) >= 15:'''),
    ('''            if float(np.median(res)) <= 10:
                SCAN_H = H
                log(f"scan map: homography from {int(inl.sum())}/{len(PX)} pairs, median residual {float(np.median(res)):.1f} mm")
                return
''', '''            if float(np.median(res)) <= 10:
                SCAN_H = H
                SCAN_HULL = cv2.convexHull(np.array(PX, np.float32))
                log(f"scan map: homography from {int(inl.sum())}/{len(PX)} pairs, median residual {float(np.median(res)):.1f} mm "
                    "(used inside the area its pairs cover; similarity elsewhere)")
'''),
    ('''    if SCAN_H is not None:
        p, _ = kin.fk(q_scan)
        xy = cv2.perspectiveTransform(np.array([[b["centre"]]], np.float32), SCAN_H)[0, 0]
        return np.array([xy[0] - p[0], xy[1] - p[1]])
    return SCAN_C @ jac_xy_from_scan(q_scan, b["centre"], c)''', '''    if SCAN_H is not None and SCAN_HULL is not None and cv2.pointPolygonTest(
            SCAN_HULL, (float(b["centre"][0]), float(b["centre"][1])), True) >= -40.0:
        p, _ = kin.fk(q_scan)
        xy = cv2.perspectiveTransform(np.array([[b["centre"]]], np.float32), SCAN_H)[0, 0]
        return np.array([xy[0] - p[0], xy[1] - p[1]])
    return SCAN_C @ jac_xy_from_scan(q_scan, b["centre"], c)  # outside the pairs: no perspective extrapolation'''),
], "h")

src = open(f"{T}/pick_any.py").read()
if "\nimport json\n" not in src:
    src = src.replace("\nimport math\n", "\nimport json\nimport math\n", 1)
    open(f"{T}/pick_any.py", "w").write(src)
    py_compile.compile(f"{T}/pick_any.py", doraise=True)
    print("pick_any: import json added")

TEST = f"{T}/tests/test_servo_corr.py"
open(TEST, "w").write('''"""Servo Jacobian correction learned from steps: converges to a rotated camera, ignores tracking jumps."""
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pick_any as pa  # noqa: E402


def _fresh(tmp):
    pa.SERVO_CORR_FILE = os.path.join(tmp, "servo_corr.json")
    pa.SERVO_CORR.update({"C": np.eye(2), "samples": [], "loaded": True})


def test_learns_a_rotated_camera():
    a = math.radians(12.0)
    C_true = 1.05 * np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as t:
        _fresh(t)
        for _ in range(25):
            u = rng.uniform(-60, 60, 2)
            pa.learn_servo_corr(u, C_true @ u + rng.normal(0, 1.5, 2))
        assert np.abs(pa.servo_corr() - C_true).max() < 0.05, pa.servo_corr()
        assert os.path.exists(pa.SERVO_CORR_FILE)


def test_ignores_small_steps_and_jumps():
    with tempfile.TemporaryDirectory() as t:
        _fresh(t)
        pa.learn_servo_corr(np.array([3.0, 2.0]), np.array([30.0, -40.0]))      # too small to teach anything
        pa.learn_servo_corr(np.array([40.0, 0.0]), np.array([-200.0, 150.0]))   # the tracker jumped to another blob
        assert np.allclose(pa.servo_corr(), np.eye(2)) and pa.SERVO_CORR["samples"] == []
''')
rt = f"{T}/tests/run_tests.py"
t = open(rt).read()
m = re.search(r"MODULES\\s*=\\s*\\[([^\\]]*)\\]", t)
if m and "test_servo_corr" not in m.group(1):
    t = t[:m.end(1)] + ', "test_servo_corr"' + t[m.end(1):]
    open(rt, "w").write(t)
print("tests registered")
