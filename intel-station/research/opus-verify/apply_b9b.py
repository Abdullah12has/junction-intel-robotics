"""B9 part 2 (sort_any scan map hull rule + servo test registration); part 1 (pick_any servo learning) is applied."""
import os
import py_compile
import re
import shutil
import sys
import time

R = "/home/intel/intel-junction-2026"
T = f"{R}/app/tools"
S = f"{T}/sort_any.py"
s = open(S).read()
edits = [
    ('\nSCAN_H = None\n', '\nSCAN_H = None\nSCAN_HULL = None  # scan-image area covered by the pairs the homography was fitted on\n'),
    ('    global SCAN_C, SCAN_H\n', '    global SCAN_C, SCAN_H, SCAN_HULL\n'),
    ('    SCAN_H = None\n    if len(PX) >= 15:', '    SCAN_H, SCAN_HULL = None, None\n    if len(PX) >= 15:'),
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
]
for old, new in edits:
    n = s.count(old)
    if n != 1:
        sys.exit(f"ABORT: anchor found {n} times: {old[:80]!r}")
    s = s.replace(old, new)
bak = f"{R}/run/bak_sort_any_{time.strftime('%H%M', time.gmtime())}h.py"
shutil.copy(S, bak)
open(S + ".tmp", "w").write(s)
py_compile.compile(S + ".tmp", doraise=True)
os.replace(S + ".tmp", S)
print("patched", S, "backup", bak)

src = open(f"{T}/pick_any.py").read()
if "\nimport json\n" not in src:
    src = src.replace("\nimport math\n", "\nimport json\nimport math\n", 1)
    open(f"{T}/pick_any.py", "w").write(src)
    py_compile.compile(f"{T}/pick_any.py", doraise=True)
    print("pick_any: import json added")

open(f"{T}/tests/test_servo_corr.py", "w").write('''"""Servo Jacobian correction learned from steps: converges to a rotated camera, ignores tracking jumps."""
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
    saved = (pa.SERVO_CORR_FILE, dict(pa.SERVO_CORR))
    try:
        with tempfile.TemporaryDirectory() as t:
            _fresh(t)
            for _ in range(25):
                u = rng.uniform(-60, 60, 2)
                pa.learn_servo_corr(u, C_true @ u + rng.normal(0, 1.5, 2))
            assert np.abs(pa.servo_corr() - C_true).max() < 0.05, pa.servo_corr()
            assert os.path.exists(pa.SERVO_CORR_FILE)
    finally:
        pa.SERVO_CORR_FILE = saved[0]
        pa.SERVO_CORR.clear()
        pa.SERVO_CORR.update(saved[1])


def test_ignores_small_steps_and_jumps():
    saved = (pa.SERVO_CORR_FILE, dict(pa.SERVO_CORR))
    try:
        with tempfile.TemporaryDirectory() as t:
            _fresh(t)
            pa.learn_servo_corr(np.array([3.0, 2.0]), np.array([30.0, -40.0]))      # too small to teach anything
            pa.learn_servo_corr(np.array([40.0, 0.0]), np.array([-200.0, 150.0]))   # the tracker jumped to another blob
            assert np.allclose(pa.servo_corr(), np.eye(2)) and pa.SERVO_CORR["samples"] == []
    finally:
        pa.SERVO_CORR_FILE = saved[0]
        pa.SERVO_CORR.clear()
        pa.SERVO_CORR.update(saved[1])
''')
rt = f"{T}/tests/run_tests.py"
t = open(rt).read()
m = re.search(r"MODULES\s*=\s*\[([^\]]*)\]", t)
if m and "test_servo_corr" not in m.group(1):
    t = t[:m.end(1)] + ', "test_servo_corr"' + t[m.end(1):]
    open(rt, "w").write(t)
print("tests registered")
