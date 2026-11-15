"""Plain test runner for the E3 inspector (pytest is not installed in the env; no installs were made).
Run: envs/data-env/bin/python -I -B app/anomaly/det2_e3/test_det2_e3.py
Real-frame checks read the harvested frames and are skipped if they are absent."""
import glob
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import det2  # noqa: E402
import inspector  # noqa: E402

FRAMES = "/home/intel/intel-junction-2026/work/P2-B-DET2/E3/frames"
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


def test_shape_bands():
    cases = [("2x2 36x36", (36, 36), 1296, "2x2"), ("2x2 34x36", (34, 36), 1224, "2x2"),
             ("2x4 36x72", (36, 72), 2592, "not_2x2"), ("side-lying 36x54", (36, 54), 1944, "unknown"),
             ("partial 36x36 area 700", (36, 36), 700, "unknown"), ("oversize 36x36 area 3000", (36, 36), 3000, "unknown")]
    for name, (w, h), area, want in cases:
        b = {"rect": ((50.0, 50.0), (float(w), float(h)), 0.0), "area": area, "touch": False}
        got, _, asp, ratio = det2.shape_of(b, 1208.0)
        check(f"shape {name}", got == want, f"got {got} asp {asp:.2f} ratio {ratio:.2f}")
    b = {"rect": ((50.0, 50.0), (36.0, 36.0), 0.0), "area": 1296, "touch": True}
    check("shape touching edge -> unknown", det2.shape_of(b, 1208.0)[0] == "unknown")


def test_tape_dark_grey():
    base = np.zeros((80, 80, 3), np.uint8)
    base[:] = (20, 120, 230)                       # orange body, BGR
    tape = base.copy()
    tape[:, :20] = (90, 90, 90)                    # dark grey strip on a quarter of the silhouette
    print_ = base.copy()
    print_[30:50, 30:50] = (120, 30, 90)           # saturated purple print, like the '2' on a red brick
    full = np.ones((80, 80), bool)
    dg_tape = det2.tape_feats(tape, cv2.cvtColor(tape, cv2.COLOR_BGR2HSV), full)["dark_grey"]
    dg_plain = det2.tape_feats(base, cv2.cvtColor(base, cv2.COLOR_BGR2HSV), full)["dark_grey"]
    dg_print = det2.tape_feats(print_, cv2.cvtColor(print_, cv2.COLOR_BGR2HSV), full)["dark_grey"]
    check("tape strip detected", dg_tape > 0.2, f"dg {dg_tape}")
    check("plain brick no tape", dg_plain < 0.01, f"dg {dg_plain}")
    check("saturated print is not tape", dg_print < 0.01, f"dg {dg_print}")


def test_routing_fail_safe():
    check("route GREEN on confident defect", inspector.route([{"route": "GREEN"}]) == "GREEN")
    check("route INSPECT on uncertain", inspector.route([{"route": "INSPECT"}]) == "INSPECT")
    check("route INSPECT on confident good", inspector.route([{"route": inspector.route_of(inspector.CONF_GOOD)}]) == "INSPECT")
    check("route INSPECT when no brick", inspector.route([]) == "INSPECT")
    routes = {inspector.route_of(v) for v in (inspector.CONF_DEFECT, inspector.CONF_GOOD, inspector.UNC)}
    check("no verdict routes to BLUE", "BLUE" not in routes and routes == {"GREEN", "INSPECT"}, str(sorted(routes)))


def test_empty_mat():
    blank = np.zeros((480, 640, 3), np.uint8)
    res = inspector.scan(top_rgb=blank, close_rgb=blank)
    check("empty mat: two no_brick entries", len(res) == 2 and all("no_brick" in r["reasons"] for r in res), f"n={len(res)}")
    check("empty mat: route INSPECT", inspector.route(res) == "INSPECT")


def test_hands_refused():
    hsv = np.zeros((480, 640, 3), np.uint8)
    hsv[:] = (0, 0, 8)
    cv2.ellipse(hsv, (320, 240), (80, 60), 0, 0, 360, (10, 100, 180), -1)   # skin-like hue, saturation, value
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    res = inspector.scan(top_rgb=img)
    check("hand over the mat refuses the top view", len(res) == 1 and res[0]["reasons"] == ["hands_present"],
          f"reasons {[r['reasons'] for r in res]}")
    check("hand refusal routes INSPECT", inspector.route(res) == "INSPECT")


def test_real_frames():
    tops = sorted(glob.glob(f"{FRAMES}/*_243622061421.png"))
    closes = sorted(glob.glob(f"{FRAMES}/*_243722064316.png"))
    if not tops or not closes:
        print("SKIP real frames: none found")
        return
    n_green, n_blue, n_refused, n_rec = 0, 0, 0, 0
    for t, c in zip(tops, closes):
        res = inspector.scan(top_rgb=cv2.imread(t), close_rgb=cv2.imread(c))
        for r in res:
            n_rec += 1
            n_blue += int(r["route"] not in ("GREEN", "INSPECT"))
            n_green += int(r["route"] == "GREEN")
            n_refused += int("hands_present" in r["reasons"])
    check("real frames: no route other than GREEN/INSPECT", n_blue == 0)
    print(f"INFO real frames: {len(tops)} pairs, {n_rec} records, GREEN {n_green}, hand refusals {n_refused}")


def test_latency():
    tops = sorted(glob.glob(f"{FRAMES}/*_243622061421.png"))
    closes = sorted(glob.glob(f"{FRAMES}/*_243722064316.png"))
    if not tops:
        print("SKIP latency: no frames")
        return
    t, c = cv2.imread(tops[0]), cv2.imread(closes[0])
    inspector.scan(top_rgb=t, close_rgb=c)
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        inspector.scan(top_rgb=t, close_rgb=c)
        times.append((time.perf_counter() - t0) * 1000.0)
    print(f"INFO CPU scan (top+close pair, 640x480, data-env, 10 runs): median {statistics.median(times):.1f} ms, "
          f"max {max(times):.1f} ms")


if __name__ == "__main__":
    for fn in (test_shape_bands, test_tape_dark_grey, test_routing_fail_safe, test_empty_mat,
               test_hands_refused, test_real_frames, test_latency):
        fn()
    failed = [n for n, ok in RESULTS if not ok]
    print(f"{len(RESULTS) - len(failed)} passed, {len(failed)} failed")
    sys.exit(1 if failed else 0)
