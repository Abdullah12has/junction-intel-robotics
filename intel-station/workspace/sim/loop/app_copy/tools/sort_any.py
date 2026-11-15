#!/usr/bin/env python3
"""Opus I-60 autonomous sort loop, PICK ANYWHERE (bricks at random places on the mat).

Scan pose (wrist camera high over the work area) -> choose the brick nearest the view target (not in a plate) ->
differential-IK move above it -> hover servo (wrist cam) + yaw align -> size-checked straight descent (jaws stop
8 mm above the mat) -> close -> lift check -> fixed inspection pose in front of the close camera, multi-view rolls ->
decide (OD-10, I-52) -> plate path -> back to scan. Every cycle recorded (30 Hz joints + MKVs) and logged:
R/logs/sort/<RUN>/ep_<NNNN>/. Stop: touch R/run/sort_stop. Pause: touch R/run/sort_pause.
usage: sort_any.py --run NAME [--n 150] [--speed 1.0]
"""
import argparse
import subprocess
import json
import math
import os
import sys
import time
import traceback

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "anomaly", "det2_f2"))
import grasp_auto as ga  # noqa: E402
import kin  # noqa: E402
import pick_any as pa  # noqa: E402
import sort_loop as sl  # noqa: E402  (close_verdict, decide, skin/hands helpers, run_fast)
import inspector as DET2  # noqa: E402

R = ga.R
SPOT_HOVER = [-12.62, 30.0, -19.33, 69.73, -45.65]
PLATES_XY = {"blue": (187.1, -86.1), "green": (194.6, 218.3)}
PLATE_KEEPOUT = 35.0  # 17:45Z: 85 excluded the cream brick 4 cm outside the green rim; plate_frac handles in-plate bricks
CFG = {"transit": 42.0, "hover": 18.0, "servo": 12.0, "desc": 10.0, "final": 4.0}
EP = {}
RUN = ""


def log(msg):
    line = time.strftime("%H:%M:%SZ ", time.gmtime()) + msg
    print(line, flush=True)
    with open(f"{RUN}/loop.log", "a") as f:
        f.write(line + "\n")


def scan_cal():
    c = pa.load_scan()
    if not c or c.get("q_scan") is None:
        raise SystemExit("scan_calib.json missing: run scan_calib.py with a brick under the hover")
    return c


def go_scan(holding=False):
    c = scan_cal()
    g = 0.0 if holding else 25.0
    st = ga.state()
    p, _ = kin.fk(st["present"][:5])
    if p[2] < kin.MAT_Z + 45:  # low: go straight up first
        q = pa.move_rel(st["present"][:5], g, [0, 0, kin.MAT_Z + 75 - p[2]], speed=CFG["hover"])
    tmove(c["q_scan"] + [g], CFG["transit"], comp=True)


SCAN_PAIRS = f"{ga.RUN_ROOT}/scan_pairs.jsonl"  # (scan px, servo-verified hover xy) from every centred approach
SCAN_C = np.eye(2)


def jac_xy_from_scan(q_scan, px, c):
    M = kin.m_base(kin.psi_of(q_scan)) * c["scale"]
    return np.linalg.solve(M, np.array(c["target_scan"]) - np.array(px, float))


SCAN_H = None


def fit_scan_corr():
    """Learned scan map from servo-verified approaches (scan_pairs.jsonl), refitted every cycle.
    >= 15 pairs: RANSAC homography scan px -> base xy (perspective; the 18:15Z brick near the image bottom was 48 mm
    off any linear model). Fewer: similarity correction (scale + rotation) of the scan Jacobian with one round of
    outlier rejection (> 25 mm). 17:58Z seed fit: scale 0.70, rot -9.5 deg."""
    global SCAN_C, SCAN_H
    c = scan_cal()
    p, _ = kin.fk(c["q_scan"])
    PX, XY = [], []
    if os.path.exists(SCAN_PAIRS):
        for line in open(SCAN_PAIRS):
            try:
                d = json.loads(line)
                PX.append([float(d["px"][0]), float(d["px"][1])])
                XY.append([float(d["xy"][0]), float(d["xy"][1])])
            except Exception:  # noqa: BLE001
                continue
    SCAN_H = None
    if len(PX) >= 15:
        H, inl = cv2.findHomography(np.array(PX, np.float32), np.array(XY, np.float32), cv2.RANSAC, 15.0)
        if H is not None and int(inl.sum()) >= 10:
            pr = cv2.perspectiveTransform(np.array(PX, np.float32)[None], H)[0]
            res = np.linalg.norm(pr - np.array(XY), axis=1)[inl.ravel() > 0]
            if float(np.median(res)) <= 10:
                SCAN_H = H
                log(f"scan map: homography from {int(inl.sum())}/{len(PX)} pairs, median residual {float(np.median(res)):.1f} mm")
                return
    if len(PX) < 4:
        SCAN_C = np.eye(2)
        return
    J = np.array([jac_xy_from_scan(c["q_scan"], px, c) for px in PX])
    A = np.array(XY) - p[:2]

    def sim(idx):
        M, y = [], []
        for dj, da in zip(J[idx], A[idx]):
            M += [[dj[0], -dj[1]], [dj[1], dj[0]]]
            y += [da[0], da[1]]
        (a, bb), *_ = np.linalg.lstsq(np.array(M), np.array(y), rcond=None)
        return np.array([[a, -bb], [bb, a]])

    idx = np.arange(len(J))
    C = sim(idx)
    res = np.linalg.norm(J @ C.T - A, axis=1)
    keep = idx[res <= 25.0]
    if len(keep) >= 4 and len(keep) < len(idx):
        C = sim(keep)
        res = np.linalg.norm(J @ C.T - A, axis=1)
    SCAN_C = C if float(np.median(res)) <= 15 else np.eye(2)
    log(f"scan map: similarity from {len(keep)}/{len(J)} pairs, scale {math.hypot(C[0, 0], C[1, 0]):.3f} "
        f"rot {math.degrees(math.atan2(C[1, 0], C[0, 0])):.1f} deg, median residual {float(np.median(res)):.1f} mm")


def record_scan_pair(px, q_hover):
    ph, _ = kin.fk(q_hover)
    with open(SCAN_PAIRS, "a") as f:
        f.write(json.dumps({"px": [float(px[0]), float(px[1])], "xy": [round(float(ph[0]), 1), round(float(ph[1]), 1)],
                            "t": time.strftime("%H:%M:%SZ", time.gmtime()), "run": os.path.basename(RUN),
                            "ep": EP.get("episode")}) + "\n")


def brick_xy_from_scan(q_scan, b, c):
    """Base-frame offset (mm) of a brick seen from the scan pose relative to the point under the jaws (scan Jacobian
    with the learned similarity correction)."""
    if SCAN_H is not None:
        p, _ = kin.fk(q_scan)
        xy = cv2.perspectiveTransform(np.array([[b["centre"]]], np.float32), SCAN_H)[0, 0]
        return np.array([xy[0] - p[0], xy[1] - p[1]])
    return SCAN_C @ jac_xy_from_scan(q_scan, b["centre"], c)


REACH_CACHE = {}  # (q_scan, 5 mm cell) -> bool


def grasp_reachable(q_scan, d):
    """Whether the approach and the descent can reach a brick at base offset d (mm) from the point under the jaws at
    the scan pose. Replays, without moving, the IK chain approach() and grasp() run: down to the hover height, the
    60/30/15 mm steps toward the brick, then the two descent stages. 3 of 6 approach failures in runs any_1731/1740
    were IK failures on bricks inside the old 335 mm radius rule; at hover height the reach ends near 300 mm."""
    key = (tuple(round(v, 1) for v in q_scan), round(d[0] / 5.0), round(d[1] / 5.0))
    if key in REACH_CACHE:
        return REACH_CACHE[key]
    ok = False
    q = kin.ik_rel(list(q_scan), [0.0, 0.0, -pa.SCAN_UP])
    rem = np.array(d, float)
    for _ in range(8):
        if q is None:
            break
        n = float(np.linalg.norm(rem))
        if n < 3:
            break
        for lim in (60.0, 30.0, 15.0):
            dd = rem if n <= lim else rem * lim / n
            qn = kin.ik_rel(q, [dd[0], dd[1], 0.0])
            if qn is not None:
                q, rem = qn, rem - dd
                break
        else:
            q = None
    if q is not None and float(np.linalg.norm(rem)) < 3:
        tip = float(kin.fk(q)[0][2] - kin.MAT_Z)
        q1 = kin.ik_rel(q, [0.0, 0.0, -(tip - 26.0)])
        ok = q1 is not None and kin.ik_rel(q1, [0.0, 0.0, -(26.0 - pa.TIP_FINAL)]) is not None
    REACH_CACHE[key] = ok
    return ok


def candidates(img, q_scan, c):
    out = []
    p, _ = kin.fk(q_scan)
    for b in pa.blobs(img, min_area=700, max_area=40000):
        if b["plate_frac"] >= 0.3 or (b.get("colour") in ("blue", "green") and b["plate_frac"] >= 0.15):
            continue  # inside a plate, or a plate rim fragment
        d = brick_xy_from_scan(q_scan, b, c)
        xy = (p[0] + d[0], p[1] + d[1])
        if min(math.hypot(xy[0] - v[0], xy[1] - v[1]) for v in PLATES_XY.values()) < PLATE_KEEPOUT:
            continue
        r = math.hypot(xy[0] - kin.PAN_XY_MM[0], xy[1] - kin.PAN_XY_MM[1])
        if r < 115 or r > 360 or not grasp_reachable(q_scan, d):  # at the base, or out of reach
            continue
        b["d_mm"] = [round(float(d[0]), 1), round(float(d[1]), 1)]
        b["xy"] = [round(float(xy[0]), 1), round(float(xy[1]), 1)]
        out.append(b)
    now = time.time()
    out = [b for b in out if not any(math.hypot(b["xy"][0] - x, b["xy"][1] - y) < 25 and now - t < 300 for x, y, t in FAILED_XY)]
    out.sort(key=lambda b: (b["border"], math.hypot(*b["d_mm"])))  # whole bricks first; edge bricks get re-centred
    return out


def wait_for_bricks():
    c = scan_cal()
    t0 = time.time()
    last = None
    while True:
        for f in ("sort_stop",):
            if os.path.exists(f"{ga.RUN_ROOT}/{f}"):
                raise SystemExit("stop requested")
        if os.path.exists(f"{ga.RUN_ROOT}/sort_pause") or sl.hands_seen(2):
            last = None
            time.sleep(0.4)
            continue
        cand = candidates(pa.frame(), c["q_scan"], c)
        if cand:
            b = cand[0]
            if last is not None and math.hypot(b["centre"][0] - last[0], b["centre"][1] - last[1]) < 8:
                return b, cand, round(time.time() - t0, 1)
            last = b["centre"]
        else:
            last = None
        time.sleep(0.15)


FAILED_XY = []   # (x, y, t) of bricks whose approach failed; skipped for 5 min


def approach(b, c):
    """Scan estimate -> descend to the hover height at the scan location -> differential-IK steps (<= 60 mm) at the
    hover height toward the estimate (more reach margin than at the scan height, 17:45Z IK failures) -> wrist servo
    + yaw align. Returns (q, blob) or (None, reason)."""
    g = 25.0
    q = pa.move_rel(list(c["q_scan"]), g, [0.0, 0.0, -pa.SCAN_UP], speed=CFG["hover"])
    if q is None:
        return None, "ik fail descending to hover"
    d = np.array(b["d_mm"], float)
    for step in range(8):
        n = float(np.linalg.norm(d))
        if n < 3:
            break
        moved = False
        for lim in (60.0, 30.0, 15.0):
            dd = d if n <= lim else d * lim / n
            qn = pa.move_rel(q, g, [dd[0], dd[1], 0.0], speed=CFG["hover"])
            if qn is not None:
                q, d, moved = qn, d - dd, True
                break
        if not moved:
            return None, f"ik fail moving toward brick (remaining {d.round(1).tolist()} mm)"
    want = b.get("colour")  # track this brick's colour class from scan to grasp (17:56Z neighbour switch)
    EP["target_colour"] = want
    bl = pa.blobs(pa.frame(), min_area=pa.SERVO_MIN_AREA)
    bh = pa.find(bl, pa.target(), 260, want)
    if bh is None:  # second look from 30 mm higher (wider view), centre there, then come back down
        qu = pa.move_rel(q, g, [0.0, 0.0, 30.0], speed=CFG["hover"])
        if qu is not None:
            bu = pa.find(pa.blobs(pa.frame(), min_area=1000), pa.target(), 320, want)
            if bu is not None:
                log(f"raised look: {want} brick at {bu['centre']}")
                qu, bu, _ = pa.servo(qu, g, bu["centre"], log=log, want=want, tol=15.0)
            qd = pa.move_rel(qu, g, [0.0, 0.0, -30.0], speed=CFG["hover"])
            if qd is not None:
                q = qd
        bh = pa.find(pa.blobs(pa.frame(), min_area=pa.SERVO_MIN_AREA), pa.target(), 260, want)
    if bh is None:
        cv2.imwrite(f"{ga.OUT}/crops/hover_not_found.jpg", pa.frame())
        return None, f"{want} brick not visible at hover"
    q, bh, ok = pa.servo(q, g, bh["centre"], log=log, want=want)
    if bh is None:
        return None, "servo lost brick"
    q, rot = pa.align_yaw(q, g, bh, log=log)
    if rot:
        bh = pa.find(pa.blobs(pa.frame()), pa.target(), 260, want, short_ref=bh["short"])
        if bh is None:
            return None, "brick lost after yaw"
        q, bh, ok = pa.servo(q, g, bh["centre"], log=log, want=want)
    if not ok or bh is None:
        return None, "could not centre at hover"
    try:
        record_scan_pair(b["centre"], q)
    except Exception as e:  # noqa: BLE001
        log(f"scan pair not recorded: {e}")
    return q, bh


def close_pre_verdict(img, want_hue=None):
    """DET2 close tape rule for the brick under the jaws, located from the jaw tips in the close image (works
    anywhere the close camera sees the gripper). None if the jaws are not in the close view."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = (((H <= 8) | (H >= 170)) & (S >= 100) & (V >= 50)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(red)
    tips = []
    for i in range(1, n):
        if st[i, 1] <= 3 and st[i, 4] > 800:
            ys, xs = np.nonzero(lab == i)
            yb = ys.max()
            tips.append((float(xs[ys >= yb - 6].mean()), float(yb)))
    if not tips:
        return None
    tips.sort(key=lambda t: -t[1])
    mx = float(np.mean([t[0] for t in tips[:2]]))
    my = float(max(t[1] for t in tips[:2])) + 40.0
    try:
        seg = DET2.segment_close(sl.mask_jaws(img))
    except Exception as e:  # noqa: BLE001
        return {"verdict": "UNCERTAIN", "reasons": [f"det2_error {e!r}"]}
    best = None
    for rec in seg["bricks"]:
        cx, cy = rec["centre_px"]
        d = math.hypot(cx - mx, cy - my)
        if best is None or d < best[0]:
            best = (d, rec)
    if best is None or best[0] > 130:
        return {"verdict": "UNCERTAIN", "reasons": ["no_brick_under_jaws_in_close_view"], "jaw_mid": [mx, my]}
    rec = best[1]
    verdict, reasons = DET2.triage(rec, DET2.load_thresholds())
    x, y, w, h = rec["bbox"]
    out = {"verdict": verdict, "reasons": reasons, "colour": rec.get("colour"), "tape_score": rec.get("tape_score"),
           "bbox": [int(x), int(y), int(w), int(h)], "dist": round(best[0], 1), "jaw_mid": [round(mx), round(my)],
           "hue": int(rec.get("hsv_med", [0])[0])}
    if want_hue is not None:
        dh = abs(out["hue"] - want_hue)
        dh = min(dh, 180 - dh)
        out["hue_diff"] = int(dh)
        if dh > 10:  # 17:41Z: the close view matched a different brick (taped cream) to the picked orange one
            out["verdict_raw"] = verdict
            out["verdict"] = "UNCERTAIN"
            out["reasons"] = list(reasons) + [f"close_brick_hue_mismatch_{int(dh)}"]
    return out


def grasp(q, bh, retry=False):
    g = 25.0
    tip = pa.tip_from_size(bh["short"])
    fk_tip = float(kin.fk(q)[0][2] - kin.MAT_Z)
    use = tip if abs(tip - fk_tip) <= 15 else fk_tip
    tip_final = pa.TIP_FINAL
    if retry:  # 18:00Z: a leaning upside-down brick looked big (size said 51 mm, fk 58) and the jaws closed too high
        use = min(max(tip, fk_tip), fk_tip + 6.0)
        tip_final = pa.TIP_FINAL - 2.0
    EP["grasp_use_mm" + ("_retry" if retry else "")] = round(use, 1)
    EP["hover_tip_size_mm"] = round(tip, 1)
    shape = ("not_2x2", f"aspect {bh['aspect']}") if bh["aspect"] > 1.4 else (
        ("not_2x2", f"narrow short {bh['short']}px vs {pa.SIZE_K / (use + pa.SIZE_C):.0f}") if bh["short"] < 0.72 * pa.SIZE_K / (use + pa.SIZE_C)
        else ("2x2", f"aspect {bh['aspect']} short {bh['short']}"))
    EP["shape"], EP["shape_why"] = shape
    cv2.imwrite(f"{ga.OUT}/crops/wrist_hover.png", pa.frame())
    cimg = ga.fresh("close", time.time() + 0.2)
    pre = close_pre_verdict(cimg, want_hue=bh.get("hue"))
    EP["pre_close"] = pre
    if pre is not None and pre.get("bbox"):
        cv2.imwrite(f"{ga.OUT}/crops/pre_close_full.jpg", cimg)
        sl.save_crop(cimg, pre.get("bbox"), f"{ga.OUT}/crops/pre_close.png")
    q1 = pa.move_rel(q, g, [0, 0, -(use - 26.0)], speed=CFG["desc"])
    if q1 is None:
        return False, "ik fail stage 1"
    b1 = pa.nearest(pa.blobs(pa.frame(), max_area=90000), pa.target(), max_d=260)
    ga.snap("stage1")
    q2 = pa.move_rel(q1, g, [0, 0, -(26.0 - tip_final)], speed=CFG["final"])
    if q2 is None:
        return False, "ik fail final descent"
    gimgs = ga.snap("grasp_open")
    bo = pa.nearest(pa.blobs(gimgs["wrist"], max_area=200000), pa.target(), max_d=400) if gimgs.get("wrist") is not None else None
    EP["open_short"] = bo["short"] if bo else None
    ga.cmd({"op": "grip", "value": 0.0, "speed": 20})
    time.sleep(0.5)
    gr = ga.state()["present"][5]
    EP["grip_closed"] = gr
    ga.snap("closed")
    ql = pa.move_rel(q2, 0.0, [0, 0, 45.0], speed=CFG["desc"])
    if ql is None:
        return False, "ik fail lift"
    imgs = ga.snap("lifted")
    gr2 = ga.state()["present"][5]
    held = pa.nearest(pa.blobs(imgs["wrist"], max_area=150000), pa.target(), max_d=300)
    EP.update({"grip_lifted": gr2, "lift_q": ql, "stage1_blob": b1, "lifted_blob": held})
    ratio = (held["short"] / EP["open_short"]) if (held and EP.get("open_short")) else None
    EP["held_size_ratio"] = round(ratio, 2) if ratio else None
    if gr2 < 1.1:  # 17:55Z: empty close reads 0.59, a held 2x2/2x4 reads 1.57-2.42 (size ratio was unreliable at the image edge)
        ga.cmd({"op": "grip", "value": 25.0, "speed": 20})
        return False, f"nothing held (gripper {gr2}, size ratio {ratio})"
    return True, shape


def station_inspect(k):
    """Put the held brick down at the station (the spot, centre of the close camera), retract out of view, inspect
    with the close camera (DET2 tape rule on a clean mat view) and the wrist camera from the scan pose, re-pick it.
    Returns (verdict dict, ok)."""
    c = scan_cal()
    hi = kin.ik_rel(SPOT_HOVER, [0, 0, 35.0])
    p, _ = kin.fk(ga.state()["present"][:5])
    if p[2] < kin.MAT_Z + 80:
        pa.move_rel(ga.state()["present"][:5], 0.0, [0, 0, kin.MAT_Z + 85 - p[2]], speed=CFG["desc"])
    ga.move(hi + [0.0], speed=CFG["transit"], comp=True)
    qd = pa.move_rel(hi, 0.0, [0, 0, -(35.0 + pa.HOVER_DZ - pa.TIP_FINAL - 1.0)], speed=CFG["desc"])
    ga.cmd({"op": "grip", "value": 25.0, "speed": 25})
    time.sleep(0.3)
    pa.move_rel(qd, 25.0, [0, 0, 40.0], speed=CFG["desc"])
    ga.move(c["q_scan"] + [25.0], speed=CFG["transit"], comp=True)
    time.sleep(0.4)
    imgs = ga.snap("station")
    v = sl.close_verdict(imgs["close"])
    cv2.imwrite(f"{ga.OUT}/crops/station_close_full.jpg", imgs["close"])
    sl.save_crop(imgs["close"], v.get("bbox"), f"{ga.OUT}/crops/station_close.png")
    log(f"ep {k}: station close view {v.get('colour')} {v.get('verdict')} {v.get('reasons')} tape {v.get('tape_score')}")
    cand = candidates(imgs["wrist"], c["q_scan"], c)
    if not cand:
        return v, False
    q, bh = approach(cand[0], c)
    if q is None:
        return v, False
    ok, info = grasp(q, bh)
    return v, ok


def tmove(q, speed, comp=False, accel=120.0):
    st = ga.cmd({"op": "move", "q": [float(x) for x in q], "speed": speed, "comp": comp, "accel": accel})
    if st["status"] != "ok":
        raise ga.Stop(f"tmove -> {st['status']} {st['msg']}")
    return st


WP = json.load(open(f"{R}/app/motion/gm/spot_fast.yaml"))["poses"]
PLATE_IN = {"blue": [4, 3, 2, 1, 0], "green": [4, 3, 2, 1, 0]}
PLATE_OUT = {"blue": [1, 2, 3], "green": [1, 2, 3, 4]}


def place(plate):
    """Short plate path (17:45Z): straight up, then only the plate end of the validated teleop path (wp4..wp0),
    release, back up wp1..wp3/4. FK heights: blue wp4 151 mm, green wp4 226 mm above the mat."""
    st = ga.state()
    q = st["present"][:5]
    p, _ = kin.fk(q)
    if p[2] < kin.MAT_Z + 85 and q[1] > -60:
        pa.move_rel(q, 0.0, [0, 0, kin.MAT_Z + 90 - p[2]], speed=CFG["desc"] * 2)
    for i in PLATE_IN[plate]:
        tmove(WP[f"{plate}_wp{i}_closed"], CFG["transit"])
    ga.cmd({"op": "grip", "value": 25.0, "speed": 30})
    time.sleep(0.2)
    ga.snap("released")
    for i in PLATE_OUT[plate]:
        tmove(WP[f"{plate}_wp{i}"], CFG["transit"])


def periodic_jaw_check(k, every=5):
    """At the scan pose with the gripper open: is the fixed jaw where it should be? (I-60 17:24Z: the fixed jaw works
    loose when it touches the ground.) Loose -> pause the loop with a clear message; small consistent drift of the
    jaw gap centre -> move the wrist-servo target with it."""
    if k % every:
        return
    try:
        r = subprocess.run([f"{R}/envs/data-env/bin/python", "-B", "jaw_check.py", "--no-move"], cwd=os.path.dirname(os.path.abspath(__file__)),
                           capture_output=True, text=True, timeout=40)
        line = [ln for ln in r.stdout.strip().splitlines() if ln.strip().startswith("{")]
        d = json.loads(line[-1]) if line else json.loads(r.stdout[r.stdout.index("{"):])
    except Exception as e:  # noqa: BLE001
        log(f"jaw check skipped ({e})")
        return
    EP["jaw_check"] = d
    if not d.get("ok"):
        log(f"JAW CHECK FAILED: fixed jaw moved {d.get('shift_px')} px -> pausing (tighten the fixed jaw, then delete run/sort_pause)")
        open(f"{ga.RUN_ROOT}/sort_pause", "w").write(f"{time.strftime('%H:%M:%SZ', time.gmtime())} jaw check failed {d.get('shift_px')}\n")
        return
    tx = (d.get("target_px") or [None])[0]
    if tx is not None and 5 < abs(tx - ga.TARGET_PX[0]) < 25:
        log(f"jaw gap centre moved: servo target x {ga.TARGET_PX[0]} -> {tx}")
        ga.TARGET_PX = (float(tx), float(ga.TARGET_PX[1]))
    else:
        log(f"jaw check ok: fixed-jaw shift {d.get('shift_px')} px, target {ga.TARGET_PX}")


def cycle(k):
    global EP
    EP = {"episode": k, "run": os.path.basename(RUN), "mode": "pick_anywhere"}
    ep_dir = f"{RUN}/ep_{k:04d}"
    os.makedirs(f"{ep_dir}/crops", exist_ok=True)
    ga.OUT = ep_dir
    ga.SUM.clear()
    ga.SUM["phases"] = []
    T = {}
    fit_scan_corr()
    periodic_jaw_check(k)
    b, cand, waited = wait_for_bricks()
    T["wait_s"] = waited
    c = scan_cal()
    EP.update({"target_scan_blob": b, "n_candidates": len(cand), "brick_xy_est": b.get("xy")})
    ga.snap("pre")
    log(f"ep {k}: {len(cand)} brick(s) in view; going for {b['centre']} est {b['xy']} mm (d {b['d_mm']})")
    ga.cmd({"op": "rec_start", "path": f"{ep_dir}/joints.csv"})
    EP["t_start"] = time.time()
    t0 = time.time()
    q, bh = approach(b, c)
    if q is None:
        EP.update({"success": False, "failure": bh})
        FAILED_XY.append((b["xy"][0], b["xy"][1], time.time()))
        log(f"ep {k}: APPROACH FAILED: {bh} (brick at {b['xy']} skipped for 5 min)")
        go_scan(False)
        ga.cmd({"op": "rec_stop"})
        return
    EP["hover_q"], EP["hover_blob"] = q, bh
    ok, info = grasp(q, bh)
    if not ok and str(info).startswith("nothing held"):
        EP["grasp_miss_1"] = info
        log(f"ep {k}: grasp missed ({info}); re-centring at hover and retrying lower")
        st = ga.state()
        qr = pa.move_rel(st["present"][:5], 25.0, [0.0, 0.0, pa.HOVER_DZ - (pa.TIP_FINAL + 45.0)], speed=CFG["hover"])
        want = EP.get("target_colour")
        b2 = pa.find(pa.blobs(pa.frame(), min_area=pa.SERVO_MIN_AREA), pa.target(), 260, want) if qr is not None else None
        if b2 is not None:
            qr, b2, okc = pa.servo(qr, 25.0, b2["centre"], log=log, want=want)
            if b2 is not None and okc:
                ok, info = grasp(qr, b2, retry=True)
                EP["grasp_retry_ok"] = ok
    T["pick_s"] = round(time.time() - t0, 1)
    if not ok:
        EP.update({"success": False, "failure": info})
        log(f"ep {k}: GRASP FAILED: {info}")
        ga.cmd({"op": "grip", "value": 25.0, "speed": 20})
        go_scan(False)
        ga.cmd({"op": "rec_stop"})
        return
    shape = info
    t1 = time.time()
    pre = EP.get("pre_close") or {}
    views = []
    if shape[0] == "2x2" and pre.get("verdict") not in ("CONFIDENT_DEFECT", "CONFIDENT_GOOD"):
        sv, ok2 = station_inspect(k)
        views = [sv]
        EP["station"] = sv
        if not ok2:
            EP.update({"success": False, "failure": "re-pick at station failed"})
            log(f"ep {k}: station re-pick failed")
            go_scan(False)
            ga.cmd({"op": "rec_stop"})
            return
    T["inspect_s"] = round(time.time() - t1, 1)
    clean = [v for v in views if not any(str(r).startswith(("merged", "colour_split")) for r in v.get("reasons", []))]
    pre = EP.get("pre_close") or {"verdict": "UNCERTAIN"}
    colour = pre.get("colour") or next((v.get("colour") for v in clean if v.get("colour")), None) or "unknown"
    EP["views_used"] = len(clean)
    plate, label, conf, reasons = sl.decide(pre, shape, clean)
    EP.update({"views": views, "colour": colour, "plate": plate, "label": label, "decision_conf": conf,
               "decision_reasons": reasons, "label_source": "det2", "needs_review": conf not in ("confident", "held_only"),
               "task": f"move the {colour} brick to the {plate} plate"})
    log(f"ep {k}: {colour} {shape[0]} -> {label} -> {plate} ({conf}; {reasons})")
    t2 = time.time()
    place(plate)
    T["place_s"] = round(time.time() - t2, 1)
    t3 = time.time()
    go_scan(False)
    T["return_s"] = round(time.time() - t3, 1)
    EP["t_end"] = time.time()
    ga.cmd({"op": "rec_stop"})
    T["cycle_s"] = round(EP["t_end"] - t0, 1)
    EP.update({"success": True, "failure": None, "timings": T})
    log(f"ep {k}: DONE {label} -> {plate} in {T['cycle_s']} s {T}")


def save_ep(k):
    if not EP:
        return
    EP["t_end"] = EP.get("t_end") or time.time()
    names = sorted(os.listdir(f"{ga.LOG_ROOT}/monitor")) if os.path.isdir(f"{ga.LOG_ROOT}/monitor") else []
    EP["cams"] = {"top": next(("logs/monitor/" + x for x in names if x.startswith("top_") and x.endswith(".mkv")), None),
                  "close": next(("logs/monitor/" + x for x in names if x.startswith("close_") and x.endswith(".mkv")), None),
                  "wrist_dir": "data/wrist/raw"}
    os.makedirs(f"{RUN}/ep_{k:04d}", exist_ok=True)
    json.dump(EP, open(f"{RUN}/ep_{k:04d}/episode.json", "w"), indent=1, default=str)
    with open(f"{RUN}/cycles.jsonl", "a") as f:
        f.write(json.dumps({x: EP.get(x) for x in ("episode", "t_start", "t_end", "task", "plate", "label", "decision_conf",
                                                  "decision_reasons", "colour", "shape", "success", "failure",
                                                  "grip_closed", "timings", "brick_xy_est", "mode")}, default=str) + "\n")


def clean_views(views):
    return [v for v in views if v.get("verdict") != "NOT_USED"
            and not any(str(r).startswith(("merged", "colour_split")) for r in v.get("reasons", []))]


def finish_held():
    """A brick is held: inspect (if no decision yet), decide, place."""
    plate = EP.get("plate")
    if not plate:
        sv, ok2 = station_inspect(EP.get("episode", 0))
        views = [sv]
        if not ok2:
            log("held brick put down at the station; re-pick failed - the next cycle picks it up")
            return None
        pre = EP.get("pre_close") or {"verdict": "UNCERTAIN"}
        clean = clean_views(views)
        plate, label, conf, reasons = sl.decide(pre, (EP.get("shape", "2x2"), EP.get("shape_why", "")), clean)
        colour = pre.get("colour") or next((v.get("colour") for v in clean if v.get("colour")), None) or "unknown"
        EP.update({"views": views, "plate": plate, "label": label, "decision_conf": conf, "colour": colour,
                   "decision_reasons": reasons, "label_source": "det2", "needs_review": conf not in ("confident", "held_only"), "task": f"move the {colour} brick to the {plate} plate"})
        log(f"held brick -> {label} {plate} ({conf}; {reasons})")
    place(plate)
    return plate


def recover():
    """After an estop: wait until the workspace is clear, finish a held brick or open, then go to the scan pose."""
    for attempt in range(40):
        try:
            ga.cmd({"op": "rec_stop"})
            sl.wait_clear("estop")
            st = ga.state()
            if st["target"][5] < 5 and 0.8 <= st["present"][5] <= 8:
                plate = finish_held()
                EP["failure"] = (EP.get("failure") or "") + f" (recovered: {'placed on ' + plate if plate else 'left at station'})"
            else:
                ga.cmd({"op": "grip", "value": 25.0, "speed": 20})
            go_scan(False)
            return True
        except ga.Stop as e:  # noqa: PERF203
            pause = min(2.0 ** min(attempt, 5), 30.0)  # 18:03Z: 40 retries burned in 1 s; now ~15 min before giving up
            log(f"recovery interrupted ({e}); retry {attempt + 1} in {pause:.0f} s")
            time.sleep(pause)
    return False


def run_one(k):
    try:
        cycle(k)
    except SystemExit as e:
        log(f"stopping: {e}")
        save_ep(k)
        return None
    except ga.Stop as e:
        EP.update({"success": False, "failure": f"stop: {e}"})
        log(f"ep {k}: STOP {e}; recovering")
        if not recover():
            save_ep(k)
            return None
    except Exception as e:  # noqa: BLE001
        EP.update({"success": False, "failure": f"exception {e!r}"})
        log(f"ep {k}: EXCEPTION {traceback.format_exc()[-900:]}")
        try:
            ga.cmd({"op": "rec_stop"})
            if not (ga.state()["target"][5] < 5 and 0.8 <= ga.state()["present"][5] <= 8):
                ga.cmd({"op": "grip", "value": 25.0, "speed": 20})
                go_scan(False)
            else:
                finish_held()
                go_scan(False)
        except ga.Stop:
            if not recover():
                save_ep(k)
                return None
    save_ep(k)
    return k + 1


def main():
    global RUN, EP
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--profile", choices=("safe", "fast"), default="safe",
                    help="fast = untested speed-ups (transit 45, hover 25, desc 14, final 6, servo 18); try it on a few "
                         "cycles and compare cycles.jsonl timings before using it for collection")
    a = ap.parse_args()
    RUN = f"{ga.LOG_ROOT}/sort/{a.run}"
    sl.RUN = RUN
    os.makedirs(RUN, exist_ok=True)
    if a.profile == "fast":
        CFG.update({"transit": 45.0, "hover": 25.0, "desc": 14.0, "final": 6.0, "servo": 18.0})
        pa.SERVO_SPEED = 18.0
    for key in ("transit", "hover", "servo", "desc"):
        CFG[key] = min(45.0, CFG[key] * a.speed)
    CFG["profile"] = a.profile
    if os.path.exists(f"{ga.RUN_ROOT}/sort_stop"):
        os.remove(f"{ga.RUN_ROOT}/sort_stop")
    log(f"start {a.run} n={a.n} cfg={CFG} target={ga.TARGET_PX}")
    jc = subprocess.run(["true"], capture_output=True, text=True) if ga.state()["target"][5] < 20 else subprocess.run([f"{R}/envs/data-env/bin/python", "-B", "jaw_check.py", "--no-move"], cwd=os.path.dirname(os.path.abspath(__file__)),
                        capture_output=True, text=True)
    log(f"jaw check: {jc.stdout.strip()[-200:]}")
    if os.path.exists(f"{ga.RUN_ROOT}/estop"):
        sl.wait_clear("estop at start")
    st0 = ga.state()
    if st0["target"][5] < 5 and 0.8 <= st0["present"][5] <= 8:
        k0 = a.start - 1
        EP = {"episode": k0, "run": os.path.basename(RUN), "mode": "carry_over", "t_start": time.time()}
        os.makedirs(f"{RUN}/ep_{k0:04d}/crops", exist_ok=True)
        ga.OUT = f"{RUN}/ep_{k0:04d}"
        log(f"holding a brick at start (gripper {st0['present'][5]}): inspect, decide, place")
        try:
            finish_held()
            EP["success"] = True
        except ga.Stop as e:
            log(f"carry-over interrupted ({e})")
            recover()
        save_ep(k0)
    try:
        go_scan(False)
    except ga.Stop:
        recover()
    k = a.start
    while k is not None and k < a.start + a.n:
        k = run_one(k)
    log("loop end")


if __name__ == "__main__":
    main()
