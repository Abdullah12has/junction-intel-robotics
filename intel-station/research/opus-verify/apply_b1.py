"""Apply B1 to app/tools/sort_any.py: close-view localisation by difference (closeloc), re-pick snapshots and fields
under the 're_' prefix, close-map pairs. Exact anchors; aborts without writing if any anchor count is wrong."""
import os
import py_compile
import re
import shutil
import sys
import time

R = "/home/intel/intel-junction-2026"
T = f"{R}/app/tools"
SRC = f"{T}/sort_any.py"
s = open(SRC).read()
orig = s


def once(old, new):
    global s
    n = s.count(old)
    if n != 1:
        sys.exit(f"ABORT: anchor found {n} times: {old[:80]!r}")
    s = s.replace(old, new)


# 1. imports and globals
once("import inspector as DET2  # noqa: E402\n",
     "import inspector as DET2  # noqa: E402\nimport closeloc as cl  # noqa: E402\n")
once("EP = {}\nRUN = \"\"\n",
     "EP = {}\nEPIMG = {}  # this episode's first frames, taken at the scan pose with the arm out of the close view\nRUN = \"\"\n"
     "CLOSE_MAP = cl.ClosePlaneMap(os.path.join(ga.RUN_ROOT, \"close_pairs.jsonl\"))\n")

# 2. grasp(): whole function replaced
i0 = s.index("def grasp(q, bh, retry=False):\n")
i1 = s.index("def station_inspect(k):\n")
GRASP = '''def rec_verdict(rec, method):
    """DET2 triage of one segmented brick, in the verdict format used by decide()."""
    verdict, reasons = DET2.triage(rec, DET2.load_thresholds())
    x, y, w, h = rec["bbox"]
    return {"verdict": verdict, "reasons": list(reasons), "colour": rec.get("colour"), "tape_score": rec.get("tape_score"),
            "bbox": [int(x), int(y), int(w), int(h)], "centre_px": [round(float(c), 1) for c in rec["centre_px"]],
            "hue": int(rec.get("hsv_med", [0])[0]), "method": method}


def _hue_check(out, want_hue):
    if want_hue is None or out.get("hue") is None:
        return out
    dh = abs(out["hue"] - want_hue)
    dh = min(dh, 180 - dh)
    out["hue_diff"] = int(dh)
    if dh > 10:  # the close view matched a different brick than the one centred in the wrist view
        out["verdict_raw"] = out["verdict"]
        out["verdict"] = "UNCERTAIN"
        out["reasons"] = list(out["reasons"]) + [f"close_brick_hue_mismatch_{int(dh)}"]
    return out


def pre_close_verdict(cimg, want_hue=None):
    """Close-camera verdict on the brick about to be picked. The jaws are the only new red parts since this episode's
    first close frame; the brick under them is judged on that first frame, where the gripper does not cover it. No
    fixed pixel is involved, so a moved close camera does not break it. Falls back to the jaw-tip rule (jaws entering
    from the image top) when the first frame is missing or no jaws are found."""
    before = (EPIMG.get("pre") or {}).get("close")
    if cimg is None:
        return None
    if before is None:
        out = close_pre_verdict(cimg, want_hue)
        if out is not None:
            out["method"] = "jaw_top_rule"
            if out.get("bbox"):
                sl.save_crop(cimg, out["bbox"], f"{ga.OUT}/crops/pre_close.png")
        return out
    tips = cl.jaw_tips_by_diff(cimg, before)
    if len(tips) < 2:
        out = close_pre_verdict(cimg, want_hue) or {"verdict": "UNCERTAIN", "reasons": ["no_jaws_in_close_view"]}
        out.update({"method": "jaw_top_rule", "jaw_tips_diff": [[round(x), round(y)] for x, y in tips]})
        if out.get("bbox"):
            sl.save_crop(cimg, out["bbox"], f"{ga.OUT}/crops/pre_close.png")
        return out
    mx = (tips[0][0] + tips[1][0]) / 2.0
    my = max(tips[0][1], tips[1][1])
    try:
        bricks = DET2.segment_close(sl.mask_jaws(before))["bricks"]
    except Exception as e:  # noqa: BLE001
        return {"verdict": "UNCERTAIN", "reasons": [f"det2_error {e!r}"], "method": "diff_first_frame"}
    rec, d = cl.brick_below(bricks, mx, my)
    if rec is None:
        return {"verdict": "UNCERTAIN", "reasons": ["no_brick_under_jaws_in_close_view"], "method": "diff_first_frame",
                "jaw_tips": [[round(x), round(y)] for x, y in tips]}
    out = rec_verdict(rec, "diff_first_frame")
    out.update({"jaw_tips": [[round(x), round(y)] for x, y in tips], "dist": d})
    sl.save_crop(before, out["bbox"], f"{ga.OUT}/crops/pre_close.png")
    return _hue_check(out, want_hue)


def grasp(q, bh, retry=False, tag=""):
    """Descend on the centred brick, close, lift, check the hold. tag="re_" for the re-pick at the station: its
    snapshots and fields get that prefix, so the first pick's record stays intact."""
    g = 25.0
    tip = pa.tip_from_size(bh["short"])
    fk_tip = float(kin.fk(q)[0][2] - kin.MAT_Z)
    use = tip if abs(tip - fk_tip) <= 15 else fk_tip
    tip_final = pa.TIP_FINAL
    if retry:  # a leaning upside-down brick looked big (size 51 mm, FK 58 mm) and the jaws closed too high
        use = min(max(tip, fk_tip), fk_tip + 6.0)
        tip_final = pa.TIP_FINAL - 2.0
    use = min(use, fk_tip + tip_final - MIN_TIP_MM)  # jaws end at tip_final + fk_tip - use: never below MIN_TIP_MM
    EP[tag + "grasp_use_mm" + ("_retry" if retry else "")] = round(use, 1)
    EP[tag + "hover_tip_size_mm"] = round(tip, 1)
    shape = ("not_2x2", f"aspect {bh['aspect']}") if bh["aspect"] > 1.4 else (
        ("not_2x2", f"narrow short {bh['short']}px vs {pa.SIZE_K / (use + pa.SIZE_C):.0f}") if bh["short"] < 0.72 * pa.SIZE_K / (use + pa.SIZE_C)
        else ("2x2", f"aspect {bh['aspect']} short {bh['short']}"))
    EP[tag + "shape"], EP[tag + "shape_why"] = shape
    cv2.imwrite(f"{ga.OUT}/crops/{tag}wrist_hover.png", pa.frame())
    cimg = ga.fresh("close", time.time() + 0.2)
    if cimg is not None:
        cv2.imwrite(f"{ga.OUT}/crops/{tag}hover_close.jpg", cimg)
    if not tag:
        EP["pre_close"] = pre_close_verdict(cimg, want_hue=bh.get("hue"))
    q1 = pa.move_rel(q, g, [0, 0, -(use - 26.0)], speed=CFG["desc"])
    if q1 is None:
        return False, "ik fail stage 1"
    b1 = pa.nearest(pa.blobs(pa.frame(), max_area=90000), pa.target(), max_d=260)
    ga.snap(tag + "stage1")
    q2 = pa.move_rel(q1, g, [0, 0, -(26.0 - tip_final)], speed=CFG["final"])
    if q2 is None:
        return False, "ik fail final descent"
    gimgs = ga.snap(tag + "grasp_open")
    bo = pa.nearest(pa.blobs(gimgs["wrist"], max_area=200000), pa.target(), max_d=400) if gimgs.get("wrist") is not None else None
    EP[tag + "open_short"] = bo["short"] if bo else None
    ga.cmd({"op": "grip", "value": 0.0, "speed": 20})
    time.sleep(0.5)
    gr = ga.state()["present"][5]
    EP[tag + "grip_closed"] = gr
    ga.snap(tag + "closed")
    ql = pa.move_rel(q2, 0.0, [0, 0, 45.0], speed=CFG["desc"])
    if ql is None:
        return False, "ik fail lift"
    imgs = ga.snap(tag + "lifted")
    gr2 = ga.state()["present"][5]
    held = pa.nearest(pa.blobs(imgs["wrist"], max_area=150000), pa.target(), max_d=300)
    EP.update({tag + "grip_lifted": gr2, tag + "lift_q": ql, tag + "stage1_blob": b1, tag + "lifted_blob": held})
    ratio = (held["short"] / EP[tag + "open_short"]) if (held and EP.get(tag + "open_short")) else None
    EP[tag + "held_size_ratio"] = round(ratio, 2) if ratio else None
    if gr2 < 1.1:  # an empty close reads 0.59, a held 2x2/2x4 reads 1.57-2.42
        ga.cmd({"op": "grip", "value": 25.0, "speed": 20})
        return False, f"nothing held (gripper {gr2}, size ratio {ratio})"
    return True, shape


'''
s = s[:i0] + GRASP + s[i1:]

# 3. station_inspect(): whole function replaced
i0 = s.index("def station_inspect(k):\n")
i1 = s.index("def tmove(q, speed, comp=False, accel=120.0):\n")
STATION = '''def station_verdict(st_img, put_xy):
    """Verdict on the brick just put down at the station: the brick that appeared since this episode's first close
    frame. The brick that vanished from that frame is the picked brick's original view (kept as first_frame_view).
    Both give a table-position -> close-pixel pair for the close map. Falls back to the fixed spot rule when there is
    no first frame or no clean changed brick."""
    before = (EPIMG.get("pre") or {}).get("close")
    if st_img is None:
        return {"verdict": "UNCERTAIN", "reasons": ["no_close_frame"], "method": "none"}
    if before is None:
        v = sl.close_verdict(st_img)
        v["method"] = "fixed_spot"
        return v
    ch = cl.changed_mask(st_img, before)
    try:
        seg_st = DET2.segment_close(sl.mask_jaws(st_img))["bricks"]
        seg_pre = DET2.segment_close(sl.mask_jaws(before))["bricks"]
    except Exception as e:  # noqa: BLE001
        return {"verdict": "UNCERTAIN", "reasons": [f"det2_error {e!r}"], "method": "diff_first_frame"}
    app, fa, fa2 = cl.pick_changed(seg_st, ch)
    van, fv, fv2 = cl.pick_changed(seg_pre, ch)
    if van is not None:
        ffv = rec_verdict(van, "vanished_first_frame")
        ffv["changed_frac"] = [fv, fv2]
        EP["first_frame_view"] = ffv
        sl.save_crop(before, ffv["bbox"], f"{ga.OUT}/crops/first_frame_view.png")
        lq = EP.get("lift_q")
        if lq is not None:
            gx, gy = (float(x) for x in kin.fk(lq)[0][:2])
            CLOSE_MAP.add(gx, gy, ffv["centre_px"][0], ffv["centre_px"][1], "grasp")
    if app is None:
        v = sl.close_verdict(st_img)
        v.update({"method": "fixed_spot", "changed_frac": [fa, fa2]})
        return v
    v = rec_verdict(app, "appeared")
    v["changed_frac"] = [fa, fa2]
    if put_xy is not None:
        CLOSE_MAP.add(float(put_xy[0]), float(put_xy[1]), v["centre_px"][0], v["centre_px"][1], "station")
    return v


def station_inspect(k):
    """Put the held brick down at the station, retract to the scan pose, judge it in the close view (station_verdict),
    re-pick it (snapshots and fields under the re_ prefix). Returns (verdict dict, ok)."""
    c = scan_cal()
    hi = kin.ik_rel(SPOT_HOVER, [0, 0, 35.0])
    p, _ = kin.fk(ga.state()["present"][:5])
    if p[2] < kin.MAT_Z + 80:
        pa.move_rel(ga.state()["present"][:5], 0.0, [0, 0, kin.MAT_Z + 85 - p[2]], speed=CFG["desc"])
    ga.move(hi + [0.0], speed=CFG["transit"], comp=True)
    qd = pa.move_rel(hi, 0.0, [0, 0, -(35.0 + pa.HOVER_DZ - pa.TIP_FINAL - 1.0)], speed=CFG["desc"])
    if qd is None:  # never put down: the brick is still held, so the cycle decides on the evidence it has
        return {"verdict": "UNCERTAIN", "reasons": ["station put-down IK failed"], "method": "none"}, True
    ga.cmd({"op": "grip", "value": 25.0, "speed": 25})
    time.sleep(0.3)
    put_xy = kin.fk(qd)[0][:2]
    pa.move_rel(qd, 25.0, [0, 0, 40.0], speed=CFG["desc"])
    ga.move(c["q_scan"] + [25.0], speed=CFG["transit"], comp=True)
    time.sleep(0.4)
    imgs = ga.snap("station")
    v = station_verdict(imgs.get("close"), put_xy)
    if imgs.get("close") is not None:
        cv2.imwrite(f"{ga.OUT}/crops/station_close_full.jpg", imgs["close"])
        sl.save_crop(imgs["close"], v.get("bbox"), f"{ga.OUT}/crops/station_close.png")
    log(f"ep {k}: station close view [{v.get('method')} {v.get('changed_frac')}] {v.get('colour')} {v.get('verdict')} "
        f"{v.get('reasons')} tape {v.get('tape_score')}")
    cand = candidates(imgs["wrist"], c["q_scan"], c)
    if not cand:
        return v, False
    q, bh = approach(cand[0], c)
    if q is None:
        return v, False
    ok, info = grasp(q, bh, tag="re_")
    return v, ok


'''
s = s[:i0] + STATION + s[i1:]

# 4. cycle(): keep the episode's first frames; use the picked brick's first-frame view as extra evidence
once('    EP.update({"target_scan_blob": b, "n_candidates": len(cand), "brick_xy_est": b.get("xy")})\n    ga.snap("pre")\n',
     '    EP.update({"target_scan_blob": b, "n_candidates": len(cand), "brick_xy_est": b.get("xy")})\n    EPIMG.clear()\n'
     '    EPIMG["pre"] = ga.snap("pre")\n')
once('        sv, ok2 = station_inspect(k)\n        views = [sv]\n        EP["station"] = sv\n',
     '        sv, ok2 = station_inspect(k)\n        views = [sv]\n        ffv = EP.get("first_frame_view")\n'
     '        if ffv and (pre.get("method") != "diff_first_frame" or pre.get("bbox") != ffv.get("bbox")):\n'
     '            views.append(ffv)  # the picked brick as first seen, when the pre-grasp view did not already judge it\n'
     '        EP["station"] = sv\n')

if s == orig:
    sys.exit("ABORT: nothing changed")
bak = f"{R}/run/bak_sort_any_{time.strftime('%H%M', time.gmtime())}.py"
shutil.copy(SRC, bak)
tmp = SRC + ".tmp"
open(tmp, "w").write(s)
py_compile.compile(tmp, doraise=True)
os.replace(tmp, SRC)
print("sort_any.py patched; backup", bak)

# 5. tests: register test_closeloc
rt = f"{T}/tests/run_tests.py"
t = open(rt).read()
m = re.search(r"MODULES\s*=\s*\[([^\]]*)\]", t)
if not m:
    sys.exit("ABORT: MODULES list not found in run_tests.py")
if "test_closeloc" not in m.group(1):
    inner = m.group(1).rstrip()
    sep = "" if inner.endswith(",") or not inner.strip() else ","
    t = t[:m.start(1)] + inner + sep + ' "test_closeloc"' + t[m.end(1):]
    open(rt, "w").write(t)
print("run_tests MODULES:", re.search(r"MODULES\s*=\s*\[[^\]]*\]", t).group(0))
