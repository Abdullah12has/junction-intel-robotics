"""Apply B6: station and first-frame localisation that survives people moving behind the table and a brick picked
from the station spot itself (any_2127 ep 7/8: a moving forearm at the top of the close view was taken for the
'appeared' brick; the white brick picked from the station and put back there barely changed). Now:
- changed regions connected to the image top edge are ignored (people beyond the table enter from the top);
- candidate bricks must have a colour compatible with the colour the wrist camera centred;
- the station's pixel is learned from clean detections (run/station_px.json) and used when the difference is weak;
- no fixed-pixel fallback any more: no clean station brick -> UNCERTAIN (no evidence), never another brick.
Exact anchors; adds tests."""
import os
import py_compile
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


patch(f"{T}/closeloc.py", [(
    '''def red_mask(img):''',
    '''def drop_top_edge(changed, rows=2):
    """Remove changed regions connected to the top image edge: people and arms beyond the table enter the close view
    from the top, a brick on the mat never touches it."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(changed)
    out = changed.copy()
    for i in range(1, n):
        if st[i, cv2.CC_STAT_TOP] < rows:
            out[lab == i] = 0
    return out


# colour the wrist camera reports for the centred brick -> colours the close-view inspector may call the same brick
COMPATIBLE = {"white": {"white", "cream"}, "yellow": {"yellow", "cream"}, "red": {"red", "pink", "orange"},
              "orange": {"orange", "red"}, "pink": {"pink", "red"}, "blue": {"blue"}, "green": {"green"}}


def colour_ok(close_colour, wrist_colour):
    """True when the close-view colour can be the brick the wrist camera centred (unknown on either side: True)."""
    if not wrist_colour or not close_colour or wrist_colour not in COMPATIBLE:
        return True
    return close_colour in COMPATIBLE[wrist_colour]


def red_mask(img):''')], "")

S = f"{T}/sort_any.py"
s = open(S).read()
i0 = s.index("def station_verdict(st_img, put_xy):\n")
i1 = s.index("def station_inspect(k):\n")
NEW = '''STATION_PX_FILE = os.path.join(ga.RUN_ROOT, "station_px.json")


def station_px():
    try:
        return json.load(open(STATION_PX_FILE))["px"]
    except (OSError, ValueError, KeyError):
        return None


def learn_station_px(px):
    old = station_px()
    new = px if old is None else [0.7 * old[0] + 0.3 * px[0], 0.7 * old[1] + 0.3 * px[1]]
    tmp = STATION_PX_FILE + ".tmp"
    json.dump({"px": [round(new[0], 1), round(new[1], 1)], "t": time.strftime("%H:%M:%SZ", time.gmtime())}, open(tmp, "w"))
    os.replace(tmp, STATION_PX_FILE)


def station_verdict(st_img, put_xy):
    """Verdict on the brick just put down at the station, never on another brick. The station brick is the brick that
    appeared since this episode's first close frame (changes connected to the image top edge = people, ignored), with
    a colour compatible with what the wrist camera centred; when the difference is weak (e.g. the brick was picked
    from the station and put back) the brick at the learned station pixel is used. No clean candidate -> UNCERTAIN.
    The picked brick's original view (the brick that vanished from the first frame) is kept as first_frame_view.
    Clean detections give table-position -> close-pixel pairs for the close map."""
    before = (EPIMG.get("pre") or {}).get("close")
    want = EP.get("target_colour")
    if st_img is None or before is None:
        return {"verdict": "UNCERTAIN", "reasons": ["no first frame or station frame"], "method": "none"}
    ch = cl.drop_top_edge(cl.changed_mask(st_img, before))
    try:
        seg_st = [r for r in DET2.segment_close(sl.mask_jaws(st_img))["bricks"] if cl.colour_ok(r.get("colour"), want)]
        seg_pre = [r for r in DET2.segment_close(sl.mask_jaws(before))["bricks"] if cl.colour_ok(r.get("colour"), want)]
    except Exception as e:  # noqa: BLE001
        return {"verdict": "UNCERTAIN", "reasons": [f"det2_error {e!r}"], "method": "none"}
    van, fv, fv2 = cl.pick_changed(seg_pre, ch, min_frac=0.5)
    if van is not None:
        ffv = rec_verdict(van, "vanished_first_frame")
        ffv["changed_frac"] = [fv, fv2]
        EP["first_frame_view"] = ffv
        sl.save_crop(before, ffv["bbox"], f"{ga.OUT}/crops/first_frame_view.png")
        lq = EP.get("lift_q")
        if lq is not None and fv >= 0.7:
            gx, gy = (float(x) for x in kin.fk(lq)[0][:2])
            CLOSE_MAP.add(gx, gy, ffv["centre_px"][0], ffv["centre_px"][1], "grasp")
    sp = station_px()
    app, fa, fa2 = cl.pick_changed(seg_st, ch, min_frac=0.45)
    if app is not None:
        d = None if sp is None else float(np.hypot(app["centre_px"][0] - sp[0], app["centre_px"][1] - sp[1]))
        if d is None or d <= 120 or fa >= 0.7:  # far from the learned spot only with a strong change (camera moved)
            v = rec_verdict(app, "appeared")
            v.update({"changed_frac": [fa, fa2], "station_px_dist": None if d is None else round(d, 1)})
            if fa >= 0.6:
                learn_station_px(v["centre_px"])
                if put_xy is not None:
                    CLOSE_MAP.add(float(put_xy[0]), float(put_xy[1]), v["centre_px"][0], v["centre_px"][1], "station")
            return v
    if sp is not None:
        rec, d = cl.nearest(seg_st, sp[0], sp[1], 90)
        if rec is not None:
            v = rec_verdict(rec, "station_px")
            v.update({"changed_frac": [fa, fa2], "station_px_dist": d})
            return v
    return {"verdict": "UNCERTAIN", "reasons": ["station brick not found (no clean change, no brick at the learned spot)"],
            "method": "none", "changed_frac": [fa, fa2]}


'''
s = s[:i0] + NEW + s[i1:]
bak = f"{R}/run/bak_sort_any_{time.strftime('%H%M', time.gmtime())}f.py"
shutil.copy(S, bak)
open(S + ".tmp", "w").write(s)
py_compile.compile(S + ".tmp", doraise=True)
os.replace(S + ".tmp", S)
print("patched", S, "backup", bak)

# seed the learned station pixel from the two clean observations with the current close-camera pose
# (any_2055 ep1 appeared at (495, 365), frac 0.71; any_2127 ep7 white brick at the station ~(530, 397))
spf = os.path.join(R, "run", "station_px.json")
if not os.path.exists(spf):
    import json
    json.dump({"px": [512.0, 381.0], "t": time.strftime("%H:%M:%SZ", time.gmtime()), "seed": "any_2055 ep1 + any_2127 ep7"},
              open(spf, "w"))
    print("seeded", spf)

TT = f"{T}/tests/test_closeloc.py"
t = open(TT).read()
add = '''

def test_people_at_the_top_edge_are_not_a_brick():
    """A forearm moving behind the table enters the close view from the top: its change must be ignored."""
    before, after = _frame(), _frame()
    cv2.rectangle(before, (100, 0), (180, 120), (90, 120, 200), -1)       # skin-coloured arm from the top edge
    cv2.rectangle(after, (120, 0), (200, 130), (90, 120, 200), -1)        # it moved
    arm = {"bbox": [120, 40, 80, 90], "centre_px": [160.0, 85.0]}
    rec, frac, _ = cl.pick_changed([arm], cl.drop_top_edge(cl.changed_mask(after, before)))
    assert rec is None and frac < 0.05, frac


def test_colour_compatibility():
    assert cl.colour_ok("cream", "white") and cl.colour_ok("pink", "red") and not cl.colour_ok("orange", "white")
    assert cl.colour_ok("blue", None) and cl.colour_ok(None, "white") and not cl.colour_ok("green", "blue")


def test_moving_forearm_rejected_on_recorded_frames():
    """Run any_2127 ep7: between the first close frame and the station frame a person's forearm moved at the top-left;
    with top-edge changes dropped and the white target's colour required, no candidate comes from that region."""
    fx = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "closeloc")
    pre = cv2.imread(os.path.join(fx, "ep7_pre_close.jpg"))
    st = cv2.imread(os.path.join(fx, "ep7_station_close.jpg"))
    assert pre is not None and st is not None, "fixture frames missing"
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                    "anomaly", "det2_f2"))
    import inspector as DET  # noqa: E402
    ch = cl.drop_top_edge(cl.changed_mask(st, pre))
    seg = [r for r in DET.segment_close(_mask_top_red(st))["bricks"] if cl.colour_ok(r.get("colour"), "white")]
    app, fa, _ = cl.pick_changed(seg, ch)
    assert app is None or app["centre_px"][1] > 250, app      # never the top-left forearm (centre y ~ 91)
    near, _ = cl.nearest(seg, 512, 381, 90)
    assert near is not None and near.get("colour") in ("white", "cream"), near
'''
if "test_people_at_the_top_edge_are_not_a_brick" not in t:
    marker = '\n\nif __name__ == "__main__":'
    t = t.replace(marker, add + marker)
    open(TT, "w").write(t)
    print("tests added")
