"""Apply B11: OD-10 fail-safe in decide(), and a first frame for a held brick after a restart.
OD-10 (binding, Hassan 13:47Z): UNCERTAIN -> GREEN. decide() sent three uncertain cases to BLUE: no usable tape view
('no_evidence', GOOD by default), a tape score below the threshold midpoint that the detector itself did not call
CONFIDENT_GOOD ('uncertain_low_score'), and any score when the thresholds could not be loaded. All three now go to
GREEN with label UNCERTAIN. A brick whose shape was never measured is not known to be 2x2: UNCERTAIN -> GREEN too.
After a loop restart with a brick in the gripper the station verdict had no first close frame ('no first frame or
station frame', 22:02:11Z): station_inspect now takes one at the scan pose first, and finish_held uses the re-pick's
shape measurement. Exact anchors; adds tests."""
import os
import py_compile
import shutil
import sys
import time

R = "/home/intel/intel-junction-2026"
T = f"{R}/app/tools"


def patch(path, edits):
    s = open(path).read()
    for old, new in edits:
        n = s.count(old)
        if n != 1:
            sys.exit(f"ABORT {os.path.basename(path)}: anchor found {n} times: {old[:90]!r}")
        s = s.replace(old, new)
    bak = f"{R}/run/bak_{os.path.basename(path)[:-3]}_{time.strftime('%H%M', time.gmtime())}_b11.py"
    shutil.copy(path, bak)
    open(path + ".tmp", "w").write(s)
    py_compile.compile(path + ".tmp", doraise=True)
    os.replace(path + ".tmp", path)
    print("patched", path, "backup", bak)


patch(f"{T}/sort_loop.py", [
    ('''def decide(pre, shape, views):
    reasons = []
    if shape[0] != "2x2":
        reasons.append("not_2x2:" + shape[1])
''', '''def decide(pre, shape, views):
    """OD-10 (binding): GOOD (2x2, no tape) -> blue; DEFECT (tape or not 2x2) -> green; UNCERTAIN -> green.
    Returns (plate, label, confidence, reasons). Blue needs a measured 2x2 shape and a CONFIDENT_GOOD view."""
    reasons = []
    shape_known = shape[0] in ("2x2", "not_2x2")
    if shape_known and shape[0] != "2x2":
        reasons.append("not_2x2:" + shape[1])
'''),
    ('''    if reasons:
        return "green", "DEFECT", "confident", reasons
    goods = [pre.get("verdict") == "CONFIDENT_GOOD"]''', '''    if reasons:
        return "green", "DEFECT", "confident", reasons
    if not shape_known:  # e.g. a held brick after a restart whose re-pick failed: not known to be 2x2
        return "green", "UNCERTAIN", "shape_unknown", ["shape not measured: UNCERTAIN -> GREEN (OD-10)"]
    goods = [pre.get("verdict") == "CONFIDENT_GOOD"]'''),
    ('''    scores = [x.get("tape_score") for x in [pre] + views if x.get("tape_score") is not None]
    thr = DET.load_thresholds() or {}
    mid = (thr.get("good", 0) + thr.get("defect", 1)) / 2 if thr else None
    if scores and mid is not None and max(scores) >= mid:
        return "green", "DEFECT", "uncertain_high_score", [f"max_tape {max(scores):.3f} >= mid {mid:.3f}"]
    if not scores:  # 18:50Z (ANOM3 review): no usable view at all used to read as GOOD 'uncertain_low_score'
        return "blue", "GOOD", "no_evidence", ["no usable tape view; GOOD by default (needs review)"]
    return "blue", "GOOD", "uncertain_low_score", [f"max_tape {max(scores)}"]
''', '''    scores = [x.get("tape_score") for x in [pre] + views if x.get("tape_score") is not None]
    if not scores:  # until 22:10Z this went to blue as 'GOOD by default'
        return "green", "UNCERTAIN", "no_evidence", ["no usable tape view: UNCERTAIN -> GREEN (OD-10)"]
    thr = DET.load_thresholds() or {}
    mid = (thr.get("good", 0) + thr.get("defect", 1)) / 2 if thr else None
    if mid is None:
        return "green", "UNCERTAIN", "no_threshold", [f"max_tape {max(scores):.3f}, thresholds not loaded (OD-10)"]
    if max(scores) >= mid:
        return "green", "DEFECT", "uncertain_high_score", [f"max_tape {max(scores):.3f} >= mid {mid:.3f}"]
    return "green", "UNCERTAIN", "uncertain_low_score", [f"max_tape {max(scores):.3f} < mid {mid:.3f}, no "
                                                         "CONFIDENT_GOOD view: UNCERTAIN -> GREEN (OD-10)"]
'''),
])

patch(f"{T}/sort_any.py", [
    ('''    ga.move(hi + [0.0], speed=CFG["transit"], comp=True)
    qd = pa.move_rel(hi, 0.0,''', '''    if (EPIMG.get("pre") or {}).get("close") is None:  # loop restarted holding a brick: no first frame yet
        ga.move(c["q_scan"] + [0.0], speed=CFG["transit"], comp=True)
        time.sleep(0.4)
        EPIMG["pre"] = ga.snap("carry_pre")
        log("station: no first close frame this episode; took one at the scan pose (brick held)")
    ga.move(hi + [0.0], speed=CFG["transit"], comp=True)
    qd = pa.move_rel(hi, 0.0,'''),
    ('''        plate, label, conf, reasons = sl.decide(pre, (EP.get("shape", "2x2"), EP.get("shape_why", "")), clean)''',
     '''        shape = (EP.get("shape") or EP.get("re_shape") or "unknown", EP.get("shape_why") or EP.get("re_shape_why") or "")
        plate, label, conf, reasons = sl.decide(pre, shape, clean)'''),
])

TT = f"{T}/tests/test_loopfix.py"
t = open(TT).read()
add = '''

def test_decide_follows_od10_failsafe():
    """OD-10: blue only for a measured 2x2 with a CONFIDENT_GOOD view; every UNCERTAIN case goes to green."""
    import sort_loop as sl
    good = {"verdict": "CONFIDENT_GOOD", "tape_score": 0.01}
    unc = {"verdict": "UNCERTAIN"}
    assert sl.decide(good, ("2x2", "a"), [])[:2] == ("blue", "GOOD")
    assert sl.decide(unc, ("2x2", "a"), [good])[:3] == ("blue", "GOOD", "held_only")
    assert sl.decide(good, ("not_2x2", "aspect 2.0"), [])[:2] == ("green", "DEFECT")
    assert sl.decide({"verdict": "CONFIDENT_DEFECT", "reasons": ["tape"]}, ("2x2", "a"), [])[:2] == ("green", "DEFECT")
    assert sl.decide(unc, ("2x2", "a"), [unc])[:3] == ("green", "UNCERTAIN", "no_evidence")
    assert sl.decide(good, ("unknown", ""), [])[:3] == ("green", "UNCERTAIN", "shape_unknown")
'''
if "test_decide_follows_od10_failsafe" not in t:
    open(TT, "w").write(t.rstrip("\n") + "\n" + add)
    print("test added")
