"""Apply B10: people behind the table stop the arm only when the top camera confirms a hand.
any_2127 21:27-21:59Z: 9 of 12 cycles were stopped by skin blobs at the top edge of the close view (people sitting
behind the far edge of the table; estop_close_*.jpg). The arm never works behind its base, and anything that reaches
into the work area crosses the top camera's view (its ROI starts at row 0 for that reason). A close-view blob that
touches only the top edge and stays inside the top 30 % of the frame now needs the top view to show a hand in the
same moment; every other close-view hand (sides, bottom, larger or deeper blobs) still stops on its own.
handdet features gain the blob's bounding box. Exact anchors; adds tests."""
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


h = open(f"{T}/handdet.py").read()
m = re.search(r'    return \{"area": a, "edge": edge, "fill": round\(a / float\(w \* h\), 3\),', h)
if not m:
    sys.exit("ABORT handdet return anchor")
patch(f"{T}/handdet.py", [
    ('    return {"area": a, "edge": edge, "fill": round(a / float(w * h), 3),',
     '    return {"area": a, "edge": edge, "bbox": [x, y, w, h], "fill": round(a / float(w * h), 3),'),
    ('''def is_hand_features(f: dict, view: str) -> bool:''',
     '''FAR_BAND = 0.30  # close view: top fraction of the frame that lies beyond the table's far edge


def far_band_only(f: dict, height: int) -> bool:
    """Close-view blob that touches only the top edge and stays in the top FAR_BAND of the frame: someone behind the
    table, beyond the far edge. Such a blob alone does not stop the arm; the top camera must confirm a hand."""
    if not f.get("bbox") or f.get("edge") not in ("T", "TL", "TR"):
        return False
    x, y, w, h = f["bbox"]
    return y + h <= FAR_BAND * height


def is_hand_features(f: dict, view: str) -> bool:'''),
], "")

patch(f"{T}/hand_watch.py", [
    ('''        hand, f = handdet.is_hand(img, view, hold)
        hits[view] = hits.get(view, 0) + 1 if hand else 0
''', '''        hand, f = handdet.is_hand(img, view, hold)
        if hand and view == "close" and handdet.far_band_only(f, img.shape[0]) and hits.get("top", 0) == 0:
            hits[view] = 0  # beyond the far edge and the top camera sees no hand: someone behind the table
            log.write(f"{time.strftime('%H:%M:%SZ', time.gmtime())} far-band close blob={f['area']}px edge={f['edge']} "
                      f"bbox={f['bbox']} (top view clear: no stop)\\n")
            log.flush()
            continue
        hits[view] = hits.get(view, 0) + 1 if hand else 0
'''),
], "")

TT = f"{T}/tests/test_safety.py"
t = open(TT).read()
add = '''

def test_safety_far_band_needs_the_top_camera():
    """A close-view skin blob touching only the top edge and inside the top 30 % is someone behind the table; one that
    reaches deeper, or touches a side, still counts on its own."""
    import handdet
    assert handdet.far_band_only({"bbox": [100, 0, 120, 90], "edge": "T"}, 480)
    assert handdet.far_band_only({"bbox": [500, 0, 140, 120], "edge": "TR"}, 480)
    assert not handdet.far_band_only({"bbox": [100, 0, 120, 200], "edge": "T"}, 480)     # reaches onto the mat
    assert not handdet.far_band_only({"bbox": [0, 60, 120, 60], "edge": "L"}, 480)       # from the side
    assert not handdet.far_band_only({"bbox": None, "edge": "T"}, 480)
'''
if "test_safety_far_band_needs_the_top_camera" not in t:
    open(TT, "w").write(t.rstrip("\n") + "\n" + add)
    print("test added")
