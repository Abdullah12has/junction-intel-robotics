"""Apply B7: when the tracked brick is not within the search radius (any_2127 ep 9 lost after a 35 deg yaw, ep 10 lost
by the raised-look servo), look in the whole frame for a blob of the same colour class and size and take the one
nearest the expected position; every loss is logged with the blobs seen and the frame saved, so the next one can be
diagnosed. Exact anchors."""
import os
import py_compile
import shutil
import sys
import time

R = "/home/intel/intel-junction-2026"
P = f"{R}/app/tools/pick_any.py"
s = open(P).read()
old = '''def find(bl, px, max_d, want=None, short_ref=None):
    """Blob nearest px of the tracked brick's colour class (exact class first, then a compatible one). 17:56Z: the
    servo jumped to the neighbouring pink brick after the yaw because it took any blob nearest the target."""
    b = nearest(bl, px, max_d=max_d, ok=_ok(want, short_ref, True))
    if b is None and want:
        b = nearest(bl, px, max_d=max_d, ok=_ok(want, short_ref, False))
    return b
'''
new = '''LOST_LOG = []  # (utc, px, blobs) of searches that failed inside the radius; the loop writes them to the episode


def find(bl, px, max_d, want=None, short_ref=None):
    """Blob nearest px of the tracked brick's colour class (exact class first, then a compatible one), within max_d.
    If nothing is within max_d, the whole frame is searched for the same colour class (and size, when short_ref is
    given): the brick moved further than expected (large yaw, a servo step at another height), it did not vanish.
    A neighbour of the same class nearer the expected position still wins, as before."""
    b = nearest(bl, px, max_d=max_d, ok=_ok(want, short_ref, True))
    if b is None and want:
        b = nearest(bl, px, max_d=max_d, ok=_ok(want, short_ref, False))
    if b is None and want:
        b = nearest(bl, px, max_d=1e9, ok=_ok(want, short_ref, True)) or nearest(bl, px, max_d=1e9, ok=_ok(want, short_ref, False))
        LOST_LOG.append((time.strftime("%H:%M:%SZ", time.gmtime()), [round(float(v), 1) for v in px], round(float(max_d)),
                         [(x.get("colour"), int(x.get("area", 0)), [round(c) for c in x["centre"]], x.get("short")) for x in bl],
                         None if b is None else [round(c) for c in b["centre"]]))
    return b
'''
if s.count(old) != 1:
    sys.exit("ABORT pick_any find anchor")
s = s.replace(old, new)
bak = f"{R}/run/bak_pick_any_{time.strftime('%H%M', time.gmtime())}.py"
shutil.copy(P, bak)
open(P + ".tmp", "w").write(s)
py_compile.compile(P + ".tmp", doraise=True)
os.replace(P + ".tmp", P)
print("patched", P, "backup", bak)

S = f"{R}/app/tools/sort_any.py"
t = open(S).read()
old2 = '''    if bh is None:
        cv2.imwrite(f"{ga.OUT}/crops/hover_not_found.jpg", pa.frame())
        return None, f"{want} brick not visible at hover"
'''
new2 = '''    if bh is None:
        fr = pa.frame()
        cv2.imwrite(f"{ga.OUT}/crops/hover_not_found.jpg", fr)
        seen = [(x.get("colour"), int(x.get("area", 0)), [round(v) for v in x["centre"]], x.get("short"))
                for x in pa.blobs(fr, min_area=pa.SERVO_MIN_AREA)]
        log(f"hover search failed: wanted {want}; blobs now {seen}")
        return None, f"{want} brick not visible at hover"
'''
if t.count(old2) != 1:
    sys.exit("ABORT sort_any hover anchor")
t = t.replace(old2, new2)
old3 = '''def save_ep(k):
    if not EP:
        return
'''
new3 = '''def save_ep(k):
    if not EP:
        return
    if pa.LOST_LOG:  # searches that had to look beyond the radius during this episode
        EP["search_beyond_radius"] = list(pa.LOST_LOG)
        pa.LOST_LOG.clear()
'''
if t.count(old3) != 1:
    sys.exit("ABORT sort_any save_ep anchor")
t = t.replace(old3, new3)
bak2 = f"{R}/run/bak_sort_any_{time.strftime('%H%M', time.gmtime())}g.py"
shutil.copy(S, bak2)
open(S + ".tmp", "w").write(t)
py_compile.compile(S + ".tmp", doraise=True)
os.replace(S + ".tmp", S)
print("patched", S, "backup", bak2)
