"""Apply B8 to pick_any.find: the whole-frame fallback (B7) must not switch to another brick. any_2127 ep 11: after a
21 deg yaw the tracked pink brick (area 9641, short 111.6) was missing from the blobs and the fallback took a red
blob 270 px away (area 13822: another brick), then missed it. The fallback now requires the exact colour class and,
when the tracked size is known, short side within +-20 % (an area check needs the reference area, which callers do
not pass). The frame of every search that leaves the radius is saved for diagnosis. Exact anchors."""
import os
import py_compile
import shutil
import sys
import time

R = "/home/intel/intel-junction-2026"
P = f"{R}/app/tools/pick_any.py"
s = open(P).read()
old = '''    if b is None and want:
        b = nearest(bl, px, max_d=1e9, ok=_ok(want, short_ref, True)) or nearest(bl, px, max_d=1e9, ok=_ok(want, short_ref, False))
        LOST_LOG.append((time.strftime("%H:%M:%SZ", time.gmtime()), [round(float(v), 1) for v in px], round(float(max_d)),
'''
new = '''    if b is None and want:
        def same(x):  # exact colour class; size within +-20 % when the tracked size is known
            return (_ok(want, None, True)(x)
                    and (short_ref is None or 0.8 * short_ref <= x["short"] <= 1.2 * short_ref))
        b = nearest(bl, px, max_d=1e9, ok=same)
        try:
            os.makedirs(f"{ga.RUN_ROOT}/lost", exist_ok=True)
            cv2.imwrite(f"{ga.RUN_ROOT}/lost/{time.strftime('%H%M%S', time.gmtime())}.jpg", frame_last())
        except Exception:  # noqa: BLE001
            pass
        LOST_LOG.append((time.strftime("%H:%M:%SZ", time.gmtime()), [round(float(v), 1) for v in px], round(float(max_d)),
'''
if s.count(old) != 1:
    sys.exit("ABORT find fallback anchor")
s = s.replace(old, new)
old2 = '''def frame():
    return ga.fresh("wrist", time.time() + 0.25)
'''
new2 = '''_LAST = {"img": None}


def frame():
    img = ga.fresh("wrist", time.time() + 0.25)
    _LAST["img"] = img
    return img


def frame_last():
    """The last frame returned by frame() (what the failed search looked at)."""
    return _LAST["img"]
'''
if s.count(old2) != 1:
    sys.exit("ABORT frame anchor")
s = s.replace(old2, new2)
if "\nimport os\n" not in s:
    s = s.replace("\nimport math\n", "\nimport math\nimport os\n", 1)
bak = f"{R}/run/bak_pick_any_{time.strftime('%H%M', time.gmtime())}b.py"
shutil.copy(P, bak)
open(P + ".tmp", "w").write(s)
py_compile.compile(P + ".tmp", doraise=True)
os.replace(P + ".tmp", P)
print("patched", P, "backup", bak)
