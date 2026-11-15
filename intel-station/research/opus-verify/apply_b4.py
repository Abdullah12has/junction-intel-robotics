"""Apply B4 to sort_any.py: plates are excluded in the wrist scan image itself (large plate-coloured regions touching
the image border), so plate exclusion no longer depends on the learned scan map (one new scan pair moved a blue plate
crescent out of the table keep-out at 21:33Z). Adds a test that holds with the scan map reset. Exact anchors."""
import os
import py_compile
import shutil
import sys
import time

R = "/home/intel/intel-junction-2026"
S = f"{R}/app/tools/sort_any.py"
s = open(S).read()


def once(old, new):
    global s
    n = s.count(old)
    if n != 1:
        sys.exit(f"ABORT: anchor found {n} times: {old[:90]!r}")
    s = s.replace(old, new)


once('''def candidates(img, q_scan, c):
    out = []
    p, _ = kin.fk(q_scan)
    for b in pa.blobs(img, min_area=700, max_area=40000):
        if b["plate_frac"] >= 0.3 or (b.get("colour") in ("blue", "green") and b["plate_frac"] >= 0.15):
            continue  # inside a plate, or a plate rim fragment
''', '''PLATE_MIN_AREA = 8000  # px at the scan pose: plates measured 9178-27306 px, bricks 850-6400 px (2x2 and 2x4 smaller than this)


def plate_regions(img):
    """Plate pixels in a wrist scan frame: plate-coloured regions of at least PLATE_MIN_AREA that touch the image
    border (both plates sit at the sides of the work area), dilated 15 px. Independent of the learned scan map."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue = np.zeros(H.shape, np.uint8)
    for lo, hi in pa.PLATE_HUES:
        hue |= ((H >= lo) & (H < hi) & (S >= 70) & (V >= 70)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(hue)
    h, w = H.shape
    plates = np.zeros(H.shape, np.uint8)
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if area >= PLATE_MIN_AREA and (x == 0 or y == 0 or x + bw >= w or y + bh >= h):
            plates[lab == i] = 1
    return cv2.dilate(plates, np.ones((15, 15), np.uint8))


def candidates(img, q_scan, c):
    out = []
    p, _ = kin.fk(q_scan)
    plates = plate_regions(img)
    for b in pa.blobs(img, min_area=700, max_area=40000):
        if b["plate_frac"] >= 0.3 or (b.get("colour") in ("blue", "green") and b["plate_frac"] >= 0.15):
            continue  # inside a plate, or a plate rim fragment
        cx, cy = int(round(b["centre"][0])), int(round(b["centre"][1]))
        if 0 <= cy < plates.shape[0] and 0 <= cx < plates.shape[1] and plates[cy, cx]:
            continue  # part of a plate, found in the image (no table map involved)
''')
bak = f"{R}/run/bak_sort_any_{time.strftime('%H%M', time.gmtime())}d.py"
shutil.copy(S, bak)
tmp = S + ".tmp"
open(tmp, "w").write(s)
py_compile.compile(tmp, doraise=True)
os.replace(tmp, S)
print("patched", S, "backup", bak)

T = f"{R}/app/tools/tests/test_perception.py"
t = open(T).read()
add = '''

def test_scan_plates_excluded_without_the_scan_map():
    """Plate exclusion must not depend on the learned scan map: with the map reset to the raw Jacobian (as after a
    camera move), scan_now.jpg still yields no plate blobs among the candidates (21:33Z: one new scan pair moved the
    blue plate crescent at (592, 307) out of the table keep-out)."""
    c = C.load_json("scan_calib.json")
    img = C.load_img("scan_now.jpg")
    saved = (C.sa.SCAN_C, C.sa.SCAN_H)
    try:
        C.sa.SCAN_C, C.sa.SCAN_H = np.eye(2), None
        cands = C.sa.candidates(img, c["q_scan"], c)
    finally:
        C.sa.SCAN_C, C.sa.SCAN_H = saved
    plate_hits = [(b["colour"], b["centre"]) for b in cands if b["colour"] in ("blue", "green") or b["plate_frac"] >= 0.3]
    assert not plate_hits, f"plate blobs among candidates with the raw scan map: {plate_hits}"
'''
if "test_scan_plates_excluded_without_the_scan_map" not in t:
    if "import numpy as np" not in t:
        t = t.replace("import cv2\n", "import cv2\nimport numpy as np\n", 1)
    t = t.rstrip("\n") + "\n" + add
    open(T, "w").write(t)
    print("test added")
