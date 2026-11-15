"""E1 offline timing of the cycle hot spots on the recorded test fixtures. Reads files only; no device access.
Run: envs/data-env/bin/python -I -B work/P3-CODE-REVIEW/E1/raw/31-time_fixtures.py
"""
import glob
import os
import statistics
import sys
import time

sys.path[:0] = ["/home/intel/intel-junction-2026/app/tools/tests", "/home/intel/intel-junction-2026/app/tools",
                "/home/intel/intel-junction-2026/app/anomaly/det2_f2"]
import conftest as C  # noqa: E402
import handdet as hd  # noqa: E402
import pick_any as pa  # noqa: E402
import sort_any as sa  # noqa: E402
import cv2  # noqa: E402


def timed(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts), max(ts)


print("== handdet.is_hand on fixtures/hand (holding=False), per view")
by_view = {}
for p in sorted(glob.glob(os.path.join(C.FIX, "hand", "*"))):
    view = os.path.basename(p).split("_")[1]
    by_view.setdefault(view, []).append(cv2.imread(p))
for view, imgs in sorted(by_view.items()):
    meds = [timed(lambda: hd.is_hand(img, view, holding=False), 3)[0] for img in imgs]
    print(f"view={view} n={len(imgs)} shape={imgs[0].shape} median_ms={statistics.median(meds):.1f} "
          f"min_ms={min(meds):.1f} max_ms={max(meds):.1f}")

print("== handdet.is_hand holding=True, views top and close")
for view in ("top", "close"):
    imgs = by_view.get(view, [])
    meds = [timed(lambda: hd.is_hand(img, view, holding=True), 3)[0] for img in imgs]
    if meds:
        print(f"view={view} n={len(meds)} median_ms={statistics.median(meds):.1f} max_ms={max(meds):.1f}")

print("== pick_any.blobs(min_area=1500) on hover, servo and split frames")
frames = [C.load_img(f) for f, *_ in C.HOVER] + [C.load_img(r[0]) for r in C.SERVO] + [C.load_img(r) for r in C.SPLIT]
meds = [timed(lambda: pa.blobs(img, min_area=1500), 3)[0] for img in frames]
print(f"frames={len(frames)} shape={frames[0].shape} median_ms={statistics.median(meds):.1f} "
      f"min_ms={min(meds):.1f} max_ms={max(meds):.1f}")

print("== pick_any.find on hover frames (colour lock, 20 reps)")
fm = []
for f, colour, *_ in C.HOVER:
    bl = pa.blobs(C.load_img(f), min_area=1500)
    fm.append(timed(lambda: pa.find(bl, pa.target(), 260, colour), 20)[0])
print(f"find median_ms={statistics.median(fm):.3f} max_ms={max(fm):.3f}")

print("== sort_any.candidates on scan_now.jpg (replays reach IK per blob)")
calib = C.load_json("scan_calib.json")
scan = C.load_img("scan_now.jpg")
m, mx = timed(lambda: sa.candidates(scan, calib["q_scan"], calib), 3)
print(f"candidates median_ms={m:.1f} max_ms={mx:.1f} shape={scan.shape}")
