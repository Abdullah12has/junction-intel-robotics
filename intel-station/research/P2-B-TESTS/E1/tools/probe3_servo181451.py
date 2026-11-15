"""Probe of builder E1 (L37): provenance of fixture servo/servo_181451_red.png (log second 18:14:51Z).

Read-only on the recorded data. Decodes frames with the same ffmpeg call as extract_frames.py (grab) and writes
one candidate PNG into the E1 raw folder only. Nothing under app/ or the fixtures folder is written here.
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/intel/intel-junction-2026/app/tools")
sys.path.insert(0, "/home/intel/intel-junction-2026/app/anomaly/det2_f2")
import pick_any as pa  # noqa: E402

RAW = "/home/intel/intel-junction-2026/data/wrist/raw"
FIX = "/home/intel/intel-junction-2026/app/tools/tests/fixtures/servo"
OUT = "/home/intel/intel-junction-2026/work/P2-B-TESTS/E1/raw"
SEG_1813 = os.path.join(RAW, "w_20261114T181351Z.mkv")
SEG_1814 = os.path.join(RAW, "w_20261114T181451Z.mkv")
SEG_1819 = os.path.join(RAW, "w_20261114T181951Z.mkv")
EPOCH_1814_51 = 1794680091  # 18:14:51 UTC, log second of servo_181451
EPOCH_1814_54 = 1794680094  # 18:14:54 UTC, log second of servo_181454 (control)
LOG_CENTRE = (420.2, 346.7)  # logged centre of servo_181451 (sort_any_any_1740.out line 277)
LOG_SHORT = 108.8


def start_time(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=start_time", "-of", "csv=p=0", path],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def grab(path, off):
    """Same ffmpeg call as extract_frames.grab: input seek to off seconds, one frame, PNG on a pipe."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{off:.3f}", "-i", path,
                        "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1"],
                       capture_output=True, check=True)
    return cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)


def diff(a, b):
    if a is None or b is None or a.shape != b.shape:
        return f"shape mismatch {None if a is None else a.shape} {None if b is None else b.shape}"
    d = np.abs(a.astype(int) - b.astype(int))
    return f"max_abs {int(d.max())} mean_abs {d.mean():.4f}"


def match(img):
    """Same scoring as extract_frames.main: pa.find on blobs(min_area=SERVO_MIN_AREA), seeded at the logged centre."""
    b = pa.find(pa.blobs(img, min_area=pa.SERVO_MIN_AREA), np.array(LOG_CENTRE), 140, "red")
    if b is None:
        return None
    dist = math.hypot(b["centre"][0] - LOG_CENTRE[0], b["centre"][1] - LOG_CENTRE[1])
    return b, dist, b["short"] - LOG_SHORT, dist + 0.1 * abs(b["short"] - LOG_SHORT)


def main():
    st13, st14, st19 = start_time(SEG_1813), start_time(SEG_1814), start_time(SEG_1819)
    print("format start_time: w_181351Z", st13, "w_181451Z", st14, "w_181951Z", st19, flush=True)
    fix51 = cv2.imread(os.path.join(FIX, "servo_181451_red.png"))
    fix54 = cv2.imread(os.path.join(FIX, "servo_181454_red.png"))
    fix17 = cv2.imread(os.path.join(FIX, "servo_182017_white.png"))
    print("control 1: servo_181454 vs w_181451Z @2.54 (offset used in L19):", diff(grab(SEG_1814, 2.54), fix54), flush=True)
    print("control 2: servo_182017 vs w_181951Z @25.738 (offset used in L19):", diff(grab(SEG_1819, 25.738), fix17), flush=True)
    neg = grab(SEG_1814, -1.26)
    zero = grab(SEG_1814, 0.0)
    print("w_181451Z @-1.26 vs @0.0:", diff(neg, zero), flush=True)
    print("fixture servo_181451 vs w_181451Z @-1.26 (offset recorded in L19):", diff(fix51, neg), flush=True)
    print("fixture servo_181451 vs w_181451Z @0.0 (first frame of the segment):", diff(fix51, zero), flush=True)
    off_prev = EPOCH_1814_51 - 0.5 - st13
    print("fixture servo_181451 vs w_181351Z @%.3f (18:14:50.50):" % off_prev,
          diff(fix51, grab(SEG_1813, off_prev)), flush=True)
    m = match(fix51)
    print("fixture servo_181451 metric (L20: centre 419.5,346.3 dist 0.8):",
          None if m is None else (m[0]["centre"], m[0]["short"], round(m[1], 2)), flush=True)
    base = EPOCH_1814_51 - st13
    print("window search in w_181351Z, nominal offset %.3f; columns: k dk offset_s nominal_utc centre short dist short_diff score"
          % base, flush=True)
    best = None
    for k in range(11):
        dk = round(-0.5 + 0.1 * k, 1)
        off = base + dk
        img = grab(SEG_1813, off)
        m = match(img)
        frame_t = EPOCH_1814_51 + dk
        if m is None:
            print(k, dk, f"{off:.3f}", f"{frame_t:.2f}", "no blob", flush=True)
            continue
        b, dist, ds, score = m
        print(k, dk, f"{off:.3f}", f"{frame_t:.2f}", b["centre"], b["short"], round(dist, 2), round(ds, 1),
              round(score, 2), flush=True)
        if best is None or score < best[0]:
            best = (score, dk, off, img, b)
    if best is None:
        print("no candidate", flush=True)
        return
    score, dk, off, img, b = best
    path = os.path.join(OUT, "L37_servo181451_candidate.png")
    cv2.imwrite(path, img)
    print("best dk", dk, "offset", f"{off:.3f}", "centre", b["centre"], "short", b["short"], "score", round(score, 2),
          "written", path, flush=True)
    print("best candidate vs current fixture:", diff(img, fix51), flush=True)


if __name__ == "__main__":
    main()
