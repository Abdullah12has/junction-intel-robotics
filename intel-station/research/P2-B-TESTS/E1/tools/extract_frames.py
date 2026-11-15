"""Scratch tool of builder E1 (not part of the repo tests). Extracts wrist frames for the replay fixtures.

Servo frames: the segment is the last data/wrist/raw file starting at or before the logged UTC second; ffmpeg -ss is
(epoch of the log time - the segment format start_time); offsets -0.5..+0.5 s in 0.1 s steps are scored with
pick_any.find (same colour class, max_d 140 as in servo()) against the logged centre and short. Best offset is kept.
Split frames: 18:16:10..18:16:30 every 2 s from the same segment family, saved without matching.
Read-only on the repo data; writes PNG only into the tests fixtures folder and a TSV into the scratch raw folder.
"""
import calendar
import glob
import math
import os
import re
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/intel/intel-junction-2026/app/tools")
sys.path.insert(0, "/home/intel/intel-junction-2026/app/anomaly/det2_f2")
import pick_any as pa  # noqa: E402

RAW = "/home/intel/intel-junction-2026/data/wrist/raw"
FIX = "/home/intel/intel-junction-2026/app/tools/tests/fixtures"
OUT = "/home/intel/intel-junction-2026/work/P2-B-TESTS/E1/raw/L19_extract_summary.tsv"

# (UTC time, colour, logged centre x, y, logged short, log line number in sort_any_any_1740.out)
SERVO = [
    ("18:00:02", "red", 422.7, 268.8, 112.6, 104),
    ("18:00:15", "red", 407.5, 268.0, 117.0, 110),
    ("18:13:48", "white", 397.8, 287.9, 109.2, 267),
    ("18:13:52", "white", 408.8, 268.4, 113.6, 269),
    ("18:13:56", "white", 404.5, 274.5, 107.0, 272),
    ("18:14:51", "red", 420.2, 346.7, 108.8, 277),
    ("18:14:54", "red", 435.9, 287.8, 116.0, 278),
    ("18:14:56", "red", 408.6, 270.2, 113.3, 279),
    ("18:15:00", "red", 365.4, 297.6, 109.2, 281),
    ("18:15:02", "red", 403.4, 268.4, 110.9, 282),
    ("18:20:06", "white", 466.0, 394.1, 71.4, 324),
    ("18:20:10", "white", 488.8, 305.5, 92.6, 325),
    ("18:20:13", "white", 406.5, 260.8, 121.2, 326),
    ("18:20:17", "white", 402.2, 267.7, 120.1, 327),
    ("18:20:19", "white", 406.9, 257.5, 113.8, 329),
    ("18:20:21", "white", 390.5, 263.0, 118.0, 330),
    ("18:20:25", "white", 405.0, 270.5, 117.0, 331),
]
SPLIT = ["18:16:%02d" % s for s in range(10, 31, 2)]


def epoch(hms):
    h, m, s = (int(x) for x in hms.split(":"))
    return calendar.timegm((2026, 11, 14, h, m, s, 0, 0, 0))


def segments():
    out = []
    for p in glob.glob(os.path.join(RAW, "w_20261114T*Z.mkv")):
        m = re.search(r"T(\d{2})(\d{2})(\d{2})Z", os.path.basename(p))
        t = calendar.timegm((2026, 11, 14, int(m.group(1)), int(m.group(2)), int(m.group(3)), 0, 0, 0))
        out.append((t, p))
    return sorted(out)


def seg_for(hms):
    t = epoch(hms)
    ts, p = max((ts, p) for ts, p in segments() if ts <= t)
    st = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=start_time", "-of", "csv=p=0", p],
                              capture_output=True, text=True, check=True).stdout.strip())
    return p, st, t - st


def grab(path, off):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{off:.3f}", "-i", path,
                        "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1"],
                       capture_output=True, check=True)
    return cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)


def main():
    rows = []
    for hms, colour, cx, cy, short, line in SERVO:
        path, st, base = seg_for(hms)
        best = None
        for k in range(11):
            dk = round(-0.5 + 0.1 * k, 1)
            off = base + dk
            img = grab(path, off)
            b = pa.find(pa.blobs(img, min_area=pa.SERVO_MIN_AREA), np.array([cx, cy]), 140, colour)
            if b is None:
                continue
            score = math.hypot(b["centre"][0] - cx, b["centre"][1] - cy) + 0.1 * abs(b["short"] - short)
            if best is None or score < best[0]:
                best = (score, dk, off, b, img)
        if best is None:
            print("NO MATCH", hms, colour, flush=True)
            continue
        score, dk, off, b, img = best
        name = f"servo_{hms.replace(':', '')}_{colour}.png"
        cv2.imwrite(os.path.join(FIX, "servo", name), img)
        rows.append([name, hms, colour, cx, cy, short, line, os.path.basename(path), round(st, 6), round(off, 3), dk,
                     b["centre"][0], b["centre"][1], b["short"], round(math.hypot(b["centre"][0] - cx, b["centre"][1] - cy), 1),
                     round(b["short"] - short, 1)])
        print("servo", name, "dk", dk, "match", b["centre"], b["short"], "dist", rows[-1][14], flush=True)
    for hms in SPLIT:
        path, st, base = seg_for(hms)
        img = grab(path, base)
        name = f"split_{hms.replace(':', '')}.png"
        cv2.imwrite(os.path.join(FIX, "split", name), img)
        whites = [(b["centre"], b["short"], b["area"]) for b in pa.blobs(img, min_area=1500) if b["colour"] == "white"]
        rows.append([name, hms, "white", "", "", "", "", os.path.basename(path), round(st, 6), round(base, 3), "",
                     "", "", "", "", "", str(whites)])
        print("split", name, "white blobs", whites, flush=True)
    with open(OUT, "w") as f:
        f.write("\t".join(["file", "utc", "colour", "logged_x", "logged_y", "logged_short", "log_line", "segment",
                           "format_start_time", "offset_used_s", "dk_s", "match_x", "match_y", "match_short",
                           "centre_dist_px", "short_diff", "white_blobs_or_note"]) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print("wrote", OUT, len(rows), "rows", flush=True)


if __name__ == "__main__":
    main()
