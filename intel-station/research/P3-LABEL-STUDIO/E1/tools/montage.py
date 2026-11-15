#!/usr/bin/env python3
"""Full-resolution side-by-side montages (two frames per image) for close checks of bricks, tape and placements.

Frames come from frames/ (extracted by build_sheets.py); missing frames are extracted from the mp4 with ffmpeg.
"""
import json
import os
import subprocess

from PIL import Image

WD = "/home/intel/intel-junction-2026/work/P3-LABEL-STUDIO/E1"
FRD = os.path.join(WD, "frames")
ZD = os.path.join(WD, "zoom")
os.makedirs(ZD, exist_ok=True)
UNIQ = {u["uid"]: u for u in json.load(open(os.path.join(WD, "out", "unique_episodes.json")))}


def frame(uid, cam, label, idx):
    name = "%s_%s_%s_f%04d.png" % (uid, cam, label, idx)
    path = os.path.join(FRD, name)
    if not os.path.exists(path):
        u = UNIQ[uid]
        ci = [c for c in u["cams"] if c and c["cam"] == cam][0]
        t = ci["from_ts"] + idx / 30.0
        subprocess.run(["ffmpeg", "-v", "error", "-ss", "%.4f" % t, "-i", ci["mp4"], "-frames:v", "1",
                        "-y", path], capture_output=True, text=True)
    return path


PAIRS = [
    ("Z1_U04_U05_close_start", [("U04", "064316", "start", 0), ("U05", "064316", "start", 0)]),
    ("Z2_U11_U19_close_start", [("U11", "064316", "start", 0), ("U19", "064316", "start", 0)]),
    ("Z3_U01_U24_close_start", [("U01", "064316", "start", 0), ("U24", "064316", "start", 0)]),
    ("Z4_U19_U20_top_start", [("U19", "061421", "start", 0), ("U20", "061421", "start", 0)]),
    ("Z5_U24_top_release_end", [("U24", "061421", "release", 534), ("U24", "061421", "end", 657)]),
    ("Z6_U10_end_top_close", [("U10", "061421", "end", 613), ("U10", "064316", "end", 613)]),
]

for name, specs in PAIRS:
    imgs = [Image.open(frame(*s)).convert("RGB") for s in specs]
    w = sum(i.size[0] for i in imgs) + 10 * (len(imgs) - 1)
    h = max(i.size[1] for i in imgs)
    out = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for i in imgs:
        out.paste(i, (x, 0))
        x += i.size[0] + 10
    out.save(os.path.join(ZD, name + ".jpg"), "JPEG", quality=92)
    print(name, out.size, [("%s %s %s f%d" % s) for s in specs])
