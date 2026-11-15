#!/usr/bin/env python3
"""Second batch of full-resolution side-by-side montages (tape, held brick, placement checks)."""
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
    ("Z7_U25_U09_close_start", [("U25", "064316", "start", 0), ("U09", "064316", "start", 0)]),
    ("Z8_U19_U20_top_end", [("U19", "061421", "end", 341), ("U20", "061421", "end", 339)]),
    ("Z9_U21_top_start_end", [("U21", "061421", "start", 0), ("U21", "061421", "end", 338)]),
    ("Z10_U24_carry_top_close", [("U24", "061421", "carry", 429), ("U24", "064316", "carry", 429)]),
    ("Z11_U12_U13_close_start", [("U12", "064316", "start", 0), ("U13", "064316", "start", 0)]),
    ("Z12_U02_U07_close_start", [("U02", "064316", "start", 0), ("U07", "064316", "start", 0)]),
    ("Z13_U22_U27_close_start", [("U22", "064316", "start", 0), ("U27", "064316", "start", 0)]),
    ("Z14_U03_U16_close_start", [("U03", "064316", "start", 0), ("U16", "064316", "start", 0)]),
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
