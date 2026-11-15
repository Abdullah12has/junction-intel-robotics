#!/usr/bin/env python3
"""Third batch of full-resolution montages: frames at the grip-release moments of the 'left plate' episodes and the U24 drop."""
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
    ("Z15_U19_U20_top_release", [("U19", "061421", "x", 150), ("U20", "061421", "x", 190)]),
    ("Z16_U19_U20_close_release", [("U19", "064316", "x", 150), ("U20", "064316", "x", 190)]),
    ("Z17_U21_top_close_release", [("U21", "061421", "x", 141), ("U21", "064316", "x", 141)]),
    ("Z18_U24_top_after_drop", [("U24", "061421", "x", 550), ("U24", "061421", "x", 600)]),
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
    print(name, out.size, [("%s %s f%d" % (s[0], s[1], s[3])) for s in specs])
