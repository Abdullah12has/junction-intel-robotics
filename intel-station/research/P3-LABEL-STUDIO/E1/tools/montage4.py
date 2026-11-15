#!/usr/bin/env python3
"""Fourth batch of full-resolution views, as 2x2 grids (frame index = episode frame at 30 fps; each panel is labelled).
U19 white brick at the start and at the release; U20 after the gripper opened (release destination, top and close);
U24 red brick over the blue plate after the drop; U21 blue brick at the start and at the release (top and close)."""
import json
import os
import subprocess

from PIL import Image, ImageDraw

WD = "/home/intel/intel-junction-2026/work/P3-LABEL-STUDIO/E1"
FRD = os.path.join(WD, "frames")
ZD = os.path.join(WD, "zoom")
os.makedirs(FRD, exist_ok=True)
os.makedirs(ZD, exist_ok=True)
UNIQ = {u["uid"]: u for u in json.load(open(os.path.join(WD, "out", "unique_episodes.json")))}


def frame(uid, cam, idx):
    path = os.path.join(FRD, "%s_%s_x_f%04d.png" % (uid, cam, idx))
    if not os.path.exists(path):
        u = UNIQ[uid]
        ci = [c for c in u["cams"] if c and c["cam"] == cam][0]
        t = ci["from_ts"] + idx / 30.0
        subprocess.run(["ffmpeg", "-v", "error", "-ss", "%.4f" % t, "-i", ci["mp4"], "-frames:v", "1",
                        "-y", path], capture_output=True, text=True)
    return path if os.path.exists(path) else None


def grid(name, specs, cols=2):
    panels = []
    for uid, cam, idx in specs:
        p = frame(uid, cam, idx)
        if p is None:
            print("MISSING", uid, cam, idx, flush=True)
            continue
        im = Image.open(p).convert("RGB")
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 300, 22], fill=(0, 0, 0))
        d.text((4, 5), "%s cam %s f%d (%.2f s)" % (uid, cam, idx, idx / 30.0), fill=(255, 255, 0))
        panels.append(im)
    if not panels:
        return
    w0, h0 = panels[0].size
    rows = (len(panels) + cols - 1) // cols
    out = Image.new("RGB", (cols * w0 + (cols - 1) * 10, rows * h0 + (rows - 1) * 10), (255, 255, 255))
    for k, im in enumerate(panels):
        r, c = divmod(k, cols)
        out.paste(im, (c * (w0 + 10), r * (h0 + 10)))
    out.save(os.path.join(ZD, name + ".jpg"), "JPEG", quality=92)
    print(name, out.size, len(panels), flush=True)


GRIDS = [
    ("Z19_U19_top_0_100_140_160", [("U19", "061421", 0), ("U19", "061421", 100),
                                   ("U19", "061421", 140), ("U19", "061421", 160)]),
    ("Z20_U20_top_200_230_260_290", [("U20", "061421", 200), ("U20", "061421", 230),
                                     ("U20", "061421", 260), ("U20", "061421", 290)]),
    ("Z21_U20_close_200_230_260_290", [("U20", "064316", 200), ("U20", "064316", 230),
                                       ("U20", "064316", 260), ("U20", "064316", 290)]),
    ("Z22_U24_top_600_620_640_650", [("U24", "061421", 600), ("U24", "061421", 620),
                                     ("U24", "061421", 640), ("U24", "061421", 650)]),
    ("Z23_U21_top_0_60_120_150", [("U21", "061421", 0), ("U21", "061421", 60),
                                  ("U21", "061421", 120), ("U21", "061421", 150)]),
    ("Z24_U21_close_0_60_120_150", [("U21", "064316", 0), ("U21", "064316", 60),
                                    ("U21", "064316", 120), ("U21", "064316", 150)]),
]

for name, specs in GRIDS:
    grid(name, specs)
