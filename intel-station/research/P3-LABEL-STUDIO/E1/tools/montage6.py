#!/usr/bin/env python3
"""Sixth batch: 3x crops of the top view around the gripper at the start of U19, U20 and U21 (which brick is held,
and where it goes), a 3x crop of the U20 release area, and 5x crops of the green plate at the start and end of U20
(second white brick or not). Frame index = episode frame at 30 fps."""
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


def crop_panel(uid, cam, idx, box, scale):
    p = frame(uid, cam, idx)
    if p is None:
        print("MISSING", uid, cam, idx, flush=True)
        return None
    im = Image.open(p).convert("RGB").crop(box)
    im = im.resize((im.size[0] * scale, im.size[1] * scale), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 260, 20], fill=(0, 0, 0))
    d.text((3, 4), "%s cam %s f%d (%.2f s) crop %dx" % (uid, cam, idx, idx / 30.0, scale), fill=(255, 255, 0))
    return im


def hstack(panels, name):
    panels = [p for p in panels if p is not None]
    if not panels:
        return
    w = sum(p.size[0] for p in panels) + 10 * (len(panels) - 1)
    h = max(p.size[1] for p in panels)
    out = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.size[0] + 10
    out.save(os.path.join(ZD, name + ".jpg"), "JPEG", quality=95)
    print(name, out.size, len(panels), flush=True)


GRIP = (300, 130, 480, 310)      # gripper and the mat below it (top view)
GRIP_B = (300, 200, 480, 380)    # U20 start: the white brick under the gripper sits lower
RELEASE = (480, 20, 640, 200)    # U20 release area at the top-right of the green plate
GREEN_END = (530, 120, 640, 210) # green plate, white and blue bricks (top view)

hstack([crop_panel("U19", "061421", 0, GRIP, 3),
        crop_panel("U19", "061421", 60, GRIP, 3),
        crop_panel("U19", "061421", 100, GRIP, 3)], "Z35_U19_gripper_f0_f60_f100")
hstack([crop_panel("U20", "061421", 0, GRIP_B, 3),
        crop_panel("U20", "061421", 60, GRIP_B, 3),
        crop_panel("U20", "061421", 120, RELEASE, 3),
        crop_panel("U20", "061421", 190, RELEASE, 3)], "Z36_U20_gripper_f0_f60_f120_f190")
hstack([crop_panel("U21", "061421", 0, GRIP, 3),
        crop_panel("U21", "061421", 30, GRIP, 3),
        crop_panel("U21", "061421", 60, GRIP, 3)], "Z37_U21_gripper_f0_f30_f60")
hstack([crop_panel("U20", "061421", 0, GREEN_END, 5),
        crop_panel("U20", "061421", UNIQ["U20"]["n_frames"] - 3, GREEN_END, 5)], "Z38_U20_green_start_and_end_5x")
