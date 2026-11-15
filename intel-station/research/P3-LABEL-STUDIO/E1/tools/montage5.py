#!/usr/bin/env python3
"""Fifth batch of full-resolution views, as 2x2 grids and 3x crops (frame index = episode frame at 30 fps).
U20 start (top and close); last frames of U19, U20, U21 and U24 (top and close); crops of the green plate
(U19, U20, U21 last frame) and of the blue plate (U24). The last index is n_frames - 1 and ffmpeg can fail on it,
so end frames use n_frames - k with k >= 2."""
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

for uid in ["U19", "U20", "U21", "U24"]:
    u = UNIQ[uid]
    print(uid, "n_frames", u["n_frames"], "dur_s", u["dur_s"], "tasks", u["tasks"],
          "members", [(m["ds"], m["ep"]) for m in u["members"]], flush=True)


def frame(uid, cam, idx):
    path = os.path.join(FRD, "%s_%s_x_f%04d.png" % (uid, cam, idx))
    if not os.path.exists(path):
        u = UNIQ[uid]
        ci = [c for c in u["cams"] if c and c["cam"] == cam][0]
        t = ci["from_ts"] + idx / 30.0
        subprocess.run(["ffmpeg", "-v", "error", "-ss", "%.4f" % t, "-i", ci["mp4"], "-frames:v", "1",
                        "-y", path], capture_output=True, text=True)
    return path if os.path.exists(path) else None


def end(uid, k):
    return UNIQ[uid]["n_frames"] - k


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


def crop_panel(uid, cam, idx, box, scale):
    p = frame(uid, cam, idx)
    if p is None:
        print("MISSING", uid, cam, idx, flush=True)
        return None
    im = Image.open(p).convert("RGB").crop(box)
    im = im.resize((im.size[0] * scale, im.size[1] * scale), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 240, 20], fill=(0, 0, 0))
    d.text((3, 4), "%s cam %s f%d crop %dx" % (uid, cam, idx, scale), fill=(255, 255, 0))
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


GREEN_BOX = (470, 60, 640, 290)   # top view, green plate on the right
BLUE_BOX = (140, 80, 300, 240)    # top view, blue plate on the left

GRIDS = [
    ("Z25_U20_top_start_0_60_120_180", [("U20", "061421", 0), ("U20", "061421", 60),
                                        ("U20", "061421", 120), ("U20", "061421", 180)]),
    ("Z26_U20_close_start_0_60_120_180", [("U20", "064316", 0), ("U20", "064316", 60),
                                          ("U20", "064316", 120), ("U20", "064316", 180)]),
    ("Z27_U24_top_end", [("U24", "061421", end("U24", 20)), ("U24", "061421", end("U24", 10)),
                         ("U24", "061421", end("U24", 5)), ("U24", "061421", end("U24", 2))]),
    ("Z28_U21_top_end", [("U21", "061421", end("U21", 20)), ("U21", "061421", end("U21", 10)),
                         ("U21", "061421", end("U21", 5)), ("U21", "061421", end("U21", 2))]),
    ("Z29_U21_close_end", [("U21", "064316", end("U21", 20)), ("U21", "064316", end("U21", 10)),
                           ("U21", "064316", end("U21", 5)), ("U21", "064316", end("U21", 2))]),
    ("Z30_U19_top_end", [("U19", "061421", end("U19", 20)), ("U19", "061421", end("U19", 10)),
                         ("U19", "061421", end("U19", 5)), ("U19", "061421", end("U19", 2))]),
    ("Z31_U20_top_end", [("U20", "061421", end("U20", 20)), ("U20", "061421", end("U20", 10)),
                         ("U20", "061421", end("U20", 5)), ("U20", "061421", end("U20", 2))]),
]

for name, specs in GRIDS:
    grid(name, specs)

hstack([crop_panel("U19", "061421", end("U19", 3), GREEN_BOX, 3),
        crop_panel("U20", "061421", end("U20", 3), GREEN_BOX, 3),
        crop_panel("U21", "061421", end("U21", 3), GREEN_BOX, 3)], "Z32_green_plate_end_crops_U19_U20_U21")
hstack([crop_panel("U21", "061421", 150, GREEN_BOX, 3),
        crop_panel("U21", "061421", end("U21", 3), GREEN_BOX, 3)], "Z33_green_plate_U21_f150_and_end")
hstack([crop_panel("U24", "061421", 600, BLUE_BOX, 3),
        crop_panel("U24", "061421", end("U24", 3), BLUE_BOX, 3)], "Z34_blue_plate_U24_f600_and_end")
