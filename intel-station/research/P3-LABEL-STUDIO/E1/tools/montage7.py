#!/usr/bin/env python3
"""Seventh batch: last frames (n_frames - 3) of the four placements that were not checked at their final frame:
U14 (P7 spot), U16 (P9 spot), U10 (P8 green, final frame previously unverified), U25 (P6 green). Top view, 2x2 grid."""
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


panels = []
for uid in ["U14", "U16", "U10", "U25"]:
    idx = UNIQ[uid]["n_frames"] - 3
    p = frame(uid, "061421", idx)
    if p is None:
        print("MISSING", uid, idx, flush=True)
        continue
    im = Image.open(p).convert("RGB")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 300, 22], fill=(0, 0, 0))
    d.text((4, 5), "%s cam 061421 f%d (%.2f s) end-3" % (uid, idx, idx / 30.0), fill=(255, 255, 0))
    panels.append(im)
    print(uid, "n_frames", UNIQ[uid]["n_frames"], "frame", idx, flush=True)

w0, h0 = panels[0].size
out = Image.new("RGB", (2 * w0 + 10, 2 * h0 + 10), (255, 255, 255))
for k, im in enumerate(panels):
    r, c = divmod(k, 2)
    out.paste(im, (c * (w0 + 10), r * (h0 + 10)))
out.save(os.path.join(ZD, "Z39_end_frames_U14_U16_U10_U25.jpg"), "JPEG", quality=92)
print("Z39_end_frames_U14_U16_U10_U25", out.size, len(panels), flush=True)
