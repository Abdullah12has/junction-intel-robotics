#!/usr/bin/env python3
"""Contact sheet for one curated dataset: top-view frames at start, grasp, release and end of each kept episode.

Usage: python3 -I verify_sheet.py <src_dataset_dir> <grip_json> <analysis_json> <dataset_name> <out_jpg> <camera_key>
Frames come from the source dataset (episode-local frame numbers): start = cut_start, grasp = closure + 10,
release = opening + 6, end = cut_end - 1.
"""
import glob
import json
import os
import subprocess
import sys
import tempfile
import textwrap

import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

FPS = 30.0
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
MEMBERS = {"PP-BLUE": [11, 12], "PP-GREEN": [15], "PP-BOTH": [11, 12, 15]}
TEXT = {
    11: "move the red brick to the blue plate",
    12: "move the red brick to the blue plate",
    15: "move the red brick to the green plate",
}
VERDICT = {
    11: "KEEP. Starts at rest, one grasp (hold 2.16), release on BLUE (pan +31.8), ends 5.4 units from rest. "
        "Brick red. Hand only at top-left edge over the laptop, off the mat.",
    12: "KEEP. Starts at rest, one grasp (hold 2.29), release on BLUE (pan +35.6), ends 9.9 units from rest "
        "still settling. Brick red. Hand only at top-left edge.",
    15: "KEEP. Starts at rest, one grasp (hold 2.22), release on GREEN (pan -53.3), ends 4.7 units from rest "
        "still settling. Brick red. Hand only at top-left edge.",
}


def main():
    src, grip_path, analysis_path, name, out_jpg, cam = sys.argv[1:7]
    eps = MEMBERS[name]
    with open(grip_path, encoding="utf-8") as fh:
        grip = json.load(fh)["episodes"]
    with open(analysis_path, encoding="utf-8") as fh:
        analysis = {int(r["episode"]): r for r in json.load(fh)}
    rows = []
    for path in sorted(glob.glob(src + "/meta/episodes/chunk-*/file-*.parquet")):
        rows += pq.read_table(path).to_pylist()
    meta = {int(r["episode_index"]): r for r in rows}

    W, H, PAD, LABEL_W, HEAD = 320, 240, 6, 470, 40
    frame_titles = ["start", "grasp (+10 f)", "release (+6 f)", "end (-5 f)"]
    font = ImageFont.truetype(FONT, 16)
    small = ImageFont.truetype(FONT, 14)
    img = Image.new("RGB", (LABEL_W + 4 * (W + PAD), HEAD + len(eps) * (H + PAD)), "white")
    draw = ImageDraw.Draw(img)
    for i, title in enumerate(frame_titles):
        draw.text((LABEL_W + i * (W + PAD) + 4, 10), title, fill="black", font=font)
    tmp = tempfile.mkdtemp(prefix="verify-sheet-")
    for r, ep in enumerate(eps):
        g = grip[str(ep)]
        a = analysis[ep]
        m = meta[ep]
        chunk = int(m[f"videos/{cam}/chunk_index"])
        fidx = int(m[f"videos/{cam}/file_index"])
        t0 = float(m[f"videos/{cam}/from_timestamp"])
        video = f"{src}/videos/{cam}/chunk-{chunk:03d}/file-{fidx:03d}.mp4"
        frames = [a["cut_start"], g["grasp"] + 10, g["release"] + 6, a["cut_end"] - 5]
        y = HEAD + r * (H + PAD)
        for c, f in enumerate(frames):
            png = os.path.join(tmp, f"ep{ep}_{c}.png")
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0 + f / FPS:.4f}", "-i", video,
                 "-frames:v", "1", "-vf", f"scale={W}:{H}", png],
                check=True,
            )
            x = LABEL_W + c * (W + PAD)
            img.paste(Image.open(png).convert("RGB"), (x, y))
            draw.rectangle([x, y + H - 20, x + W, y + H], fill="black")
            draw.text((x + 4, y + H - 19), f"ep{ep} f{f} {f / FPS:.1f}s", fill="yellow", font=small)
        lines = [f"ep{ep:02d}  {TEXT[ep]}", f"release pan {g['pan_release']}  dest {g['dest']}"]
        lines += textwrap.wrap(VERDICT[ep], width=62)
        ty = y + 2
        for k, line in enumerate(lines):
            draw.text((6, ty), line, fill="black", font=font if k == 0 else small)
            ty += 22 if k == 0 else 17
    img.save(out_jpg, quality=90)
    print("wrote", out_jpg, "episodes", eps, "size", img.size)


if __name__ == "__main__":
    main()
