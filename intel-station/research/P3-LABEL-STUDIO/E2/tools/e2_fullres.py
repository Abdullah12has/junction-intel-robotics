import glob, os, subprocess, sys
import pandas as pd
from PIL import Image, ImageDraw

R = "/home/intel/intel-junction-2026"
WD = os.path.join(R, "work/P3-LABEL-STUDIO/E2")
SD = os.path.join(R, "data", "studio-datasets")
FRD = os.path.join(WD, "frames")
os.makedirs(FRD, exist_ok=True)
FPS = 30
prefix = sys.argv[1]
eps = [int(x) for x in sys.argv[2].split(",")]
ddir = [d for d in os.listdir(SD) if d.startswith(prefix)][0]
base = os.path.join(SD, ddir)
meta = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(base, "meta/episodes/chunk-*/file-*.parquet")))], ignore_index=True)
cdir = "observation.images.realsense 243622061421"
for ep in eps:
    m = meta[meta.episode_index == ep].iloc[0]
    length = int(m["length"])
    fi = int(m["videos/%s/file_index" % cdir])
    ts0 = float(m["videos/%s/from_timestamp" % cdir])
    mp4 = os.path.join(base, "videos", cdir, "chunk-000", "file-%03d.mp4" % fi)
    panels = []
    for tag, idx in (("start", 30), ("end", length - 3)):
        out = os.path.join(FRD, "full_%s_ep%d_%s_f%04d.png" % (prefix, ep, tag, idx))
        if not os.path.exists(out):
            subprocess.run(["ffmpeg", "-v", "error", "-ss", "%.4f" % (ts0 + idx / FPS), "-i", mp4,
                            "-frames:v", "1", "-y", out], capture_output=True, text=True)
        panels.append((tag, idx, Image.open(out).convert("RGB")))
    W, H = panels[0][2].size
    canvas = Image.new("RGB", (W * 2, H + 20), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    for k, (tag, idx, im) in enumerate(panels):
        canvas.paste(im, (k * W, 20))
        d.text((k * W + 6, 4), "%s ep%d %s f%d (%.1fs) top cam" % (prefix, ep, tag, idx, idx / FPS), fill=(0, 0, 0))
    outp = os.path.join(WD, "out", "full_%s_ep%d.png" % (prefix, ep))
    canvas.save(outp)
    print("written", outp, canvas.size)
