import glob, json, os, subprocess, sys
import pandas as pd
from PIL import Image, ImageDraw

R = "/home/intel/intel-junction-2026"
WD = os.path.join(R, "work/P3-LABEL-STUDIO/E2")
SD = os.path.join(R, "data", "studio-datasets")
FRD = os.path.join(WD, "frames")
os.makedirs(FRD, exist_ok=True)
FPS = 30
CAMS = [("top", "observation.images.realsense 243622061421"),
        ("close", "observation.images.realsense 243722064316"),
        ("ext", "observation.images.usb2_0_cam1 external camera")]

prefix = sys.argv[1]                     # dataset uuid prefix, e.g. 8a8bb05e
ep = int(sys.argv[2])                    # episode index
nfr = int(sys.argv[3]) if len(sys.argv) > 3 else 8
ddir = [d for d in os.listdir(SD) if d.startswith(prefix)][0]
base = os.path.join(SD, ddir)
meta = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(base, "meta/episodes/chunk-*/file-*.parquet")))], ignore_index=True)
m = meta[meta.episode_index == ep].iloc[0]
length = int(m["length"])
W, H = 256, 192
rows = len(CAMS)
canvas = Image.new("RGB", (W * nfr, H * rows + 16), (255, 255, 255))
draw = ImageDraw.Draw(canvas)
idxs = [min(int(round(i * (length - 1) / (nfr - 1))), length - 3) for i in range(nfr)]
CAMS = [c for c in CAMS if os.path.isdir(os.path.join(base, "videos", c[1]))]
rows = len(CAMS)
canvas = Image.new("RGB", (W * nfr, H * rows + 16), (255, 255, 255))
draw = ImageDraw.Draw(canvas)
for ri, (cname, cdir) in enumerate(CAMS):
    fi = int(m["videos/%s/file_index" % cdir])
    ts0 = float(m["videos/%s/from_timestamp" % cdir])
    mp4 = os.path.join(base, "videos", cdir, "chunk-000", "file-%03d.mp4" % fi)
    for ci, idx in enumerate(idxs):
        out = os.path.join(FRD, "%s_ep%d_%s_f%04d.png" % (prefix, ep, cname, idx))
        if not os.path.exists(out):
            subprocess.run(["ffmpeg", "-v", "error", "-ss", "%.4f" % (ts0 + idx / FPS), "-i", mp4,
                            "-frames:v", "1", "-y", out], capture_output=True, text=True)
        im = Image.open(out).convert("RGB").resize((W, H))
        canvas.paste(im, (ci * W, 16 + ri * H))
        draw.rectangle([ci * W, 16 + ri * H, ci * W + 150, 16 + ri * H + 13], fill=(0, 0, 0))
        draw.text((ci * W + 3, 16 + ri * H + 1), "%s f%d %.1fs" % (cname, idx, idx / FPS), fill=(255, 255, 255))
draw.text((4, 2), "%s ep%d length %d frames (%.1f s), columns = frames at even spacing" % (prefix, ep, length, length / FPS), fill=(0, 0, 0))
outp = os.path.join(WD, "out", "mont_%s_ep%d.jpg" % (prefix, ep))
canvas.save(outp, quality=85)
print("written", outp, canvas.size, "idxs", idxs)
