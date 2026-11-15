import json, os, subprocess
from PIL import Image, ImageDraw

R = "/home/intel/intel-junction-2026"
WD = os.path.join(R, "work/P3-LABEL-STUDIO/E1")
OUTD = os.path.join(WD, "out")
FRD = os.path.join(WD, "frames", "p17")
SD = os.path.join(R, "data", "studio-datasets")
SERIALS = {"top": "243622061421", "close": "243722064316"}
FPS = 30
os.makedirs(FRD, exist_ok=True)

e = None
for line in open(os.path.join(OUTD, "episodes_census.jsonl"), encoding="utf-8"):
    x = json.loads(line)
    if x["dataset"].startswith("7e99a9e4"):
        e = x
print("entry", e["dataset"][:8], e["episode_index"], "n_frames", e["n_frames"], "dur", e["dur_s"])
IDXS = [0, 2800, 5600, 8000]
W, H = 320, 240
canvas = Image.new("RGB", (W * len(IDXS), H * 2), (255, 255, 255))
draw = ImageDraw.Draw(canvas)
for ri, cam in enumerate(("top", "close")):
    k = [x for x in e["video_segments"] if x.endswith(SERIALS[cam])][0]
    s = e["video_segments"][k]
    mp4 = os.path.join(SD, e["dataset"], s["mp4"])
    for ci, idx in enumerate(IDXS):
        out = os.path.join(FRD, "7e99a9e4_%s_f%04d.png" % (cam, idx))
        if not os.path.exists(out):
            subprocess.run(["ffmpeg", "-v", "error", "-ss", "%.4f" % (s["from_ts"] + idx / FPS), "-i", mp4,
                            "-frames:v", "1", "-y", out], capture_output=True, text=True)
        im = Image.open(out).convert("RGB").resize((W, H))
        canvas.paste(im, (ci * W, ri * H))
        lab = "7e99a9e4:0 %s f%d (%.1f s)" % (cam, idx, idx / FPS)
        draw.rectangle([ci * W, ri * H, ci * W + 200, ri * H + 14], fill=(0, 0, 0))
        draw.text((ci * W + 3, ri * H + 2), lab, fill=(255, 255, 255))
canvas.save(os.path.join(OUTD, "p17_montage.jpg"), quality=85)
print("written", canvas.size)
