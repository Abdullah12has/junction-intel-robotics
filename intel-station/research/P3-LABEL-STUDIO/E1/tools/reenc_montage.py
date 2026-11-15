import json, os, subprocess
from PIL import Image, ImageDraw

R = "/home/intel/intel-junction-2026"
WD = os.path.join(R, "work/P3-LABEL-STUDIO/E1")
OUTD = os.path.join(WD, "out")
FRD = os.path.join(WD, "frames", "offset_check")
SD = os.path.join(R, "data", "studio-datasets")
FPS = 30
SERIALS = {"top": "243622061421", "close": "243722064316"}
CELL = (320, 240)

EPS = {}
for line in open(os.path.join(OUTD, "episodes_census.jsonl"), encoding="utf-8"):
    e = json.loads(line)
    EPS["%s:%d" % (e["dataset"][:8], e["episode_index"])] = e


def frame(e, cam, idx, tag):
    k = [x for x in e["video_segments"] if x.endswith(SERIALS[cam])][0]
    s = e["video_segments"][k]
    out = os.path.join(FRD, "%s_%s_f%04d.png" % (tag, cam, idx))
    if not os.path.exists(out):
        mp4 = os.path.join(SD, e["dataset"], s["mp4"])
        subprocess.run(["ffmpeg", "-v", "error", "-ss", "%.4f" % (s["from_ts"] + idx / FPS), "-i", mp4,
                        "-frames:v", "1", "-y", out], capture_output=True, text=True)
    return out if os.path.exists(out) else None


ROWS = [("a4ba6ede:0", "6d6bdce3:3", 150), ("a4ba6ede:1", "6d6bdce3:4", 150),
        ("a4ba6ede:2", "6d6bdce3:5", 150), ("a4ba6ede:3", "6d6bdce3:6", 150),
        ("18468a09:6", "6d6bdce3:12", 708)]
W, H = CELL
canvas = Image.new("RGB", (4 * W, len(ROWS) * H), (255, 255, 255))
draw = ImageDraw.Draw(canvas)
for ri, (ck, rk, idx) in enumerate(ROWS):
    c, r = EPS[ck], EPS[rk]
    cells = [(c, "top", ck.replace(":", "_")), (r, "top", rk.replace(":", "_")),
             (c, "close", ck.replace(":", "_")), (r, "close", rk.replace(":", "_"))]
    labels = ["%s top f%d" % (ck, idx), "%s top f%d" % (rk, idx),
              "%s close f%d" % (ck, idx), "%s close f%d" % (rk, idx)]
    for ci, ((e, cam, tag), lab) in enumerate(zip(cells, labels)):
        p = frame(e, cam, idx, tag)
        if p is None:
            print("missing", lab)
            continue
        im = Image.open(p).convert("RGB").resize((W, H))
        canvas.paste(im, (ci * W, ri * H))
        draw.rectangle([ci * W, ri * H, ci * W + 175, ri * H + 14], fill=(0, 0, 0))
        draw.text((ci * W + 3, ri * H + 2), lab, fill=(255, 255, 255))
canvas.save(os.path.join(OUTD, "reenc_montage.jpg"), quality=85)
print("written", os.path.join(OUTD, "reenc_montage.jpg"), canvas.size)
