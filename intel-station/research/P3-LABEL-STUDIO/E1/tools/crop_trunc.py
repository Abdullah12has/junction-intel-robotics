import os
from PIL import Image, ImageDraw

WD = "/home/intel/intel-junction-2026/work/P3-LABEL-STUDIO/E1"
FRD = os.path.join(WD, "frames", "offset_check")
OUTD = os.path.join(WD, "out")
BOX = (120, 0, 360, 240)   # top view, 640x480 frame: gripper above the BLUE plate
SCALE = 2.5
pairs = [("18468a09_6_top_f0708.png", "18468a09:6 copy, f708 (23.6 s)"),
         ("6d6bdce3_12_top_f0708.png", "6d6bdce3:12 rep, f708 (same frame)")]
tiles = []
for fn, lab in pairs:
    im = Image.open(os.path.join(FRD, fn)).convert("RGB")
    print(fn, im.size)
    c = im.crop(BOX).resize((int((BOX[2] - BOX[0]) * SCALE), int((BOX[3] - BOX[1]) * SCALE)))
    d = ImageDraw.Draw(c)
    d.rectangle([0, 0, 260, 14], fill=(0, 0, 0))
    d.text((3, 2), lab, fill=(255, 255, 255))
    tiles.append(c)
W = sum(t.size[0] for t in tiles) + 10
H = max(t.size[1] for t in tiles)
canvas = Image.new("RGB", (W, H), (255, 255, 255))
x = 0
for t in tiles:
    canvas.paste(t, (x, 0))
    x += t.size[0] + 10
canvas.save(os.path.join(OUTD, "trunc_crop_top_f708.jpg"), quality=88)
print("written", canvas.size)
