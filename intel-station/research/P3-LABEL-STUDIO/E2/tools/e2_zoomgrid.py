import os, sys
from PIL import Image, ImageDraw

R = "/home/intel/intel-junction-2026/work/P3-LABEL-STUDIO/E2"
FRD = os.path.join(R, "frames")
prefix = sys.argv[1]
eps = [int(x) for x in sys.argv[2].split(",")]
tag = sys.argv[3]                       # start or end
box = tuple(int(x) for x in sys.argv[4].split(","))  # x0,y0,x1,y1 in 640x480 top frame
scale = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
cols = 3
tiles = []
for e in eps:
    fn = [f for f in os.listdir(FRD) if f.startswith("full_%s_ep%d_%s_" % (prefix, e, tag))]
    im = Image.open(os.path.join(FRD, fn[0])).convert("RGB")
    # full_ panels are 640x480 with a 20 px header in the saved composite; the tag selects the frame
    tiles.append((e, im))
W = int((box[2] - box[0]) * scale)
H = int((box[3] - box[1]) * scale)
rows = (len(tiles) + cols - 1) // cols
canvas = Image.new("RGB", (cols * W, rows * (H + 18)), (255, 255, 255))
d = ImageDraw.Draw(canvas)
for k, (e, im) in enumerate(tiles):
    x = (k % cols) * W
    y = (k // cols) * (H + 18)
    c = im.crop(box).resize((W, H), Image.LANCZOS)
    canvas.paste(c, (x, y + 18))
    d.text((x + 4, y + 3), "%s ep%d %s" % (prefix, e, tag), fill=(0, 0, 0))
outp = os.path.join(R, "out", "zoom_%s_%s.png" % (prefix, tag))
canvas.save(outp)
print("written", outp, canvas.size)
