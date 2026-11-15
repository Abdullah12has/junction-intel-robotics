# V1 verifier: zoomed 3-panel view (start | grasp instant | lift) of a crop, grid in ORIGINAL pixel coordinates
import sys
from PIL import Image, ImageDraw

OUT = "/home/intel/intel-junction-2026/work/P2-R8-PICK/V1"
name, x0, y0, x1, y1, scale = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6])
W, H = int((x1 - x0) * scale), int((y1 - y0) * scale)
sheet = Image.new("RGB", (3 * W + 20, H), (0, 0, 0))
for k, lab in enumerate(("start", "grasp", "lift")):
    im = Image.open("%s/frames/%s_%s.png" % (OUT, name, lab)).convert("RGB").crop((x0, y0, x1, y1))
    im = im.resize((W, H), Image.NEAREST)
    d = ImageDraw.Draw(im)
    for x in range(((x0 + 19) // 20) * 20, x1, 20):
        X = (x - x0) * scale
        d.line([(X, 0), (X, H)], fill=(255, 255, 0) if x % 100 == 0 else (120, 120, 0), width=1)
        d.text((X + 2, 2), str(x), fill=(255, 255, 0))
    for y in range(((y0 + 19) // 20) * 20, y1, 20):
        Y = (y - y0) * scale
        d.line([(0, Y), (W, Y)], fill=(0, 255, 255) if y % 100 == 0 else (0, 120, 120), width=1)
        d.text((2, Y + 2), str(y), fill=(0, 255, 255))
    d.text((W - 60, H - 14), lab, fill=(255, 255, 255))
    sheet.paste(im, (k * (W + 10), 0))
sheet.save("%s/sheets/zoom_%s.png" % (OUT, name))
print("zoom", name, sheet.size)
