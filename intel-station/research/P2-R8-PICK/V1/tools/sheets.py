# V1 verifier: contact sheets (start | lift) with a pixel grid, for labelling the grasped brick
import os, sys
from PIL import Image, ImageDraw

OUT = "/home/intel/intel-junction-2026/work/P2-R8-PICK/V1"


def grid(im):
    im = im.convert("RGB")
    d = ImageDraw.Draw(im)
    W, H = im.size
    for x in range(0, W, 80):
        d.line([(x, 0), (x, H)], fill=(255, 255, 0), width=1)
        d.text((x + 2, 2), str(x), fill=(255, 255, 0))
    for y in range(0, H, 80):
        d.line([(0, y), (W, y)], fill=(0, 255, 255), width=1)
        d.text((2, y + 2), str(y), fill=(0, 255, 255))
    return im


os.makedirs(OUT + "/sheets", exist_ok=True)
for name in sys.argv[1:]:
    a = Image.open("%s/frames/%s_start.png" % (OUT, name))
    b = Image.open("%s/frames/%s_lift.png" % (OUT, name))
    W, H = a.size
    sheet = Image.new("RGB", (W * 2 + 10, H), (0, 0, 0))
    sheet.paste(grid(a), (0, 0))
    sheet.paste(grid(b), (W + 10, 0))
    sheet.save("%s/sheets/%s_start_vs_lift.png" % (OUT, name))
    print(name, a.size, b.size)
