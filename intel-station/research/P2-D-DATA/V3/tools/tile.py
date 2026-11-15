import sys
from PIL import Image, ImageDraw

# usage: python3 -I tile.py OUT.jpg COLS "path|label" "path|label" ...
out = sys.argv[1]
cols = int(sys.argv[2])
items = [a.split("|", 1) for a in sys.argv[3:]]
W, H = 320, 240
rows = (len(items) + cols - 1) // cols
sheet = Image.new("RGB", (cols * W, rows * H), (0, 0, 0))
for i, (p, lab) in enumerate(items):
    im = Image.open(p).convert("RGB").resize((W, H))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 14], fill=(0, 0, 0))
    d.text((3, 2), lab, fill=(255, 255, 0))
    sheet.paste(im, ((i % cols) * W, (i // cols) * H))
sheet.save(out, quality=90)
print("TILE_OK", out, len(items))
