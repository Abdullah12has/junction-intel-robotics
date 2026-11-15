import os, sys
from PIL import Image, ImageDraw, ImageFont

R = sys.argv[1]
OUT = sys.argv[2]
MON = os.path.join(R, 'logs/monitor')
items = [
    ('estop_top_1794673145.jpg', (0, 0, 200, 150)),
    ('estop_top_1794673200.jpg', (0, 0, 200, 150)),
    ('estop_wrist_1794679401.jpg', (0, 330, 200, 480)),
    ('estop_wrist_1794679402.jpg', (0, 330, 200, 480)),
    ('estop_top_1794677939.jpg', (140, 0, 340, 150)),
    ('estop_top_1794677973.jpg', (160, 0, 360, 150)),
]
S = 3
try:
    font = ImageFont.load_default(size=16)
except TypeError:
    font = ImageFont.load_default()
W, H = 200 * S, 150 * S
sheet = Image.new('RGB', (3 * W, 2 * H), (20, 20, 20))
for i, (f, box) in enumerate(items):
    im = Image.open(os.path.join(MON, f)).convert('RGB').crop(box)
    im = im.resize((W, H))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 270, 24], fill=(0, 0, 0))
    d.text((4, 3), f, fill=(255, 255, 0), font=font)
    sheet.paste(im, ((i % 3) * W, (i // 3) * H))
dst = os.path.join(OUT, 'zoom_checks.jpg')
sheet.save(dst, quality=92)
print('OK', dst, sheet.size[0], sheet.size[1])
