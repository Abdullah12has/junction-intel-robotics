import glob, os, sys
from PIL import Image, ImageDraw, ImageFont

R = sys.argv[1]
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
try:
    font = ImageFont.load_default(size=13)
except TypeError:
    font = ImageFont.load_default()
TW, TH = 240, 180
COLS = 4


def fit(img, w, h):
    im = img.convert('RGB')
    im.thumbnail((w, h))
    canvas = Image.new('RGB', (w, h), (0, 0, 0))
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return canvas


for view in ['top', 'close', 'wrist']:
    files = sorted(glob.glob(os.path.join(R, 'logs/monitor', 'estop_' + view + '_*.jpg')))
    rows = (len(files) + COLS - 1) // COLS
    sheet = Image.new('RGB', (TW * COLS, TH * rows), (20, 20, 20))
    for i, f in enumerate(files):
        c = fit(Image.open(f), TW, TH)
        d = ImageDraw.Draw(c)
        name = os.path.basename(f)
        d.rectangle([0, 0, 150, 17], fill=(0, 0, 0))
        d.text((3, 1), name, fill=(255, 255, 0), font=font)
        sheet.paste(c, ((i % COLS) * TW, (i // COLS) * TH))
    dst = os.path.join(OUT, 'estop_' + view + '_sheet.jpg')
    sheet.save(dst, quality=90)
    print('OK', dst, 'files=%d' % len(files), 'size=%dx%d' % sheet.size)
