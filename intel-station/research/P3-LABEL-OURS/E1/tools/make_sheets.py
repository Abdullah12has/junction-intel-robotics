import glob, os, sys
from PIL import Image, ImageDraw, ImageFont

R = sys.argv[1]
OUT = sys.argv[2]
only = sys.argv[3:]  # optional episode dirs like any_1740/ep_0001
os.makedirs(OUT, exist_ok=True)
try:
    font = ImageFont.load_default(size=13)
except TypeError:
    font = ImageFont.load_default()
PHASES = ['pre', 'stage1', 'grasp_open', 'closed', 'lifted', 'released', 'station']
CAMS = ['top', 'close', 'wrist']
TW, TH = 320, 240
CW, CH = 240, 240
EW, EH = 480, 360
W = 960


def fit(img, w, h):
    im = img.convert('RGB')
    im.thumbnail((w, h))
    canvas = Image.new('RGB', (w, h), (0, 0, 0))
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return canvas


def caption(canvas, text):
    d = ImageDraw.Draw(canvas)
    tw = d.textlength(text, font=font)
    d.rectangle([0, 0, tw + 6, 17], fill=(0, 0, 0))
    d.text((3, 1), text, fill=(255, 255, 0), font=font)
    return canvas


eps = sorted(glob.glob(os.path.join(R, 'logs/sort/*/ep_*')))
for e in eps:
    run = os.path.basename(os.path.dirname(e))
    ep = os.path.basename(e)
    if only and (run + '/' + ep) not in only:
        continue
    files = {os.path.basename(p): p for p in glob.glob(os.path.join(e, '*.jpg'))}
    blocks = []
    for ph in PHASES:
        cells = []
        for cam in CAMS:
            name = ph + '_' + cam + '.jpg'
            if name in files:
                cells.append(caption(fit(Image.open(files[name]), TW, TH), run + '/' + ep + ' ' + ph + '/' + cam))
            else:
                cells.append(None)
        if any(c is not None for c in cells):
            row = Image.new('RGB', (W, TH), (20, 20, 20))
            for i, c in enumerate(cells):
                if c is not None:
                    row.paste(c, (i * TW, 0))
            blocks.append(row)
    crop_files = sorted(glob.glob(os.path.join(e, 'crops', '*')))
    for k in range(0, len(crop_files), 4):
        row = Image.new('RGB', (W, CH), (20, 20, 20))
        for i, cp in enumerate(crop_files[k:k + 4]):
            c = fit(Image.open(cp), CW, CH)
            row.paste(caption(c, 'crop ' + os.path.basename(cp)), (i * CW, 0))
        blocks.append(row)
    known = set(ph + '_' + cam + '.jpg' for ph in PHASES for cam in CAMS)
    extras = [files[n] for n in sorted(files) if n not in known]
    for k in range(0, len(extras), 2):
        row = Image.new('RGB', (W, EH), (20, 20, 20))
        for i, p in enumerate(extras[k:k + 2]):
            c = fit(Image.open(p), EW, EH)
            row.paste(caption(c, 'extra ' + os.path.basename(p)), (i * EW, 0))
        blocks.append(row)
    if not blocks:
        print('SKIP ' + run + '/' + ep + ' (no images)')
        continue
    H = sum(b.height for b in blocks)
    sheet = Image.new('RGB', (W, H), (20, 20, 20))
    y = 0
    for b in blocks:
        sheet.paste(b, (0, y))
        y += b.height
    dst = os.path.join(OUT, run + '_' + ep + '.jpg')
    sheet.save(dst, quality=88)
    print('OK ' + dst + ' size=%dx%d rows=%d' % (W, H, len(blocks)))
