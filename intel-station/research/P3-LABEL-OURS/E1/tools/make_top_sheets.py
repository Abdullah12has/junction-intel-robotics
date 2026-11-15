import os, sys
from PIL import Image, ImageDraw, ImageFont

R = sys.argv[1]
OUT = sys.argv[2]
SV = os.path.join(R, 'logs/depth/servo')
P = os.path.join
try:
    font = ImageFont.load_default(size=14)
except TypeError:
    font = ImageFont.load_default()


def cell(path, w, h, caption):
    canvas = Image.new('RGB', (w, h), (40, 40, 40))
    if not os.path.exists(path):
        ImageDraw.Draw(canvas).text((6, h // 2), 'missing', fill=(255, 255, 255), font=font)
    else:
        im = Image.open(path).convert('RGB')
        im.thumbnail((w, h - 18))
        canvas.paste(im, ((w - im.size[0]) // 2, 18 + (h - 18 - im.size[1]) // 2))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, w, 17], fill=(0, 0, 0))
    d.text((3, 1), caption, fill=(255, 255, 0), font=font)
    return canvas


def sheet(rows, ncols, w, h, name):
    S = Image.new('RGB', (ncols * w, len(rows) * h), (20, 20, 20))
    for r, row in enumerate(rows):
        for c, (path, cap) in enumerate(row):
            S.paste(cell(path, w, h, cap), (c * w, r * h))
    dst = P(OUT, name + '.jpg')
    S.save(dst, quality=90)
    print('OK', dst, S.size[0], S.size[1])


top1 = ['grip_25.jpg', 'grip_30.jpg', 'grip_35.jpg', 'grip_40.jpg', 'grip_45.jpg',
        'hover12.jpg', 'jaw_calib.jpg', 'jaw_check.jpg', 'jawcmp.jpg', 'mask2.jpg',
        'masks.jpg', 'now3.jpg', 'now4.jpg', 'now5.jpg', 'now6.jpg']
top2 = ['now7.jpg', 'now8.jpg', 'now9.jpg', 'now10.jpg', 'now11.jpg',
        'now12.jpg', 'now13.jpg', 'now14.jpg', 'now15.jpg', 'park.jpg',
        'peek.jpg', 'scan_now.jpg', 'after_scan_hover.jpg', 'estop_close_ov.jpg', 'estop_dbg.jpg']


def rows_of(files, ncols):
    rows, cur = [], []
    for f in files:
        cur.append((P(SV, f), f[:-4]))
        if len(cur) == ncols:
            rows.append(cur)
            cur = []
    if cur:
        rows.append(cur)
    return rows


sheet(rows_of(top1, 5), 5, 320, 240, 'servo_top_a')
sheet(rows_of(top2, 5), 5, 320, 240, 'servo_top_b')

quads = [('164609_a2_desc1', 'quad.jpg'), ('164722_a3_desc2', 'quad.jpg'), ('164822_a4_grasph', 'quad.jpg'),
         ('164913_a5_closed', 'quad.jpg'), ('165016_a6_lift', 'quad.jpg'), ('165247_a7_released', 'rel.jpg')]
q = [(P(SV, s, fn), s[:6] + ' ' + fn[:-4]) for s, fn in quads]
sheet([q[0:3], q[3:6]], 3, 320, 240, 'servo_quads')
