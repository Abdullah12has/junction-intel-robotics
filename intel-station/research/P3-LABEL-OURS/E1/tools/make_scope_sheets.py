import os, sys
from PIL import Image, ImageDraw, ImageFont

R = sys.argv[1]
OUT = sys.argv[2]
D = os.path.join(R, 'logs/depth')
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


calib = P(R, 'logs/grasp_auto/calib_171531')
g1 = P(D, 'g1_162903')
c424 = P(g1, 'c424x240@15')
c480 = P(g1, 'c480x270@15')
rows1 = [
    [(P(calib, 'c0_top.jpg'), 'calib c0_top'), (P(calib, 'c0_wrist.jpg'), 'calib c0_wrist'), (P(calib, 'c0_close.jpg'), 'calib c0_close')],
    [(P(g1, 'top_color.jpg'), 'g1 top_color'), (P(g1, 'top_ir.png'), 'g1 top_ir'), (P(g1, 'top_height_viz.png'), 'g1 top_height_viz')],
    [(P(g1, 'close_color.jpg'), 'g1 close_color'), (P(g1, 'close_ir.png'), 'g1 close_ir'), (P(g1, 'close_height_viz.png'), 'g1 close_height_viz')],
    [(P(c424, 'close_color.jpg'), 'c424 close_color'), (P(c424, 'close_ir.png'), 'c424 close_ir'), (P(c424, 'close_height_viz.png'), 'c424 close_height_viz')],
    [(P(c480, 'close_color.jpg'), 'c480 close_color'), (P(c480, 'close_ir.png'), 'c480 close_ir'), (P(c480, 'close_height_viz.png'), 'c480 close_height_viz')],
]
sheet(rows1, 3, 320, 240, 'scope_calib_g1')

J = P(D, 'jprobe', '164223')
rows2 = []
for pr in ['base', 'el', 'pan', 'roll', 'sl', 'wf']:
    rows2.append([
        (P(J, pr, 'live_wrist.jpg'), pr + ' live_wrist'),
        (P(J, pr, 'live_top.jpg'), pr + ' live_top'),
        (P(J, pr, 'live_close.jpg'), pr + ' live_close'),
        (P(J, pr, 'close_color.jpg'), pr + ' close_color'),
    ])
sheet(rows2, 4, 240, 180, 'scope_jprobe')

SV = P(D, 'servo')
sess = ['163312_p0_base', '163325_p1_sl33', '163504_p2_wf73', '163733_p3_up', '163740_p4_roll',
        '164033_p5_srv', '164509_a1_roll', '164609_a2_desc1', '164722_a3_desc2', '164822_a4_grasph',
        '164913_a5_closed', '165016_a6_lift', '165247_a7_released']
types = [('live_top.jpg', 'live_top'), ('live_wrist.jpg', 'live_wrist'), ('live_close.jpg', 'live_close'),
         ('close_color.jpg', 'close_color'), ('close_height_viz.png', 'height_viz'), ('close_ir.png', 'ir')]


def srows(ss):
    return [[(P(SV, s, fn), s[:6] + ' ' + lab) for fn, lab in types] for s in ss]


sheet(srows(sess[0:5]), 6, 240, 180, 'scope_servo_a')
sheet(srows(sess[5:10]), 6, 240, 180, 'scope_servo_b')
sheet(srows(sess[10:13]), 6, 240, 180, 'scope_servo_c')
