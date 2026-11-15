import os, sys
from PIL import Image, ImageDraw, ImageFont

R = sys.argv[1]
OUT = sys.argv[2]
groups = {
    'verify_A': ['estop_top_1794677654.jpg', 'estop_top_1794677779.jpg', 'estop_top_1794677889.jpg', 'estop_top_1794677902.jpg'],
    'verify_B': ['estop_top_1794677923.jpg', 'estop_top_1794677939.jpg', 'estop_top_1794677947.jpg', 'estop_top_1794677973.jpg'],
    'verify_C': ['estop_top_1794678583.jpg', 'estop_close_1794677939.jpg', 'estop_close_1794679859.jpg', 'estop_wrist_1794675172.jpg'],
}
try:
    font = ImageFont.load_default(size=16)
except TypeError:
    font = ImageFont.load_default()
for name, files in groups.items():
    sheet = Image.new('RGB', (1280, 960), (20, 20, 20))
    for i, f in enumerate(files):
        im = Image.open(os.path.join(R, 'logs/monitor', f)).convert('RGB')
        im = im.resize((640, 480))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 300, 22], fill=(0, 0, 0))
        d.text((4, 2), f, fill=(255, 255, 0), font=font)
        sheet.paste(im, ((i % 2) * 640, (i // 2) * 480))
    dst = os.path.join(OUT, name + '.jpg')
    sheet.save(dst, quality=92)
    print('OK', dst)
