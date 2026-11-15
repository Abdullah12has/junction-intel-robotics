import os, sys
from PIL import Image, ImageDraw, ImageFont

R = sys.argv[1]
OUT = sys.argv[2]
MON = os.path.join(R, 'logs/monitor')

# native size of every e-stop frame (no resizing)
for f in sorted(os.listdir(MON)):
    if f.startswith('estop_') and f.endswith('.jpg'):
        with Image.open(os.path.join(MON, f)) as im:
            print('NATIVE', f, im.size[0], im.size[1])

groups = {
    'verify_D': ['estop_top_1794673145.jpg', 'estop_top_1794673200.jpg', 'estop_top_1794675022.jpg', 'estop_top_1794677184.jpg'],
    'verify_E': ['estop_top_1794677552.jpg', 'estop_top_1794677636.jpg', 'estop_top_1794678371.jpg', 'estop_top_1794679239.jpg'],
    'verify_F': ['estop_close_1794673200.jpg', 'estop_close_1794677636.jpg', 'estop_close_1794679304.jpg', 'estop_close_1794679373.jpg'],
    'verify_G': ['estop_close_1794679818.jpg', 'estop_close_1794679874.jpg', 'estop_close_1794679987.jpg'],
    'verify_H': ['estop_wrist_1794679388.jpg', 'estop_wrist_1794679389.jpg', 'estop_wrist_1794679390.jpg', 'estop_wrist_1794679391.jpg'],
    'verify_I': ['estop_wrist_1794679392.jpg', 'estop_wrist_1794679393.jpg', 'estop_wrist_1794679394.jpg', 'estop_wrist_1794679395.jpg'],
    'verify_J': ['estop_wrist_1794679396.jpg', 'estop_wrist_1794679397.jpg', 'estop_wrist_1794679398.jpg', 'estop_wrist_1794679399.jpg'],
    'verify_K': ['estop_wrist_1794679401.jpg', 'estop_wrist_1794679402.jpg'],
}
try:
    font = ImageFont.load_default(size=18)
except TypeError:
    font = ImageFont.load_default()

for name, files in groups.items():
    frames = []
    for f in files:
        frames.append((f, Image.open(os.path.join(MON, f)).convert('RGB')))
    W = max(im.size[0] for _, im in frames)
    H = max(im.size[1] for _, im in frames)
    sheet = Image.new('RGB', (2 * W, 2 * H), (20, 20, 20))
    for i, (f, im) in enumerate(frames):
        if im.size != (W, H):
            im = im.resize((W, H))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 340, 26], fill=(0, 0, 0))
        d.text((4, 3), f, fill=(255, 255, 0), font=font)
        sheet.paste(im, ((i % 2) * W, (i // 2) * H))
    dst = os.path.join(OUT, name + '.jpg')
    sheet.save(dst, quality=92)
    print('OK', dst, 2 * W, 2 * H)
