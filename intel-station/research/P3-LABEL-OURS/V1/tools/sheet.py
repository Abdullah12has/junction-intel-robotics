import os, sys
from PIL import Image, ImageDraw
base = '/home/intel/intel-junction-2026/logs/sort/any_1740'
out = '/home/intel/intel-junction-2026/work/P3-LABEL-OURS/V1/sheets'
os.makedirs(out, exist_ok=True)
phases = ['pre', 'stage1', 'grasp_open', 'closed', 'lifted', 'released', 'station']
cams = ['top', 'close', 'wrist']
for ep in sys.argv[1:]:
    d = os.path.join(base, ep)
    tiles = []
    for ph in phases:
        for cam in cams:
            p = os.path.join(d, ph + '_' + cam + '.jpg')
            if os.path.exists(p):
                tiles.append((ph + '_' + cam, p))
    if not tiles:
        print(ep, 'none')
        continue
    cols, tw, th = 3, 320, 240
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * tw, rows * (th + 18)), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    for i, (lab, p) in enumerate(tiles):
        im = Image.open(p).convert('RGB').resize((tw, th))
        x = (i % cols) * tw
        y = (i // cols) * (th + 18)
        sheet.paste(im, (x, y + 18))
        dr.text((x + 4, y + 3), lab, fill=(0, 0, 0))
    sheet.save(os.path.join(out, ep + '.jpg'), quality=85)
    print(ep, len(tiles), 'tiles')
