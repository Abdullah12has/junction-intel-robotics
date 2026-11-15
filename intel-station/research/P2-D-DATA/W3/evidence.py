#!/usr/bin/env python3
"""P2-D-DATA W3: high-resolution evidence sheet per episode (read-only input).
Usage: python -I evidence.py EPISODES_JSON DATA_ROOT OUT_DIR
Rows = the two RealSense cameras; columns = grasp closure, S1 cut (1.0 s after lift), release, end."""
import json, os, subprocess, sys
from PIL import Image, ImageDraw

EP_JSON, DATA_ROOT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
TW, TH = 320, 240
FPS = 30
CAMS = ['realsense 243722064316', 'realsense 243622061421']
os.makedirs(OUT, exist_ok=True)
eps = json.load(open(EP_JSON))


def frame(ddir, cam, vid, t_rel, png):
    path = os.path.join(DATA_ROOT, ddir, 'videos', 'observation.images.' + cam,
                        'chunk-%03d' % vid['chunk'], 'file-%03d.mp4' % vid['file'])
    t_abs = vid['from_ts'] + t_rel
    if not (os.path.exists(png) and os.path.getsize(png) > 0):
        subprocess.run(['ffmpeg', '-v', 'error', '-ss', '%.4f' % t_abs, '-i', path, '-frames:v', '1',
                        '-vf', 'scale=%d:%d' % (TW, TH), '-y', png], capture_output=True)
    return png if (os.path.exists(png) and os.path.getsize(png) > 0) else None


for r in eps:
    cut = None
    for sg in r['segments']:
        if sg['skill'] == 'S1':
            cut = sg['end_idx']
    cols = [('grasp', r['grasp_idx']), ('S1 cut', cut), ('release', r['release_idx']),
            ('end', r['data_rows'] - 1)]
    cdir = r['dataset']
    ddir = [d for d in os.listdir(DATA_ROOT) if d.startswith(cdir)][0]
    W = 150 + len(cols) * TW
    H = 30 + 2 * TH
    im = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(im)
    d.text((4, 4), 'ep%d %s %s' % (r['episode'], r['cls_final'], r['verdict']), fill='black')
    for j, (name, idx) in enumerate(cols):
        x = 150 + j * TW
        tt = 'n/a' if idx is None else '%s %.2fs' % (name, idx / FPS)
        d.text((x + 4, 4), tt, fill='black')
        for i, cam in enumerate(CAMS):
            y = 30 + i * TH
            if idx is None:
                d.rectangle([x, y, x + TW - 1, y + TH - 1], outline='red')
                d.text((x + 4, y + 4), 'no event', fill='red')
                continue
            png = os.path.join(OUT, '%s_ep%d_%s_%s.png' % (cdir[:8], r['episode'], cam.split()[-1][-6:], name.replace(' ', '')))
            got = frame(ddir, cam, r['video'][cam], idx / FPS, png)
            if got:
                im.paste(Image.open(got).convert('RGB'), (x, y))
            else:
                d.rectangle([x, y, x + TW - 1, y + TH - 1], outline='red')
                d.text((x + 4, y + 4), 'decode failed', fill='red')
    for i, cam in enumerate(CAMS):
        d.text((4, 30 + i * TH + 4), 'cam ' + cam.split()[-1][-6:], fill='black')
    im.save(os.path.join(OUT, '%s_ep%d.png' % (cdir[:8], r['episode'])))
print('EVIDENCE_OK', len(eps))
