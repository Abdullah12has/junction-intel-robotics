import glob, os, sys
from PIL import Image

R = sys.argv[1]
eps = sorted(glob.glob(os.path.join(R, 'logs/sort/*/ep_*')))
tot_j = tot_c = 0
for e in eps:
    rel = os.path.relpath(e, R)
    imgs = sorted(glob.glob(os.path.join(e, '*.jpg')))
    crops = sorted(glob.glob(os.path.join(e, 'crops', '*')))
    tot_j += len(imgs)
    tot_c += len(crops)
    dims = set()
    for im in imgs[:2] + crops[:2]:
        with Image.open(im) as I:
            dims.add(I.size)
    print(rel + ' jpgs=%d crops=%d dims=%s' % (len(imgs), len(crops), sorted(dims)))
    print('   jpgs: ' + ' '.join(os.path.basename(i) for i in imgs))
    if crops:
        print('   crops(%d): ' % len(crops) + ' '.join(os.path.basename(c) for c in crops[:80]))
print('TOTAL jpgs=%d crops=%d episodes=%d' % (tot_j, tot_c, len(eps)))
