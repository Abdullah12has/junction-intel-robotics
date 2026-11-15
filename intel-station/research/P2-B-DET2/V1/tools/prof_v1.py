"""V1 seam profile on the two live stills (analysis only). usage: python3 -I -B prof_v1.py APP_DIR"""
import sys

import cv2
import numpy as np

sys.path.insert(0, sys.argv[1])
import inspector as I  # noqa: E402

top = cv2.imread("/home/intel/intel-junction-2026/logs/detlive/top_153426.jpg")
h, s, v = I._split(top)
roi = I._roi(h.shape, (I.TOP_X0, 0, h.shape[1], 338))
fg = cv2.morphologyEx((roi & (v > 70)).astype(np.uint8), cv2.MORPH_OPEN, I.K3).astype(bool)
m, a, st = [c for c in I._components(fg, 350) if c[0][260, 420]][0]
x0, y0, w, hh = [int(t) for t in st[:4]]
print("TOP blob", a, "bbox", x0, y0, w, hh)
chroma = m & (s >= 45)
white = m & (s < 25) & (v > 150)
for y in range(y0, y0 + hh, 2):
    row = []
    for x in range(x0 - 2, x0 + w + 2, 2):
        row.append("W" if white[y, x] else ("R" if chroma[y, x] else ("w" if m[y, x] else ".")))
    print("%3d %s" % (y, "".join(row)))
print("V/S at x=400,410,420,430,440 (rows 236..345 step 3)")
for y in range(236, 346, 3):
    print(y, " ".join("%3d/%3d" % (v[y, x], s[y, x]) for x in (400, 410, 420, 430, 440)))

close = cv2.imread("/home/intel/intel-junction-2026/logs/detlive/close_153426.jpg")
hc, sc, vc = I._split(close)
H, W = hc.shape
roi_c = I._roi(hc.shape, I.CLOSE_ROI)
fgc = cv2.morphologyEx((roi_c & (((sc > 85) & (vc > 90)) | ((vc > 170) & (sc < 70)))).astype(np.uint8),
                       cv2.MORPH_OPEN, I.K3).astype(bool)
comps = I._components(fgc, 1500)
for m2, a2_, st2 in comps:
    med = [round(float(np.median(hc[m2])), 1), round(float(np.median(sc[m2])), 1), round(float(np.median(vc[m2])), 1)]
    print("CLOSE comp", a2_, [int(t) for t in st2[:4]], "hsv_med", med)
big = max(comps, key=lambda t: t[1])
m2, a2_, st2 = big
cx0, cy0, cw, ch = [int(t) for t in st2[:4]]
chromaC = m2 & (sc >= 45)
whiteC = m2 & (sc < 25) & (vc > 150)
print("CLOSE largest", a2_, "bbox", cx0, cy0, cw, ch)
for y in range(cy0, cy0 + ch, 6):
    print("".join("W" if whiteC[y, x] else ("R" if chromaC[y, x] else ("w" if m2[y, x] else "."))
                  for x in range(cx0, cx0 + cw, 6)))
