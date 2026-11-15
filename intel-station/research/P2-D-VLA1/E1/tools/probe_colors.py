import sys

import numpy as np

sys.path.insert(0, "/home/intel/intel-junction-2026/app/vla")
import build_dataset as B

WRIST = "/home/intel/intel-junction-2026/data/wrist/raw"
C = float(sys.argv[1])
segs = B.list_media(WRIST, "w")
cand = B.candidates(segs, C - 1.0, C + 1.0, B.WRIST_SEG_S)
frames = list(B.decode_window(cand, C - 1.0, C + 1.0))
ft = np.array([tt for tt, _ in frames])
k = int(np.argmin(np.abs(ft - (C - 0.726))))
img = frames[k][1].reformat(format="rgb24").to_ndarray().astype(np.int16)
print("frame %d at %+.0f ms from centre" % (k, 1000 * (ft[k] - C)))
pts = {"left finger": (420, 150), "brick face": (400, 430), "brick side": (380, 330), "right finger": (430, 600),
       "green plate": (200, 100), "black mat": (120, 400)}
for name, (y, x) in pts.items():
    r, g, b = img[y - 3:y + 4, x - 3:x + 4].reshape(-1, 3).mean(axis=0)
    mx, mn = max(r, g, b), min(r, g, b)
    s = (mx - mn) / mx if mx > 0 else 0
    print("%-12s RGB %6.1f %6.1f %6.1f  sat %.2f" % (name, r, g, b, s))
band = img[360:480]
r, g, b = band[..., 0], band[..., 1], band[..., 2]
mx = np.maximum(np.maximum(r, g), b).astype(float)
mn = np.minimum(np.minimum(r, g), b).astype(float)
sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0)
for thr in (0.45, 0.55, 0.65):
    m = (sat > thr) & (r > 120)
    cols = np.where(m.sum(axis=0) > 3)[0]
    print("sat > %.2f and R > 120: pixels %d, columns %s" % (thr, int(m.sum()), (cols.min(), cols.max()) if len(cols) else None))
