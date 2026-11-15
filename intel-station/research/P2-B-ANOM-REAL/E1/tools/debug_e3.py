import sys
sys.path.insert(0, "/home/intel/intel-junction-2026/app/anomaly")
import glob
import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import spot_samples as S

ep = pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(f"{S.ROOT}/meta/episodes/chunk-*/file-*.parquet"))], ignore_index=True)
d = pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(f"{S.ROOT}/data/chunk-*/file-*.parquet"))], ignore_index=True)
e = 3
res, post = S.analyse(d[d["episode_index"] == e])
r = ep[ep["episode_index"] == e].iloc[0]
cam = S.CAMS[0]
cr = int(r[f"videos/observation.images.{cam}/chunk_index"])
fi = int(r[f"videos/observation.images.{cam}/file_index"])
base = int(round(float(r[f"videos/observation.images.{cam}/from_timestamp"]) * S.FPS))
path = f"{S.ROOT}/videos/observation.images.{cam}/chunk-{cr:03d}/file-{fi:03d}.mp4"
got = S.decode(path, [base, base + post])
pre, pst = got[base], got[base + post]
pg = cv2.GaussianBlur(pre, (5, 5), 0).astype(np.int16)
qg = cv2.GaussianBlur(pst, (5, 5), 0).astype(np.int16)
dd = np.abs(qg - pg).max(axis=2)
hp = cv2.cvtColor(pre, cv2.COLOR_BGR2HSV).astype(np.int16)
hq = cv2.cvtColor(pst, cv2.COLOR_BGR2HSV).astype(np.int16)
vpre, vpost = hp[..., 2], hq[..., 2]
hd = np.abs(hq[..., 0] - hp[..., 0]); hd = np.minimum(hd, 180 - hd)
gained = (vpost > S.POST_BRIGHT) & ((vpre < S.PRE_DARK) | ((hq[..., 1] > 80) & (hd > 18)))
m1 = ((dd > S.DIFF_T) & gained).astype(np.uint8) * 255
sr = S.red_mask(pre) & S.red_mask(pst)
sr = cv2.dilate(sr.astype(np.uint8) * 255, np.ones((9, 9), np.uint8)) > 0
m2 = m1.copy(); m2[sr] = 0
print("m1 px", int((m1 > 0).sum()), "m2 px", int((m2 > 0).sum()), "static red px", int(sr.sum()))
for name, m in (("m1", m1), ("m2", m2)):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    comps = sorted([(int(st[k][4]), k) for k in range(1, n)], reverse=True)[:8]
    for a, k in comps:
        x, y, w, h, _ = st[k]
        print(f"{name} comp area={a} bbox=({x},{y},{w},{h}) V_pre={float(vpre[lab==k].mean()):.0f} V_post={float(vpost[lab==k].mean()):.0f} S_post={float(hq[...,1][lab==k].mean()):.0f}")
cv2.imwrite("/home/intel/intel-junction-2026/work/P2-B-ANOM-REAL/E1/raw/dbg_e3_close_m2.png", m2)
cv2.imwrite("/home/intel/intel-junction-2026/work/P2-B-ANOM-REAL/E1/raw/dbg_e3_close_post.png", pst)
