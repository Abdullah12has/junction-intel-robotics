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
for e in (3, 5, 6):
    res, post = S.analyse(d[d["episode_index"] == e])
    r = ep[ep["episode_index"] == e].iloc[0]
    for cam in S.CAMS:
        cr = int(r[f"videos/observation.images.{cam}/chunk_index"])
        fi = int(r[f"videos/observation.images.{cam}/file_index"])
        base = int(round(float(r[f"videos/observation.images.{cam}/from_timestamp"]) * S.FPS))
        path = f"{S.ROOT}/videos/observation.images.{cam}/chunk-{cr:03d}/file-{fi:03d}.mp4"
        got = S.decode(path, [base, base + post])
        pre, pst = got[base], got[base + post]
        pg = cv2.GaussianBlur(pre, (5, 5), 0).astype(np.int16)
        qg = cv2.GaussianBlur(pst, (5, 5), 0).astype(np.int16)
        dd = np.abs(qg - pg).max(axis=2)
        vpre = cv2.cvtColor(pre, cv2.COLOR_BGR2HSV)[..., 2].astype(np.int16)
        vpst = cv2.cvtColor(pst, cv2.COLOR_BGR2HSV)[..., 2].astype(np.int16)
        gain = (vpst - vpre) > 40
        m0 = ((dd > 45) & gain).astype(np.uint8) * 255
        print(f"EP {e} {S.CAM_ROLE[cam]} diff>45 px={int((dd>45).sum())} gain px={int(gain.sum())} both px={int(m0.astype(bool).sum())}")
        n, lab, st, _ = cv2.connectedComponentsWithStats(m0)
        comps = sorted([(int(st[k][4]), k) for k in range(1, n)], reverse=True)[:6]
        for a, k in comps:
            x, y, w, h, _ = st[k]
            print(f"   comp area={a} bbox=({x},{y},{w},{h}) V_pre={float(vpre[lab==k].mean()):.0f} V_post={float(vpst[lab==k].mean()):.0f}")
        red_pre, red_post = S.red_mask(pre), S.red_mask(pst)
        print(f"   red pre px={int(red_pre.sum())} post px={int(red_post.sum())} static red px={int((red_pre & red_post).sum())}")
        cv2.imwrite(f"/home/intel/intel-junction-2026/work/P2-B-ANOM-REAL/E1/raw/dbg_e{e}_{S.CAM_ROLE[cam]}_diffmask.png", m0)
