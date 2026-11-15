import sys
sys.path.insert(0, "/home/intel/intel-junction-2026/app/anomaly")
import glob
import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import spot_samples as S

DS = "/home/intel/intel-junction-2026/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af"
ep = pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(f"{DS}/meta/episodes/chunk-*/file-*.parquet"))], ignore_index=True)
d = pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(f"{DS}/data/chunk-*/file-*.parquet"))], ignore_index=True)
picks = [(0, 300), (1, 200), (3, 450)]
tiles = []
for e, fi in picks:
    r = ep[ep["episode_index"] == e].iloc[0]
    st = np.stack(d[d["episode_index"] == e]["observation.state"].to_numpy())
    print("EP", e, "frame", fi, "state", np.round(st[fi], 1).tolist())
    row = []
    for cam in S.CAMS:
        cr = int(r[f"videos/observation.images.{cam}/chunk_index"])
        fidx = int(r[f"videos/observation.images.{cam}/file_index"])
        base = int(round(float(r[f"videos/observation.images.{cam}/from_timestamp"]) * 30))
        path = f"{DS}/videos/observation.images.{cam}/chunk-{cr:03d}/file-{fidx:03d}.mp4"
        im = S.decode(path, [base + fi])[base + fi]
        t = cv2.resize(im, (320, 240))
        cv2.putText(t, f"e{e} f{fi} {S.CAM_ROLE[cam]}", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        row.append(t)
    tiles.append(np.hstack(row))
out = np.vstack(tiles)
cv2.imwrite("/home/intel/intel-junction-2026/work/P2-B-ANOM-REAL/E1/raw/a2_held_probe.jpg", out)
print("ok", out.shape)
