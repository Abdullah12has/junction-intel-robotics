import glob
import sys
import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = "/home/intel/intel-junction-2026/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af"
KEY = "realsense 243722064316"
ep = pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(f"{ROOT}/meta/episodes/chunk-*/file-*.parquet"))], ignore_index=True)
d = pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(f"{ROOT}/data/chunk-*/file-*.parquet"))], ignore_index=True)
for _, r in ep.iterrows():
    e = int(r["episode_index"])
    sub = d[d["episode_index"] == e]
    g = np.array([x[5] for x in sub["observation.state"]])
    pan = np.array([x[0] for x in sub["observation.state"]])
    print(f"EP {e} len={len(sub)} gripper_min={g.min():.1f} gripper_max={g.max():.1f}")
    print("  grip/10f:", " ".join(f"{v:.0f}" for v in g[::10]))
    print("  pan/10f :", " ".join(f"{v:.0f}" for v in pan[::10]))
# AV1 decode test on the first file of this key
f = f"{ROOT}/videos/observation.images.{KEY}/chunk-000/file-000.mp4"
cap = cv2.VideoCapture(f, cv2.CAP_FFMPEG)
print("cv2 opened", cap.isOpened())
ok, fr = cap.read()
print("cv2 read", ok, None if fr is None else fr.shape)
n = 0
while ok and n < 1000:
    ok, fr = cap.read()
    n += 1
print("cv2 frames after first", n)
