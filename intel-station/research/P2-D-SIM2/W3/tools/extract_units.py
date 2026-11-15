#!/usr/bin/env python3
"""P2-D-SIM2 W3: read-only unit check of the sim LeRobot extract vs the real dataset (action value ranges)."""
import glob
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

R = "/home/intel/intel-junction-2026"
SIM_DIR = f"{R}/data/sim/2026-11-14/extract-lerobot/dataset-150"
REAL_DIR = f"{R}/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af"
sim_files = sorted(glob.glob(f"{SIM_DIR}/data/chunk-000/*.parquet"))
s = pd.concat([pq.read_table(f).to_pandas() for f in sim_files], ignore_index=True)
A = np.stack(s["action"].to_numpy())
print("sim_extract rows", len(s), "cols", list(s.columns))
print("sim action min", np.round(A.min(0), 3).tolist(), "max", np.round(A.max(0), 3).tolist())
real_files = sorted(glob.glob(f"{REAL_DIR}/data/chunk-000/*.parquet"))
d = pd.concat([pq.read_table(f).to_pandas() for f in real_files], ignore_index=True)
B = np.stack(d["action"].to_numpy())
print("real action min", np.round(B.min(0), 2).tolist(), "max", np.round(B.max(0), 2).tolist())
m = json.load(open(f"{SIM_DIR}/meta/info.json"))
print("sim info codebase", m["codebase_version"], "total_episodes", m.get("total_episodes"),
      "total_frames", m.get("total_frames"))
print("sim camera keys", [k for k in m["features"] if k.startswith("observation.images")])
