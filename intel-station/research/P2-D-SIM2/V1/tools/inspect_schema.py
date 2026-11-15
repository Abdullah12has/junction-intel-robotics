import json, os, sys, glob
import numpy as np
import pyarrow.parquet as pq

SIM = "/home/intel/intel-junction-2026/data/sim/2026-11-14/dataset-150-raw"
REAL = "/home/intel/intel-junction-2026/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af"
ROBOTS = "/home/intel/intel-junction-2026/app/motion/project_robots.json"

def short(x, n=400):
    s = json.dumps(x, default=str)[:n]
    return s

print("== sim attempt-0000 metadata.json keys")
md = json.load(open(os.path.join(SIM, "attempt-0000", "metadata.json")))
print(short(md, 1500))
print("== sim attempt-0000 files", sorted(os.listdir(os.path.join(SIM, "attempt-0000"))))
tr = json.load(open(os.path.join(SIM, "attempt-0000", "trace.json")))
print("== trace.json type", type(tr).__name__, (list(tr.keys())[:30] if isinstance(tr, dict) else len(tr)))
print(short(tr, 800))
z = np.load(os.path.join(SIM, "attempt-0000", "trajectory.npz"), allow_pickle=False)
print("== trajectory.npz keys/shapes")
for k in z.files:
    print(k, z[k].shape, z[k].dtype)
print("== top-level sim dir", sorted(os.listdir(SIM))[:12])
print("== attempt folders", len(glob.glob(os.path.join(SIM, "attempt-*"))))
print("== recording_context.json")
rc = json.load(open(os.path.join(SIM, "..", "dataset-150-raw", "recording_context.json")))
print(short(rc, 1200))
print("== real dir", sorted(os.listdir(REAL)))
info = json.load(open(os.path.join(REAL, "meta", "info.json")))
print("== real info keys", list(info.keys()))
print(short(info.get("features", {}), 1500))
print("== real data files", sorted(glob.glob(os.path.join(REAL, "data", "**", "*.parquet"), recursive=True))[:5])
print("== real video files", sorted(glob.glob(os.path.join(REAL, "videos", "**", "*.mp4"), recursive=True))[:8])
pf = pq.ParquetFile(sorted(glob.glob(os.path.join(REAL, "data", "**", "*.parquet"), recursive=True))[0])
print("== real parquet schema", pf.schema_arrow)
rob = json.load(open(ROBOTS))
print("== robots top keys", list(rob.keys())[:10] if isinstance(rob, dict) else type(rob).__name__)
print(short(rob, 1500))
