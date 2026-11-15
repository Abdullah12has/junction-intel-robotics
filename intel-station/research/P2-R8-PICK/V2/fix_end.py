"""Verifier V2: re-extract a missing end frame by stepping back from the last frame. Usage: python3 -I fix_end.py HERE EP"""
import glob
import os
import subprocess
import sys

import pyarrow.parquet as pq

ROOT = "/home/intel/intel-junction-2026/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af"
TOP = "observation.images.realsense 243622061421"
HERE, EP = sys.argv[1], int(sys.argv[2])
length = chunk = fidx = t0 = None
for f in sorted(glob.glob(ROOT + "/meta/episodes/**/*.parquet", recursive=True)):
    d = pq.read_table(f).to_pydict()
    for i in range(len(d["episode_index"])):
        if int(d["episode_index"][i]) == EP:
            length = int(d["length"][i])
            chunk = int(d[f"videos/{TOP}/chunk_index"][i])
            fidx = int(d[f"videos/{TOP}/file_index"][i])
            t0 = float(d[f"videos/{TOP}/from_timestamp"][i])
path = os.path.join(ROOT, "videos", TOP, "chunk-%03d" % chunk, "file-%03d.mp4" % fidx)
out = os.path.join(HERE, "frames", "ep%02d_end.png" % EP)
for back in range(0, 30):
    idx = length - 1 - back
    ts = t0 + idx / 30.0
    if os.path.exists(out):
        os.remove(out)
    subprocess.run(["ffmpeg", "-v", "error", "-ss", "%.4f" % ts, "-i", path, "-frames:v", "1", "-y", out], check=False)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print("ok: ep", EP, "end frame index used", idx, "of length", length, "ts", round(ts, 4), flush=True)
        break
else:
    print("FAILED: no decodable end frame for ep", EP, flush=True)
