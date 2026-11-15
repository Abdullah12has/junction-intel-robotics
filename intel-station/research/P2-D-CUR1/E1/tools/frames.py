#!/usr/bin/env python3
"""Contact sheet (3x2 tiles, evenly spaced over one episode) for one camera of a LeRobot v3 dataset (read-only).

Usage: python3 -I frames.py <dataset_dir> <episode> <camera_key> <out_jpg>
"""
import glob
import subprocess
import sys

import pyarrow.parquet as pq


def main():
    ds, ep, cam, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    rows = []
    for path in sorted(glob.glob(ds + "/meta/episodes/chunk-*/file-*.parquet")):
        rows += pq.read_table(path).to_pylist()
    row = next(r for r in rows if int(r["episode_index"]) == ep)
    chunk = int(row[f"videos/{cam}/chunk_index"])
    file_idx = int(row[f"videos/{cam}/file_index"])
    t0 = float(row[f"videos/{cam}/from_timestamp"])
    t1 = float(row[f"videos/{cam}/to_timestamp"])
    video = f"{ds}/videos/{cam}/chunk-{chunk:03d}/file-{file_idx:03d}.mp4"
    dur = t1 - t0
    vf = f"fps=6/{dur:.4f},scale=320:240,tile=3x2"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0:.4f}", "-t", f"{dur:.4f}", "-i", video,
         "-vf", vf, "-frames:v", "1", out],
        check=True,
    )
    print("ok", "ep", ep, cam, "from", round(t0, 3), "to", round(t1, 3), "->", out)


if __name__ == "__main__":
    main()
