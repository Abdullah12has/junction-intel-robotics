"""P2-B-DET2 E3 harvest. Run: envs/data-env/bin/python -I -B tools/harvest.py
Per episode: home-and-still windows (joints 0-4 within 8 units of HOME for >= 15 frames).
The first window (begins in first 10% of the episode) gives 'start', the last window (ends in last 10%) gives 'end'.
The middle frame of each window is extracted from both RealSense videos with ffmpeg (libdav1d)."""
import glob
import json
import os
import subprocess

import numpy as np
import pyarrow.parquet as pq

ROOT = "/home/intel/intel-junction-2026"
WORK = f"{ROOT}/work/P2-B-DET2/E3"
OUT = f"{WORK}/frames"
DS = {"6d6bdce3": f"{ROOT}/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af",
      "a4ba6ede": f"{ROOT}/data/studio-datasets/a4ba6ede-092c-450f-8c76-559f66991dbd"}
CAM = {"top": "243622061421", "close": "243722064316"}
HOME = np.array([-6.0, -98.2, 96.6, 60.5, -48.3])
TOL, MIN_RUN, FPS = 8.0, 15, 30.0


def runs(mask, min_len):
    out, s = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and s is None:
            s = i
        elif not m and s is not None:
            if i - s >= min_len:
                out.append((s, i - 1))
            s = None
    return out


def read_ds(ds_dir):
    frames, meta = {}, {}
    for f in sorted(glob.glob(f"{ds_dir}/data/chunk-*/file-*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state"]).to_pydict()
        for e, fi, s in zip(t["episode_index"], t["frame_index"], t["observation.state"]):
            frames.setdefault(e, {})[fi] = s
    for f in sorted(glob.glob(f"{ds_dir}/meta/episodes/chunk-*/file-*.parquet")):
        for r in pq.read_table(f).to_pylist():
            meta[r["episode_index"]] = r
    return frames, meta


def main():
    os.makedirs(OUT, exist_ok=True)
    index, log = [], []
    for ds, ds_dir in DS.items():
        frames, meta = read_ds(ds_dir)
        log.append(f"DS {ds} episodes={len(meta)} frames={sum(len(v) for v in frames.values())}")
        for ep in sorted(frames):
            fis = sorted(frames[ep])
            st = np.array([frames[ep][f][:5] for f in fis], dtype=float)
            dev = np.max(np.abs(st - HOME), axis=1)
            picks, n = [], len(fis)
            for (s, e) in runs(dev < TOL, MIN_RUN):
                if s <= 0.1 * n and not any(p[0] == "start" for p in picks):
                    picks.append(("start", (s, e)))
                elif e >= 0.9 * (n - 1) and not any(p[0] == "end" for p in picks):
                    picks.append(("end", (s, e)))
            log.append(f"  ep{ep:02d} len={n} still_runs={[(int(fis[s]), int(fis[e])) for s, e in runs(dev < TOL, MIN_RUN)]} picks={[p[0] for p in picks]}")
            for which, (s, e) in picks:
                mid = (s + e) // 2
                fi = int(fis[mid])
                for cam, serial in CAM.items():
                    key = f"observation.images.realsense {serial}"
                    m = meta[ep]
                    video = f"{ds_dir}/videos/{key}/chunk-{m[f'videos/{key}/chunk_index']:03d}/file-{m[f'videos/{key}/file_index']:03d}.mp4"
                    t_sec = float(m[f"videos/{key}/from_timestamp"]) + fi / FPS
                    name = f"{ds}_e{ep:02d}_{which}_{serial}"
                    path = f"{OUT}/{name}.png"
                    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t_sec:.5f}", "-i", video,
                                    "-frames:v", "1", path], check=True)
                    index.append({"id": name, "ds": ds, "ep": int(ep), "which": which, "cam": cam,
                                  "serial": serial, "frame_index": fi, "t_sec": round(t_sec, 5),
                                  "still_frames": [int(fis[s]), int(fis[e])], "dev_at_pick": round(float(dev[mid]), 2),
                                  "png": path, "video": video})
    with open(f"{WORK}/out/samples_index.json", "w") as fh:
        json.dump(index, fh, indent=1)
    log.append(f"SAMPLES {len(index)}")
    for ds in DS:
        for which in ("start", "end"):
            log.append(f"  {ds} {which}: {sum(1 for r in index if r['ds'] == ds and r['which'] == which and r['cam'] == 'top')} episodes")
    print("\n".join(log))


if __name__ == "__main__":
    main()
