#!/usr/bin/env python3
"""P2-D-SIM2 W1 (read-only): 2x2 composite for visual review at t = 3 s.
Top row: sim overhead render | real top RealSense video. Bottom row: sim wrist (fixed close) render | real close video.
Usage: python3 -I composite_video_frames.py <out dir>
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
SIM = Path("/home/intel/intel-junction-2026/data/sim/2026-11-14/dataset-150-raw/attempt-0000")
REAL = Path("/home/intel/intel-junction-2026/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af/videos")


def frame_at(path, t=3.0):
    cmd = ["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(480, 640, 3)


panels = {
    "sim_overhead": frame_at(SIM / "overhead.mp4"),
    "real_top": frame_at(REAL / "observation.images.realsense 243622061421/chunk-000/file-000.mp4"),
    "sim_wrist_close": frame_at(SIM / "wrist.mp4"),
    "real_close": frame_at(REAL / "observation.images.realsense 243722064316/chunk-000/file-000.mp4"),
}
comp = np.zeros((960, 1280, 3), dtype=np.uint8)
comp[:480, :640] = panels["sim_overhead"]
comp[:480, 640:] = panels["real_top"]
comp[480:, :640] = panels["sim_wrist_close"]
comp[480:, 640:] = panels["real_close"]
Image.fromarray(comp).save(OUT / "composite_sim_vs_real_t3s.jpg", quality=85)
for name, f in panels.items():
    luma = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    print(f"{name}: mean_luma={luma.mean():.1f} p98-p2={np.percentile(luma, 98) - np.percentile(luma, 2):.1f}")
