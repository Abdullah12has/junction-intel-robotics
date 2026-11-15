#!/usr/bin/env python3
"""P2-D-SIM2 W1 (read-only): sim render vs real RealSense image statistics, sim camera key mapping,
and a consistency check of the sim LeRobot export against the raw npz trajectories.

Usage: python3 -I camera_and_export_check.py <out dir>
Only reads files; decodes video with ffmpeg (1 frame per second, first 5 s).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
SIM = Path("/home/intel/intel-junction-2026/data/sim/2026-11-14/dataset-150-raw")
SIM_LR = Path("/home/intel/intel-junction-2026/data/sim/2026-11-14/extract-lerobot/dataset-150")
REAL = Path("/home/intel/intel-junction-2026/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af")
VIEWS = Path("/home/intel/intel-junction-2026/data/inspection/views")


def frames(path, n=5):
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vf", "fps=1", "-frames:v", str(n),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    arr = np.frombuffer(raw, dtype=np.uint8)
    k = arr.size // (480 * 640 * 3)
    return arr[: k * 480 * 640 * 3].reshape(k, 480, 640, 3).astype(np.float32)


def metrics(F):
    R, G, B = F[..., 0], F[..., 1], F[..., 2]
    L = 0.299 * R + 0.587 * G + 0.114 * B
    mx, mn = F.max(-1), F.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    grad = (np.abs(np.diff(L, axis=2)).mean() + np.abs(np.diff(L, axis=1)).mean()) / 2
    bluish = ((B > 150) & (G > 110) & (R < 90) & (B > R + 80)).mean()
    greenish = ((G > 120) & (G > R + 50) & (G > B + 40)).mean()
    return {"frames": int(F.shape[0]), "mean_R": float(R.mean()), "mean_G": float(G.mean()),
            "mean_B": float(B.mean()), "mean_luma": float(L.mean()),
            "luma_contrast_p98_minus_p2": float(np.percentile(L, 98) - np.percentile(L, 2)),
            "saturation_mean": float(sat.mean()), "gradient_mean_luma": float(grad),
            "frac_bluish_px": float(bluish), "frac_greenish_px": float(greenish)}


def jpg(path):
    return np.asarray(Image.open(path).convert("RGB").resize((640, 480)), dtype=np.float32)[None]


result = {}
sim_ov = frames(SIM / "attempt-0000" / "overhead.mp4")
sim_wr = frames(SIM / "attempt-0000" / "wrist.mp4")
real_top = frames(REAL / "videos/observation.images.realsense 243622061421/chunk-000/file-000.mp4")
real_close = frames(REAL / "videos/observation.images.realsense 243722064316/chunk-000/file-000.mp4")
top_jpg = jpg(VIEWS / "rs_video8.jpg")
close_jpg = jpg(VIEWS / "rs_video14.jpg")
result["metrics"] = {
    "sim_overhead_mp4": metrics(sim_ov), "sim_wrist_mp4": metrics(sim_wr),
    "real_top_video_file0": metrics(real_top), "real_close_video_file0": metrics(real_close),
    "real_top_view_jpg_rs_video8": metrics(top_jpg), "real_close_view_jpg_rs_video14": metrics(close_jpg),
}

# Which LeRobot export camera key holds which raw render (first frame, mean absolute pixel difference).
lr1 = frames(SIM_LR / "videos/observation.images.camera1/chunk-000/file-000.mp4", 1)
lr2 = frames(SIM_LR / "videos/observation.images.camera2/chunk-000/file-000.mp4", 1)
f_ov, f_wr = sim_ov[:1], sim_wr[:1]
result["lerobot_key_mapping_mae_0_255"] = {
    "camera1_vs_overhead": float(np.abs(lr1 - f_ov).mean()), "camera1_vs_wrist": float(np.abs(lr1 - f_wr).mean()),
    "camera2_vs_overhead": float(np.abs(lr2 - f_ov).mean()), "camera2_vs_wrist": float(np.abs(lr2 - f_wr).mean()),
}

# Export consistency: LeRobot parquet rows and first-episode states vs raw trajectory.npz.
t = pq.read_table(SIM_LR / "data/chunk-000/file-000.parquet").to_pandas()
verif = json.loads((SIM / "verification.json").read_text())
raw_frames = sum(json.loads((d / "metadata.json").read_text())["frames"] for d in sorted(SIM.glob("attempt-*")))
st0 = np.stack(t[t["episode_index"] == 0]["observation.state"].to_numpy()).astype(float)
with np.load(SIM / "attempt-0000" / "trajectory.npz") as z:
    raw0 = z["state"].astype(float)
result["export_check"] = {
    "lerobot_rows": int(len(t)), "raw_frames_sum": int(raw_frames),
    "verification_json_frames": verif.get("frames"), "verification_json_episodes": verif.get("verified_episodes"),
    "episodes_in_parquet": int(t["episode_index"].nunique()),
    "episode0_rows": int(len(st0)), "episode0_raw_rows": int(len(raw0)),
    "episode0_state_max_abs_diff_rad": float(np.abs(st0[: len(raw0)] - raw0[: len(st0)]).max()) if len(st0) == len(raw0) else None,
    "summary_json_entries": len(json.loads((SIM / "summary.json").read_text())),
    "summary_json_accepted": sum(bool(e["accepted"]) for e in json.loads((SIM / "summary.json").read_text())),
}

# Side-by-side composite for visual review: top row sim overhead | real top, bottom row sim wrist | real close.
def tile(a):
    return np.asarray(a, dtype=np.uint8)
comp = np.zeros((960, 1280, 3), dtype=np.uint8)
comp[:480, :640] = tile(sim_ov[0]); comp[:480, 640:] = tile(top_jpg[0])
comp[480:, :640] = tile(sim_wr[0]); comp[480:, 640:] = tile(close_jpg[0])
Image.fromarray(comp).save(OUT / "composite_sim_vs_real.jpg", quality=85)

(OUT / "camera_export_metrics.json").write_text(json.dumps(result, indent=1) + "\n")
print(json.dumps(result, indent=1))
