#!/usr/bin/env python3
"""E1 (P2-D-BUILD): trim kept episodes and write one LeRobot v3 dataset per keep_as class.

Run: python3 -I curate.py EPISODES_CSV SRC_ROOT OUT_ROOT VERSION RESULTS_JSON
- SRC_ROOT/<dataset_id> is a LeRobot v3 dataset (read only).
- Frame i of an episode has timestamp i / fps. A frame is kept when start_s <= t < end_s;
  end_s < 0 means the episode end.
- Output: OUT_ROOT/<CLASS>_v<VERSION>/ with only the kept episodes, re-indexed by lerobot.
  Videos are re-encoded with the source settings (libsvtav1, yuv420p, GOP 2, CRF 30, preset 12,
  pyav). Stats are computed by lerobot on save_episode and finalize.
- Classes with no kept episode are not built.
"""

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from lerobot.configs import RGBEncoderConfig
from lerobot.configs.video import DepthEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoDecoderCache, decode_video_frames_torchcodec

TASKS = {
    "S1": "move the brick to the inspection spot",
    "PP_BLUE": "put the brick on the blue plate",
    "PP_GREEN": "put the brick on the green plate",
}
AUTO_KEYS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
ENCODER = {
    "vcodec": "libsvtav1",
    "pix_fmt": "yuv420p",
    "g": 2,
    "crf": 30,
    "preset": 12,
    "fast_decode": 0,
    "video_backend": "pyav",
}


class NoDepthEncoder(DepthEncoderConfig):
    """RGB-only data: placeholder that skips the depth codec probe (as in Studio's video_encoding_settings.py)."""

    def __post_init__(self):
        pass

    def get_codec_options(self, encoder_threads=None, as_strings=False):
        raise NotImplementedError("depth is not recorded")


def read_rows(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            keep = raw["keep_as"].strip()
            if keep == "DROP":
                continue
            if keep not in TASKS:
                raise SystemExit(f"unknown keep_as {keep!r} for episode {raw['episode_index']}")
            rows.append(
                {
                    "dataset_id": raw["dataset_id"].strip(),
                    "episode_index": int(raw["episode_index"]),
                    "keep_as": keep,
                    "start_s": float(raw["start_s"]),
                    "end_s": float(raw["end_s"]),
                    "note": raw.get("note", "").strip(),
                }
            )
    return rows


def window(length, fps, start_s, end_s):
    keep = []
    for i in range(length):
        t = i / fps
        if t < start_s - 1e-9:
            continue
        if end_s >= 0 and t >= end_s - 1e-9:
            break
        keep.append(i)
    return keep


def episode_row(src, ep):
    for row in src.meta.episodes:
        if int(row["episode_index"]) == ep:
            return row
    raise SystemExit(f"episode {ep} not found in {src.root}")


def decode_frames(src, erow, cam, idxs, fps):
    path = Path(src.root) / src.meta.get_video_file_path(int(erow["episode_index"]), cam)
    base = float(erow[f"videos/{cam}/from_timestamp"])
    timestamps = [base + i / fps for i in idxs]
    cache = VideoDecoderCache(max_size=1)
    try:
        frames = decode_video_frames_torchcodec(str(path), timestamps, src.tolerance_s, decoder_cache=cache, return_uint8=True)
    finally:
        cache.clear()
    frames = torch.as_tensor(frames)
    if frames.dtype != torch.uint8:
        frames = (frames.clamp(0, 1) * 255).round().to(torch.uint8)
    return frames.permute(0, 2, 3, 1).contiguous().numpy()


def build(cls, rows, sources, fps, out_root, version, results):
    name = f"{cls}_v{version}"
    out_dir = out_root / name
    if out_dir.exists():
        shutil.rmtree(out_dir)  # only this script's previous output for the same name
    first = sources[rows[0]["dataset_id"]]
    features = {}
    for key, ft in first.meta.features.items():
        if key in AUTO_KEYS:
            continue
        ft = dict(ft)
        ft.pop("info", None)
        features[key] = ft
    cam_keys = [k for k, ft in features.items() if ft["dtype"] == "video"]
    ds = LeRobotDataset.create(
        repo_id=f"local/{name}",
        root=out_dir,
        fps=fps,
        features=features,
        robot_type=first.meta.robot_type,
        use_videos=True,
        streaming_encoding=True,
        encoder_threads=None,
        encoder_queue_maxsize=60,
        rgb_encoder=RGBEncoderConfig(**ENCODER),
        depth_encoder=NoDepthEncoder(),
    )
    per_episode, total = [], 0
    for r in rows:
        src = sources[r["dataset_id"]]
        erow = episode_row(src, r["episode_index"])
        idxs = window(int(erow["length"]), fps, r["start_s"], r["end_s"])
        if not idxs or idxs != list(range(idxs[0], idxs[-1] + 1)):
            raise SystemExit(f"{name}: empty or non-contiguous window for episode {r['episode_index']}")
        g0 = int(erow["dataset_from_index"]) + idxs[0]
        g1 = g0 + len(idxs)
        hf = src.hf_dataset[g0:g1]
        action = np.asarray(hf["action"], dtype=np.float32)
        state = np.asarray(hf["observation.state"], dtype=np.float32)
        cams = {c: decode_frames(src, erow, c, idxs, fps) for c in cam_keys}
        for j in range(len(idxs)):
            frame = {"action": action[j], "observation.state": state[j], "task": TASKS[cls]}
            for c in cam_keys:
                frame[c] = cams[c][j]
            ds.add_frame(frame)
        ds.save_episode()
        per_episode.append(
            {
                "dataset_id": r["dataset_id"],
                "source_episode": r["episode_index"],
                "source_length": int(erow["length"]),
                "window_first_frame": idxs[0],
                "window_last_frame": idxs[-1],
                "frames": len(idxs),
                "start_s": r["start_s"],
                "end_s": r["end_s"],
                "note": r["note"],
            }
        )
        total += len(idxs)
        print(f"{name}: episode {r['episode_index']} -> {len(idxs)} frames", flush=True)
    if ds.writer is not None and getattr(ds.writer, "_streaming_encoder", None) is None:
        ds.writer.stop_image_writer()
    ds.finalize()
    results["classes"][name] = {
        "class": cls,
        "task": TASKS[cls],
        "path": str(out_dir),
        "episodes": len(per_episode),
        "frames": total,
        "cameras": cam_keys,
        "per_episode": per_episode,
    }


def main():
    csv_path, src_root, out_root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    version, results_path = int(sys.argv[4]), Path(sys.argv[5])
    rows = read_rows(csv_path)
    sources = {}
    for r in rows:
        if r["dataset_id"] not in sources:
            sources[r["dataset_id"]] = LeRobotDataset(f"local/{r['dataset_id']}", root=src_root / r["dataset_id"])
    fps = int(next(iter(sources.values())).fps)
    results = {"csv": str(csv_path), "version": version, "fps": fps, "classes": {}}
    for cls in TASKS:
        cls_rows = [r for r in rows if r["keep_as"] == cls]
        if cls_rows:
            build(cls, cls_rows, sources, fps, out_root, version, results)
        else:
            results["classes"][f"{cls}_v{version}"] = {"status": "no kept episodes, not built"}
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_episode"} for k, v in results["classes"].items()}, indent=2))


if __name__ == "__main__":
    main()
