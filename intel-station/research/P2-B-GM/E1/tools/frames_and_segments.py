"""Save the recorded release->home segments (npy) and the top-view frame at each release moment (jpg).

Read-only on the dataset. Video is decoded with ffmpeg (argument list, no shell). No camera, no port.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

TOOLS = Path("/home/intel/intel-junction-2026/work/P2-B-GM/E1/tools")
sys.path.insert(0, str(TOOLS))
import analyze_releases as ar  # noqa: E402

E1 = Path("/home/intel/intel-junction-2026/work/P2-B-GM/E1")
TOP = "observation.images.realsense 243622061421"
PLATES = {"blue": 11, "green": 9}


def main() -> None:
    seg = json.loads((E1 / "segments.json").read_text(encoding="utf-8"))
    data = ar.read_all("data/chunk-000/*.parquet").sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    states = np.stack(data["observation.state"].to_numpy()).astype(np.float64)
    ep_ids = data["episode_index"].to_numpy()
    meta = ar.read_all("meta/episodes/chunk-000/*.parquet")
    (E1 / "frames").mkdir(parents=True, exist_ok=True)
    for name, ep in PLATES.items():
        s = states[ep_ids == ep]
        r = seg[name]["release_frame"]
        e = seg[name]["end_frame"]
        np.save(E1 / f"{name}_recorded_segment.npy", s[r : e + 1])
        row = meta[meta["episode_index"] == ep].iloc[0]
        chunk = int(row[f"videos/{TOP}/chunk_index"])
        file_idx = int(row[f"videos/{TOP}/file_index"])
        from_ts = float(row[f"videos/{TOP}/from_timestamp"])
        video = ar.ROOT / f"videos/{TOP}/chunk-{chunk:03d}/file-{file_idx:03d}.mp4"
        ts = from_ts + r / ar.FPS
        out = E1 / "frames" / f"{name}_release_ep{ep}_top.jpg"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{ts:.4f}", "-i", str(video),
            "-frames:v", "1", "-vf", "scale=640:480", "-q:v", "4", str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        print(name, "ep", ep, "video", video.name, "from_ts", from_ts, "ts", round(ts, 4), "rc", proc.returncode,
              "stderr", proc.stderr.strip()[:300], "bytes", out.stat().st_size if out.exists() else None)
        print("   segment shape", s[r : e + 1].shape)


if __name__ == "__main__":
    main()
