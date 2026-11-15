"""Prep: read the curated S1 dataset (LeRobot v3 layout) and check AV1 decode parity.

Runs in envs/lerobot-env (pyarrow, PyAV, ffmpeg CLI). Read-only on the dataset;
writes only under OUT. Dataset: a4ba6ede (the training set of our model 389da0fd).
"""
import json
import os
import subprocess

import av
import numpy as np
import pyarrow.parquet as pq

DS = "/home/intel/.local/share/physicalai/datasets/a4ba6ede-092c-450f-8c76-559f66991dbd"
OUT = "/home/intel/intel-junction-2026/work/P2-B-OVRUN/E1/data"
CAMS = ["realsense 243722064316", "realsense 243622061421"]
H, W = 480, 640


def video_path(cam):
    return f"{DS}/videos/observation.images.{cam}/chunk-000/file-000.mp4"


def ffmpeg_frames(path, count=None):
    cmd = ["ffmpeg", "-v", "error", "-i", path]
    if count:
        cmd += ["-frames:v", str(count)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def ffprobe_count(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames,codec_name,pix_fmt,width,height",
         "-of", "json", path],
        capture_output=True, check=True, text=True).stdout
    return json.loads(out)["streams"][0]


def main():
    os.makedirs(OUT, exist_ok=True)
    data = pq.read_table(f"{DS}/data/chunk-000/file-000.parquet").to_pandas()
    eps = pq.read_table(f"{DS}/meta/episodes/chunk-000/file-000.parquet").to_pandas()
    tasks = pq.read_table(f"{DS}/meta/tasks.parquet").to_pandas().reset_index()
    np.savez(
        f"{OUT}/a4ba6ede_arrays.npz",
        state=np.stack(data["observation.state"].to_numpy()).astype(np.float32),
        action=np.stack(data["action"].to_numpy()).astype(np.float32),
        episode_index=data["episode_index"].to_numpy(np.int64),
        frame_index=data["frame_index"].to_numpy(np.int64),
        task_index=data["task_index"].to_numpy(np.int64),
    )
    text_col = [c for c in tasks.columns if c != "task_index"][0]
    task_text = {int(r["task_index"]): str(r[text_col]) for _, r in tasks.iterrows()}
    vid_cols = [c for c in eps.columns if c.startswith("videos/")]
    episodes = []
    for _, row in eps.iterrows():
        rec = {"episode_index": int(row["episode_index"]), "length": int(row["length"])}
        if "tasks" in row:
            rec["tasks"] = [str(x) for x in row["tasks"]]
        for c in vid_cols:
            v = row[c]
            rec[c] = v.item() if hasattr(v, "item") else str(v)
        episodes.append(rec)

    p1 = video_path(CAMS[1])
    container = av.open(p1)
    stream = container.streams.video[0]
    pyav_codec = stream.codec_context.name
    pyav_meta_frames = stream.frames
    pyav_frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i >= 30:
            break
        pyav_frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    pa = np.stack(pyav_frames)
    ff = ffmpeg_frames(p1, 30)[: len(pa)]
    diff = np.abs(pa.astype(np.int16) - ff.astype(np.int16))

    probes = {}
    for cam in CAMS:
        st = ffprobe_count(video_path(cam))
        probes[cam] = {"codec": st.get("codec_name"), "pix_fmt": st.get("pix_fmt"),
                       "width": st.get("width"), "height": st.get("height"),
                       "nb_read_frames": st.get("nb_read_frames"), "path": video_path(cam),
                       "bytes": os.path.getsize(video_path(cam))}
    result = {
        "dataset": DS,
        "data_rows": int(len(data)),
        "data_columns": list(data.columns),
        "episode_columns": list(eps.columns),
        "video_columns": vid_cols,
        "task_text": task_text,
        "episodes": episodes,
        "cameras": CAMS,
        "video_probe": probes,
        "pyav": {"codec": pyav_codec, "stream_frames_meta": pyav_meta_frames, "first_frames": len(pa)},
        "decode_parity_first30_pyav_vs_ffmpeg": {
            "shape": list(pa.shape), "max_abs_diff": int(diff.max()),
            "mean_abs_diff": float(diff.mean()), "frames_differ": int((diff.reshape(len(pa), -1).max(axis=1) > 0).sum())},
    }
    with open(f"{OUT}/prep.json", "w") as fh:
        json.dump(result, fh, indent=1, default=str)
    print(json.dumps({k: result[k] for k in ["data_rows", "data_columns", "episode_columns", "video_columns",
                                              "task_text", "video_probe", "pyav", "decode_parity_first30_pyav_vs_ffmpeg"]},
                     indent=1, default=str))
    print("EPISODES", json.dumps(episodes, default=str)[:3000])


if __name__ == "__main__":
    main()
