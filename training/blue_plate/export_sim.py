"""Export accepted simulator candidates without lossy video re-encoding."""

import argparse, json, shutil, sys, hashlib
from pathlib import Path
import av, numpy as np, pyarrow as pa, pyarrow.parquet as pq
from prepare import OVER, WRIST, TASK, dump
from variants import TargetTracker

SIM = Path(__file__).resolve().parents[2] / "simulation"
sys.path.insert(0, str(SIM))
from sorting_env import BrickSortingEnv
from station_coordinates import StationCoordinates, constrain_simulation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    dest = root / "sim_corrected"
    if dest.exists():
        raise FileExistsError(dest)
    real = root / "real_corrected"
    info = json.loads((real / "meta/info.json").read_text())
    accepted = []
    for path in sorted(
        (root / "sim_candidates").glob("attempt-[0-9][0-9][0-9][0-9]/metadata.json")
    ):
        m = json.loads(path.read_text())
        if m["accepted"]:
            accepted.append((path.parent, m))
    if not accepted:
        raise ValueError("No accepted simulation episodes")
    env = BrickSortingEnv()
    coords = StationCoordinates(env.limits)
    constrain_simulation(env, coords)
    rows = []
    episodes = []
    markers = []
    provenance = []
    offset = 0
    try:
        for eid, (path, m) in enumerate(accepted):
            trace = json.loads((path / "trace.json").read_text())
            n = m["frames"]
            arr = np.load(path / "trajectory.npz")
            states = arr["state"]
            actions = arr["action"]
            assert (
                m["destination"] == "blue"
                and m["physical_grasp_verified"]
                and trace[-1]["success"]
            )
            assert (
                m["action_units"] == "so101_station_normalized_v1"
                and m["state_units"] == m["action_units"]
            )
            assert (
                states.shape == actions.shape == (n, 6)
                and np.isfinite(states).all()
                and np.isfinite(actions).all()
            )
            assert np.all(actions >= np.array([-100] * 5 + [0])) and np.all(
                actions <= 100
            )
            assert len(trace) == n and any(
                t["object_xyz"][2] > 0.06 and len(t["gripper_contact_bodies"]) == 2
                for t in trace
            )
            assert not trace[-1]["gripper_contact_bodies"]
            assert np.max(abs(np.diff([t["time"] for t in trace]) - 1 / 30)) < 1e-6
            initial = m.get("initial_target_xy")
            if initial is None:
                env.reset(m["seed"], target_color=m["target_color"], destination="blue")
                cam = env.model.camera("overhead")
                cid = cam.id
                point = env.data.body(env.brick_body_name).xpos
                local = env.data.cam_xmat[cid].reshape(3, 3).T @ (
                    point - env.data.cam_xpos[cid]
                )
                f = 240 / np.tan(np.deg2rad(float(cam.fovy[0])) / 2)
                initial = [
                    320 + f * local[0] / -local[2],
                    240 - f * local[1] / -local[2],
                ]
            ep = {
                "episode_index": eid,
                "tasks": [TASK],
                "length": n,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": offset,
                "dataset_to_index": offset + n,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
            for key, video in [(OVER, "overhead"), (WRIST, "wrist_mounted")]:
                target = dest / "videos" / key / "chunk-000" / f"file-{eid:03d}.mp4"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path / f"{video}.mp4", target)
                count = 0
                with av.open(str(target)) as c:
                    assert (
                        c.streams.video[0].width == 640
                        and c.streams.video[0].height == 480
                        and c.streams.video[0].average_rate == 30
                    )
                    for frame in c.decode(video=0):
                        if key == OVER:
                            rgb = frame.to_ndarray(format="rgb24")
                            if count == 0:
                                tracker = TargetTracker(rgb, initial)
                            xy, _ = tracker.update(rgb)
                            markers.append(xy)
                        count += 1
                assert count == n, (path, key, count, n)
                for field, value in [
                    ("chunk_index", 0),
                    ("file_index", eid),
                    ("from_timestamp", 0.0),
                    ("to_timestamp", n / 30),
                ]:
                    ep[f"videos/{key}/{field}"] = value
            for j in range(n):
                rows.append(
                    {
                        "action": actions[j].tolist(),
                        "observation.state": states[j].tolist(),
                        "timestamp": j / 30,
                        "frame_index": j,
                        "episode_index": eid,
                        "index": offset + j,
                        "task_index": 0,
                    }
                )
            episodes.append(ep)
            offset += n
            provenance.append(
                {
                    "episode_index": eid,
                    "source_attempt": path.name,
                    "seed": m["seed"],
                    "target_color": m["target_color"],
                    "frames": n,
                    "initial_target_xy": initial,
                    "trajectory_sha256": hashlib.file_digest(
                        (path / "trajectory.npz").open("rb"), "sha256"
                    ).hexdigest(),
                    "camera_alignment_verified": False,
                }
            )
    finally:
        env.close()
    (dest / "data/chunk-000").mkdir(parents=True)
    (dest / "meta/episodes/chunk-000").mkdir(parents=True)
    schema = pq.read_schema(next((real / "data").rglob("*.parquet")))
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        dest / "data/chunk-000/file-000.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(episodes),
        dest / "meta/episodes/chunk-000/file-000.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([{"task_index": 0, "task": TASK}]),
        dest / "meta/tasks.parquet",
    )
    info.update(total_episodes=len(episodes), total_frames=offset)
    for key in [OVER, WRIST]:
        info["features"][key]["info"].update({"video.codec": "h264", "video.g": None})
    dump(dest / "meta/info.json", info)
    shutil.copy2(real / "meta/stats.json", dest / "meta/stats.json")
    np.savez_compressed(
        root / "sim_markers.npz", xy=np.asarray(markers, dtype=np.float32)
    )
    seeds = sorted({p["seed"] for p in provenance})
    np.random.default_rng(20261115).shuffle(seeds)
    held = set(seeds[: max(1, round(0.2 * len(seeds)))])
    splits = {
        s: [
            p["episode_index"]
            for p in provenance
            if (p["seed"] in held) == (s == "validation")
        ]
        for s in ["train", "validation"]
    }
    dump(root / "sim_splits.json", splits)
    dump(
        root / "sim_provenance.json",
        {
            "episodes": provenance,
            "normalization": "copied exclusively from real training partition",
            "camera_alignment_verified": False,
            "status": "exported candidates, physical transfer compatibility unverified",
            "statistics_note": "image normalization is ImageNet in training; numeric normalization is real-only",
            "total_frames": offset,
        },
    )
    print(
        json.dumps(
            {
                "accepted_exported": len(episodes),
                "frames": offset,
                "split_counts": {k: len(v) for k, v in splits.items()},
            }
        )
    )


if __name__ == "__main__":
    main()
