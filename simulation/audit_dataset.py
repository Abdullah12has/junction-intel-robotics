"""Read-only LeRobot v3 structural audit; does not certify task success."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

STATION = json.loads((Path(__file__).parent / "station.json").read_text())


def audit(root, task_mode="inspection"):
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    errors, warnings = [], []
    if info.get("codebase_version") != "v3.0":
        errors.append("Expected LeRobot v3.0")
    if info.get("fps") != STATION["fps"]:
        errors.append("Expected 30 FPS")
    for key in ("action", "observation.state"):
        if info["features"].get(key, {}).get("names") != STATION["joint_names"]:
            errors.append(f"Joint order mismatch in {key}")
    for camera in STATION["camera_rename_map"]:
        if camera not in info["features"]:
            errors.append(f"Missing camera: {camera}")
        if not list((root / "videos" / camera).glob("chunk-*/*.mp4")):
            errors.append(f"No video files for {camera}")
    paths = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not paths:
        raise ValueError("No episode data files")
    table = pa.concat_tables([pq.read_table(p) for p in paths])
    if table.num_rows != info["total_frames"]:
        errors.append("Frame count differs from metadata")
    ep = np.asarray(table["episode_index"])
    ids = sorted(int(i) for i in np.unique(ep))
    if ids != list(range(info["total_episodes"])):
        errors.append("Episode IDs/count differ from metadata")
    ranges = {}
    for key in ("action", "observation.state"):
        values = np.asarray(table[key].to_pylist())
        if values.shape != (table.num_rows, 6) or not np.isfinite(values).all():
            errors.append(f"Invalid values or shape: {key}")
            continue
        ranges[key] = {"min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist()}
        lower = np.array([-100] * 5 + [0])
        if np.any(values < lower) or np.any(values > 100):
            warnings.append(f"{key} exceeds documented normalized limits; review before training, do not silently clamp")
    times = np.asarray(table["timestamp"])
    frames = np.asarray(table["frame_index"])
    episodes = []
    for i in ids:
        mask = ep == i
        t, f = times[mask], frames[mask]
        if not np.array_equal(f, np.arange(len(f))):
            errors.append(f"Episode {i}: frame indices are not contiguous")
        if not np.isfinite(t).all() or not np.allclose(t, np.arange(len(t)) / info["fps"], atol=1e-4):
            errors.append(f"Episode {i}: timestamps do not match FPS")
        episodes.append({"episode_index": i, "frames": len(t), "seconds": len(t) / info["fps"]})
    tasks = pq.read_table(root / "meta/tasks.parquet").to_pylist()
    allowed = {STATION['task']} if task_mode == "inspection" else {
        STATION['sorting_task_template'].format(color=c, plate=p)
        for c in STATION['sorting_colors'] for p in STATION['sorting_plates']}
    if not tasks or not {r['task'] for r in tasks}.issubset(allowed):
        errors.append(f"Task labels differ from the {task_mode} task family")
    task_lookup = {r['task_index']: r['task'] for r in tasks}
    for episode in episodes:
        indices = set(table.filter(pa.compute.equal(table['episode_index'], episode['episode_index']))['task_index'].to_pylist())
        if len(indices) != 1 or not indices.issubset(task_lookup):
            errors.append(f"Episode {episode['episode_index']}: invalid or changing task")
        else:
            episode['task'] = task_lookup[next(iter(indices))]
    if len(ids) < 50:
        warnings.append(f"Only {len(ids)} episodes: collect/review 50 for the planned experiment")
    return {"root": str(root.resolve()), "structural_checks_passed": not errors,
            "errors": errors, "warnings": warnings, "frames": table.num_rows,
            "episodes": episodes, "tasks": tasks, "ranges": ranges,
            "video_decoding_verified": False, "physical_success_reviewed": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset", type=Path)
    p.add_argument("--task-mode", choices=("inspection", "sorting"), default="inspection")
    p.add_argument("--output", type=Path, default=Path("outputs/dataset-audit.json"))
    a = p.parse_args()
    report = audit(a.dataset, a.task_mode)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(report["errors"]))
