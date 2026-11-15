#!/usr/bin/env python3
"""E1 (P2-D-BUILD): validate built curated datasets against their source and write contact sheets.

Run: python3 -I validate_curated.py EPISODES_CSV SRC_ROOT RESULTS_JSON SHEET_DIR BUILD_JSON [--zip ZIP_DIR]
Checks per class: episode and frame counts, fps, robot_type, feature keys, task text per episode,
exact equality of action and observation.state against the source window, first and last frame
decode per camera (non-black), mean absolute difference of the first output frame vs the source
first frame (0-255 scale). Writes one contact sheet per class. With --zip, writes ZIP_DIR/<name>.zip
only for classes that pass, with the dataset files at the archive root (the lerobot_v3 layout).
"""

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, "/home/intel/intel-junction-2026/app/data")
from curate import AUTO_KEYS, TASKS, decode_frames, episode_row, read_rows, window  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

TILE_W, TILE_H = 160, 120
# OD-10 (coordinator, FINAL): PP classes are defined by the physical release plate only.
# Top view: BLUE = left plate, GREEN = right plate; positive shoulder_pan = toward BLUE (FACTS-LIVE 13:52Z).
PLATE_SIGN = {"PP_BLUE": 1, "PP_GREEN": -1}


def release_pan(action):
    """shoulder_pan at the last gripper opening (gripper action crosses 4 -> 5) of an episode, or None."""
    grip = action[:, 5]
    crossings = [i for i in range(1, len(grip)) if grip[i] > 5.0 and grip[i - 1] <= 4.0]
    return float(action[crossings[-1], 0]) if crossings else None


def tile(arr_hwc_uint8):
    return Image.fromarray(arr_hwc_uint8).resize((TILE_W, TILE_H), Image.BILINEAR)


def to_uint8(chw_float):
    return (chw_float.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()


def check_class(name, info, src_by_id, fps, sheet_dir):
    cls = info["class"]
    problems = []
    out_dir = Path(info["path"])
    ds = LeRobotDataset(f"local/{name}", root=out_dir)
    first_src = next(iter(src_by_id.values()))
    src_keys = {k for k in first_src.meta.features if k not in AUTO_KEYS}
    out_keys = {k for k in ds.meta.features if k not in AUTO_KEYS}
    cam_keys = info["cameras"]
    if ds.meta.total_episodes != len(info["per_episode"]):
        problems.append(f"episodes {ds.meta.total_episodes} != {len(info['per_episode'])}")
    if ds.meta.total_frames != info["frames"]:
        problems.append(f"frames {ds.meta.total_frames} != {info['frames']}")
    if int(ds.fps) != fps:
        problems.append(f"fps {ds.fps} != {fps}")
    if ds.meta.robot_type != first_src.meta.robot_type:
        problems.append(f"robot_type {ds.meta.robot_type} != {first_src.meta.robot_type}")
    if src_keys != out_keys:
        problems.append(f"feature keys differ: {sorted(src_keys ^ out_keys)}")
    ep_rows = list(ds.meta.episodes)
    cams_report, eps_report = {}, []
    sheet_rows = []
    for j, pe in enumerate(info["per_episode"]):
        erow = ep_rows[j]
        start, end = int(erow["dataset_from_index"]), int(erow["dataset_to_index"])
        if int(erow["length"]) != pe["frames"] or end - start != pe["frames"]:
            problems.append(f"episode {j}: length {erow['length']} != {pe['frames']}")
        if list(erow["tasks"]) != [TASKS[cls]]:
            problems.append(f"episode {j}: tasks {list(erow['tasks'])}")
        src = src_by_id[pe["dataset_id"]]
        srow = episode_row(src, pe["source_episode"])
        g0 = int(srow["dataset_from_index"]) + pe["window_first_frame"]
        out_hf, src_hf = ds.hf_dataset[start:end], src.hf_dataset[g0 : g0 + pe["frames"]]
        exact = {}
        for key in ("action", "observation.state"):
            a = np.asarray(out_hf[key], dtype=np.float32)
            b = np.asarray(src_hf[key], dtype=np.float32)
            exact[key] = bool(a.shape == b.shape and np.array_equal(a, b))
            if not exact[key]:
                problems.append(f"episode {j}: {key} values differ from source window")
        act_out = np.asarray(out_hf["action"], dtype=np.float32)
        rel = release_pan(act_out) if cls in PLATE_SIGN else None
        if cls in PLATE_SIGN and (rel is None or np.sign(rel) != PLATE_SIGN[cls]):
            problems.append(f"episode {j}: release pan {rel} is not on the {cls} plate (OD-10 physical check)")
        row_tiles = []
        ep_entry = {"out_episode": j, "source_episode": pe["source_episode"], "frames": pe["frames"], "exact": exact, "release_pan": rel, "cams": {}}
        for cam in cam_keys:
            first_out = to_uint8(ds[start][cam])
            last_out = to_uint8(ds[end - 1][cam])
            src_first = decode_frames(src, srow, cam, [pe["window_first_frame"]], fps)[0]
            diff = float(np.abs(first_out.astype(np.int16) - src_first.astype(np.int16)).mean())
            mean_px = float(first_out.mean())
            ep_entry["cams"][cam] = {"first_vs_source_mad": round(diff, 2), "first_mean_px": round(mean_px, 1), "last_mean_px": round(float(last_out.mean()), 1), "shape": list(first_out.shape)}
            if first_out.shape != (480, 640, 3) or mean_px < 2.0 or float(last_out.mean()) < 2.0:
                problems.append(f"episode {j} {cam}: frame shape/black check failed")
            row_tiles.extend([tile(first_out), tile(last_out)])
        cams_report.setdefault("per_episode", []).append(ep_entry)
        eps_report.append(ep_entry)
        sheet_rows.append((j, pe["source_episode"], row_tiles))
    # contact sheet: one row per episode, columns = (cam1 first, cam1 last, cam2 first, cam2 last, ...)
    n_cols = 2 * len(cam_keys)
    header = 18
    sheet = Image.new("RGB", (n_cols * TILE_W, header + len(sheet_rows) * TILE_H), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    labels = []
    for c in cam_keys:
        labels += [f"{c.split('.')[-1]} first", f"{c.split('.')[-1]} last"]
    for k, lab in enumerate(labels):
        draw.text((k * TILE_W + 4, 3), lab[:28], fill=(255, 255, 255))
    for r, (j, src_ep, tiles) in enumerate(sheet_rows):
        for k, t in enumerate(tiles):
            sheet.paste(t, (k * TILE_W, header + r * TILE_H))
        draw.text((4, header + r * TILE_H + 2), f"out {j} = src ep {src_ep}", fill=(255, 255, 0))
    sheet_path = sheet_dir / f"contact_{name}.png"
    sheet.save(sheet_path)
    report = {
        "name": name,
        "class": cls,
        "path": str(out_dir),
        "episodes": ds.meta.total_episodes,
        "frames": ds.meta.total_frames,
        "problems": problems,
        "pass": not problems,
        "contact_sheet": str(sheet_path),
        "per_episode": eps_report,
    }
    return report


def pack(name, out_dir, zip_dir):
    zip_path = zip_dir / f"{name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(out_dir).as_posix())
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    checks = {
        "meta/info.json": "meta/info.json" in names,
        "meta/tasks.parquet": "meta/tasks.parquet" in names,
        "data_parquet": any(n.startswith("data/") and "/file-" in n and n.endswith(".parquet") for n in names),
        "no_v2_markers": not any(n.endswith(("meta/tasks.jsonl", "meta/episodes.jsonl")) for n in names),
        "no_absolute_or_dotdot": not any(n.startswith("/") or ".." in n.split("/") for n in names),
    }
    return {"zip": str(zip_path), "entries": len(names), "bytes": zip_path.stat().st_size, "adapter_markers": checks, "ok": all(checks.values())}


def main():
    csv_path, src_root = Path(sys.argv[1]), Path(sys.argv[2])
    results_path, sheet_dir = Path(sys.argv[3]), Path(sys.argv[4])
    build_json = json.loads(Path(sys.argv[5]).read_text(encoding="utf-8"))
    zip_dir = Path(sys.argv[7]) if len(sys.argv) > 7 and sys.argv[6] == "--zip" else None
    sheet_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(csv_path)
    src_by_id = {}
    for r in rows:
        if r["dataset_id"] not in src_by_id:
            src_by_id[r["dataset_id"]] = LeRobotDataset(f"local/{r['dataset_id']}", root=src_root / r["dataset_id"])
    fps = int(build_json["fps"])
    out = {"classes": {}, "all_pass": True}
    for name, info in build_json["classes"].items():
        if "per_episode" not in info:
            out["classes"][name] = {"status": info.get("status", "not built")}
            continue
        rep = check_class(name, info, src_by_id, fps, sheet_dir)
        if zip_dir is not None and rep["pass"]:
            zip_dir.mkdir(parents=True, exist_ok=True)
            rep["zip"] = pack(name, Path(info["path"]), zip_dir)
            rep["pass"] = rep["pass"] and rep["zip"]["ok"]
        out["classes"][name] = rep
        out["all_pass"] = out["all_pass"] and rep["pass"]
        status = "PASS" if rep["pass"] else "FAIL"
        print(f"{name}: {status} episodes={rep['episodes']} frames={rep['frames']} problems={rep['problems']}", flush=True)
    results_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"all_pass={out['all_pass']}")


if __name__ == "__main__":
    main()
