"""Create immutable-source-derived metadata, grouped splits, and target QA frames."""

import argparse, hashlib, json, shutil
from pathlib import Path
import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

OVER = "observation.images.realsense 243622061421"
WRIST = "observation.images.usb2_0_cam1 external camera"
TASK = "Pick up a cube and put it on the blue plate."


def dump(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    src = a.source.resolve()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    ann = json.loads(Path(__file__).with_name("annotations.json").read_text())
    info = json.loads((src / "meta/info.json").read_text())
    eps = pq.read_table(src / "meta/episodes").to_pylist()
    data = pq.read_table(src / "data").to_pydict()
    groups = ann["groups"]
    assert sorted(sum(groups, [])) == list(range(64))
    # Entire contiguous reset families; fixed seed, no outcome-based optimization.
    order = np.random.default_rng(20261115).permutation(len(groups)).tolist()
    bysplit = {"validation": order[:4], "test": order[4:8], "train": order[8:]}
    splits = {k: sorted(e for g in gs for e in groups[g]) for k, gs in bysplit.items()}
    dump(
        out / "splits.json",
        {
            "seed": 20261115,
            "groups": groups,
            "group_assignment": bysplit,
            "episodes": splits,
            "review": "conservative manual reset-family grouping",
        },
    )
    root = out / "real_corrected"
    if not root.exists():
        shutil.copytree(src, root)
    # All original data rows/videos remain byte-identical. Only derived metadata is rewritten.
    pq.write_table(
        pa.Table.from_pylist([{"task_index": 0, "task": TASK}]),
        root / "meta/tasks.parquet",
    )
    for file in (root / "meta/episodes").rglob("*.parquet"):
        table = pq.read_table(file)
        rows = table.to_pylist()
        for r in rows:
            r["tasks"] = [TASK]
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), file)
    # Remove unused close view from feature schema so policy factory cannot silently add it.
    info["features"] = {
        k: v
        for k, v in info["features"].items()
        if v["dtype"] != "video" or k in (OVER, WRIST)
    }
    # Arbitrary group membership is represented in splits.json, not a misleading contiguous split.
    info.pop("splits", None)
    dump(root / "meta/info.json", info)
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.utils.utils import unflatten_dict

    stats = []
    for e in eps:
        if e["episode_index"] in splits["train"]:
            raw = unflatten_dict(
                {k: v for k, v in e.items() if k.startswith("stats/")}
            )["stats"]
            stats.append(
                {
                    k: {kk: np.asarray(vv) for kk, vv in v.items()}
                    for k, v in raw.items()
                    if k in info["features"]
                }
            )
    merged = aggregate_stats(stats)
    # Exact train-only numeric stats, including quantiles, recomputed from raw rows.
    mask = np.isin(data["episode_index"], splits["train"])
    for k in ["action", "observation.state"]:
        v = np.asarray(data[k])[mask]
        merged[k] = {
            "min": v.min(0),
            "max": v.max(0),
            "mean": v.mean(0),
            "std": v.std(0),
            "count": np.array([len(v)]),
            **{
                f"q{p:02}": np.quantile(v, p / 100, axis=0) for p in [1, 10, 50, 90, 99]
            },
        }
    dump(
        root / "meta/stats.json",
        {k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in merged.items()},
    )
    provenance = {
        "source": str(src),
        "task": TASK,
        "trim": "none; preserve full skills and timing",
        "normalization_episodes": splits["train"],
        "camera_roles": {"overhead": OVER, "wrist": WRIST},
        "source_sha256": {
            str(p.relative_to(src)): hashlib.file_digest(
                p.open("rb"), "sha256"
            ).hexdigest()
            for p in src.rglob("*")
            if p.is_file()
        },
    }
    dump(out / "provenance.json", provenance)
    requests = {}
    initial = {}
    for e in eps:
        path = src / info["video_path"].format(
            video_key=OVER,
            chunk_index=e[f"videos/{OVER}/chunk_index"],
            file_index=e[f"videos/{OVER}/file_index"],
        )
        requests.setdefault(path, {})[
            round(e[f"videos/{OVER}/from_timestamp"] * 30)
        ] = e["episode_index"]
    for path, lookup in requests.items():
        with av.open(str(path)) as c:
            for i, f in enumerate(c.decode(video=0)):
                if i in lookup:
                    initial[lookup[i]] = f.to_ndarray(format="rgb24")
    qa = out / "qa"
    qa.mkdir(exist_ok=True)
    manifest = []
    for i, e in enumerate(eps):
        color, x, y = ann["targets"][i]
        xy = [x * 2, y * 2]
        manifest.append(
            {
                "episode_index": i,
                "target_color": color,
                "initial_target_xy": xy,
                "layout_group": next(g for g, ids in enumerate(groups) if i in ids),
                "split": next(s for s, ids in splits.items() if i in ids),
                "outcome": "blue_placement_visible_in_sampled_review",
                "review_scope": ann["review"],
                "trim_start": 0,
                "trim_end": e["length"],
                "target_point_reviewed": ann.get("initial_points_reviewed", False),
            }
        )
    for start in range(0, 64, 16):
        sheet = Image.new("RGB", (1280, 4 * 270), "white")
        d = ImageDraw.Draw(sheet)
        for j, i in enumerate(range(start, start + 16)):
            im = Image.fromarray(initial[i])
            color, x, y = ann["targets"][i]
            draw = ImageDraw.Draw(im)
            xx, yy = x * 2, y * 2
            draw.ellipse(
                (xx - 18, yy - 18, xx + 18, yy + 18), outline="magenta", width=2
            )
            im.save(qa / f"initial-{i:02}.png")
            ox = (j % 4) * 320
            oy = (j // 4) * 270
            sheet.paste(im.resize((320, 240)), (ox, oy + 25))
            d.text((ox + 4, oy + 4), f"{i}: {color}", fill="black")
        sheet.save(qa / f"targets-{start:02}.jpg")
    dump(out / "annotations.json", manifest)
    print(
        json.dumps(
            {
                "output": str(out),
                "split_counts": {k: len(v) for k, v in splits.items()},
                "frames": len(data["index"]),
            }
        )
    )


if __name__ == "__main__":
    main()
