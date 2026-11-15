import argparse, json
from pathlib import Path
import av, numpy as np, pyarrow.parquet as pq
from PIL import Image, ImageDraw
from variants import TargetTracker, transform
from prepare import OVER, dump


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    root = a.root
    ds = root / "real_corrected"
    info = json.loads((ds / "meta/info.json").read_text())
    ann = json.loads((root / "annotations.json").read_text())
    eps = pq.read_table(ds / "meta/episodes").to_pylist()
    n = info["total_frames"]
    xy = np.full((n, 2), -1, dtype=np.float32)
    scores = np.zeros(n, dtype=np.float32)
    requests = {}
    for e in eps:
        path = ds / info["video_path"].format(
            video_key=OVER,
            chunk_index=e[f"videos/{OVER}/chunk_index"],
            file_index=e[f"videos/{OVER}/file_index"],
        )
        start = round(e[f"videos/{OVER}/from_timestamp"] * 30)
        for j in range(e["length"]):
            requests.setdefault(path, {})[start + j] = (e, j)
    previews = {}
    reports = []
    for path, lookup in requests.items():
        with av.open(str(path)) as c:
            for i, f in enumerate(c.decode(video=0)):
                e, j = lookup[i]
                eid = e["episode_index"]
                rgb = f.to_ndarray(format="rgb24")
                index = e["dataset_from_index"] + j
                if j == 0:
                    tracker = TargetTracker(rgb, ann[eid]["initial_target_xy"])
                    shown = 0
                    last = None
                point, score = tracker.update(rgb)
                xy[index] = point
                scores[index] = score
                if min(point) >= 0:
                    shown += 1
                    last = (rgb.copy(), point)
                if j == 0:
                    previews[eid] = [
                        transform(rgb, 0, np.random.default_rng(0), xy=point)
                    ]
                if j == e["length"] - 1:
                    if last:
                        previews[eid].append(
                            transform(last[0], 0, np.random.default_rng(0), xy=last[1])
                        )
                    reports.append(
                        {
                            "episode": eid,
                            "visible_frames": shown,
                            "total_frames": e["length"],
                            "first_loss_frame": shown if shown < e["length"] else None,
                            "initial_point_reviewed": ann[eid]["target_point_reviewed"],
                        }
                    )
    np.savez_compressed(root / "real_markers.npz", xy=xy, score=scores)
    dump(
        root / "marker_report.json",
        {
            "method": "causal fixed-template local tracker, latch suppression on first loss; no reacquisition",
            "episodes": reports,
            "deployment_tracker_validated": False,
        },
    )
    for start in range(0, 64, 8):
        sheet = Image.new("RGB", (1280, 4 * 270), "white")
        d = ImageDraw.Draw(sheet)
        for k, eid in enumerate(range(start, start + 8)):
            for col, im in enumerate(previews[eid]):
                x = (k % 2) * 640 + col * 320
                y = (k // 2) * 270
                sheet.paste(Image.fromarray(im).resize((320, 240)), (x, y + 25))
                d.text(
                    (x + 3, y + 3),
                    f"Ep{eid} " + ("initial" if col == 0 else "last visible"),
                    fill="black",
                )
        sheet.save(root / "qa" / f"tracks-{start:02}.jpg")
    print(
        json.dumps(
            {
                "frames": n,
                "marked_frames": int((xy[:, 0] >= 0).sum()),
                "minimum_visible": min(r["visible_frames"] for r in reports),
            }
        )
    )


if __name__ == "__main__":
    main()
