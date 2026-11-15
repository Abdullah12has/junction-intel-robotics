#!/usr/bin/env python3
# Load a built dataset with lerobot (lerobot-env) and check it; decode random frames per camera to PNG.
# Usage: nice -n 19 taskset -c 0-3 <R>/envs/lerobot-env/bin/python -I <R>/app/vla/validate.py --root OUT --png-dir DIR [--report OUT.build_report.json]
# Exit code 0 when no check fails. The nearest-frame dt check is a WARN: listed in validate_report.json, does not fail.

import argparse
import glob
import json
import os
import random
import subprocess
import sys

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import av
import numpy as np
import pandas as pd

FPS = 30


def main():
    ap = argparse.ArgumentParser(description="Validate a dataset built by build_dataset.py.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--png-dir", required=True)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default=None, help="build report json, used for the nearest-frame dt check")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    checks = []

    def check(name, ok, detail="", level="FAIL"):
        checks.append({"check": name, "level": level, "pass": bool(ok), "detail": str(detail)})
        print(("PASS " if ok else level + " ") + name + ((" | " + str(detail)) if str(detail) else ""))

    def warn(name, ok, detail=""):
        check(name, ok, detail, level="WARN")

    with open(os.path.join(root, "meta/info.json"), encoding="utf-8") as fh:
        info = json.load(fh)
    feats = info["features"]
    video_keys = [k for k, v in feats.items() if v.get("dtype") == "video"]
    data = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(root, "data/chunk-*/file-*.parquet")))],
                     ignore_index=True)
    eps = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(root, "meta/episodes/chunk-*/file-*.parquet")))],
                    ignore_index=True)
    tasks_df = pd.read_parquet(os.path.join(root, "meta/tasks.parquet"))
    if "task" in tasks_df.columns:
        text_of = dict(zip(tasks_df["task_index"].astype(int), tasks_df["task"]))
    else:
        text_of = {int(i): t for i, t in zip(tasks_df["task_index"], tasks_df.index)}

    n_rows = len(data)
    check("row count equals info total_frames", n_rows == info["total_frames"], "%d vs %d" % (n_rows, info["total_frames"]))
    check("episode count equals info total_episodes", len(eps) == info["total_episodes"],
          "%d vs %d" % (len(eps), info["total_episodes"]))
    frm = eps["dataset_from_index"].to_numpy()
    to = eps["dataset_to_index"].to_numpy()
    length = eps["length"].to_numpy()
    check("episode length equals dataset_to - dataset_from", bool((length == to - frm).all()))
    check("episodes tile the global index without gaps", bool((frm[1:] == to[:-1]).all()) and frm[0] == 0 and to[-1] == n_rows)
    rows_per_ep = data.groupby("episode_index").size()
    check("rows per episode equal episodes.length",
          all(int(rows_per_ep.get(int(e), -1)) == int(l) for e, l in zip(eps["episode_index"], length)))
    check("index column equals 0..N-1", bool((data["index"].to_numpy() == np.arange(n_rows)).all()))
    fi_ok, ts_err = True, 0.0
    for _, g in data.groupby("episode_index"):
        f = g["frame_index"].to_numpy()
        fi_ok = fi_ok and bool((f == np.arange(len(f))).all())
        ts_err = max(ts_err, float(np.abs(g["timestamp"].to_numpy(dtype=np.float64) - f / FPS).max()))
    check("frame_index per episode is 0..n-1", fi_ok)
    check("timestamp equals frame_index / fps", ts_err < 1e-5, "max err %.2e s" % ts_err)

    act = np.stack(data["action"].to_numpy()).astype(np.float64)
    st = np.stack(data["observation.state"].to_numpy()).astype(np.float64)
    nan_a, nan_s = int(np.isnan(act).sum()), int(np.isnan(st).sum())
    check("no NaN or inf in action, state, timestamp",
          nan_a == 0 and nan_s == 0 and bool(np.isfinite(act).all() and np.isfinite(st).all()),
          "nan action %d, nan state %d" % (nan_a, nan_s))
    ranges_ok = bool(((act >= -180) & (act <= 180)).all() and ((st >= -180) & (st <= 180)).all()
                     and (act[:, 5] >= 0).all() and (act[:, 5] <= 100).all()
                     and (st[:, 5] >= 0).all() and (st[:, 5] <= 100).all())
    check("joint values inside -180..180 deg and gripper 0..100", ranges_ok,
          "action min %s max %s | state min %s max %s" % (np.round(act.min(axis=0), 2).tolist(),
                                                           np.round(act.max(axis=0), 2).tolist(),
                                                           np.round(st.min(axis=0), 2).tolist(),
                                                           np.round(st.max(axis=0), 2).tolist()))
    ti_by_ep = data.groupby("episode_index")["task_index"].first().to_dict()
    tasks_ok = all(text_of.get(int(ti_by_ep[int(e)])) == list(t)[0] for e, t in zip(eps["episode_index"], eps["tasks"]))
    check("tasks.parquet text matches every episode", tasks_ok, "tasks %s" % json.dumps(list(text_of.values())))

    for key in video_keys:
        rel = info["video_path"].format(video_key=key, chunk_index=0, file_index=0)
        path = os.path.join(root, rel)
        exists = os.path.isfile(path)
        check("video file exists: %s" % key, exists, rel)
        if not exists:
            continue
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                "stream=codec_name,pix_fmt,width,height,r_frame_rate", "-of", "json", path],
                               capture_output=True, text=True, check=True)
        s = json.loads(probe.stdout)["streams"][0]
        check("codec, pix_fmt, fps, size match team settings: %s" % key,
              s["codec_name"] == "av1" and s["pix_fmt"] == "yuv420p" and s["r_frame_rate"] == "30/1"
              and s["width"] == 640 and s["height"] == 480,
              "codec %s pix %s fps %s size %sx%s" % (s["codec_name"], s["pix_fmt"], s["r_frame_rate"], s["width"], s["height"]))
        c = av.open(path)
        vs = c.streams.video[0]
        n_pk = sum(1 for p in c.demux(vs) if p.pts is not None)
        c.close()
        check("video packets equal total_frames: %s" % key, n_pk == n_rows, "%d vs %d" % (n_pk, n_rows))
        f_from = eps["videos/%s/from_timestamp" % key].to_numpy(dtype=np.float64)
        f_to = eps["videos/%s/to_timestamp" % key].to_numpy(dtype=np.float64)
        check("episode video offsets match dataset index: %s" % key,
              bool((np.abs(f_from - frm / FPS) < 1e-6).all() and (np.abs(f_to - to / FPS) < 1e-6).all()))

    if args.report and os.path.isfile(args.report):
        with open(args.report, encoding="utf-8") as fh:
            rep = json.load(fh)
        worst = rep.get("max_nearest_frame_dt_s", {})
        warn("nearest camera frame within 2 frames of each joint sample", all(v <= 2.0 / FPS for v in worst.values()),
             json.dumps(worst))
        if rep.get("dt_summary"):
            print("build dt summary per camera: " + json.dumps(rep["dt_summary"]))

    os.makedirs(args.png_dir, exist_ok=True)
    try:
        from PIL import Image
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset(repo_id="local/validate", root=root, video_backend="pyav")
        check("lerobot loads the dataset", True, "num_frames %d num_episodes %d fps %s" % (ds.num_frames, ds.num_episodes, ds.fps))
        check("lerobot counts equal info", ds.num_frames == info["total_frames"] and ds.num_episodes == info["total_episodes"])
        rnd = random.Random(args.seed)
        for key in video_keys:
            label = key.replace("observation.images.", "").replace(" ", "_")
            lum = []
            for _ in range(args.samples):
                idx = rnd.randrange(ds.num_frames)
                item = ds[idx]
                img = (item[key].permute(1, 2, 0).cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
                lum.append(float(img.std()))
                name = "%s_idx%06d_ep%03d_f%04d.png" % (label, idx, int(item["episode_index"]), int(item["frame_index"]))
                Image.fromarray(img).save(os.path.join(args.png_dir, name))
            check("decoded frames are not blank: %s" % key, min(lum) > 1.0, "pixel std per frame %s" % [round(x, 1) for x in lum])
    except Exception as exc:  # noqa: BLE001
        check("lerobot loads the dataset", False, "%s: %s" % (type(exc).__name__, str(exc)[:400]))

    failed = [c["check"] for c in checks if c["level"] == "FAIL" and not c["pass"]]
    warnings = [c["check"] for c in checks if c["level"] == "WARN" and not c["pass"]]
    result = "PASS" if not failed else "FAIL"
    with open(os.path.join(args.png_dir, "validate_report.json"), "w", encoding="utf-8") as fh:
        json.dump({"root": root, "result": result, "failed": failed, "warnings": warnings, "checks": checks}, fh, indent=1)
    print("RESULT %s (%d checks, %d failed, %d warnings)" % (result, len(checks), len(failed), len(warnings)))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
