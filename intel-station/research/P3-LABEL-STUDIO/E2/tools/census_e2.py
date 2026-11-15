import json, os, hashlib
import numpy as np
import pandas as pd

ROOT = "/home/intel/intel-junction-2026/data/studio-datasets"
OUT = "/home/intel/intel-junction-2026/work/P3-LABEL-STUDIO/E2/out/census_e2.json"

eps = []
datasets = {}
for d in sorted(os.listdir(ROOT)):
    base = os.path.join(ROOT, d)
    if not os.path.isdir(base):
        continue
    info = json.load(open(os.path.join(base, "meta/info.json")))
    import glob
    meta = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(base, "meta/episodes/chunk-*/file-*.parquet")))], ignore_index=True)
    data = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(base, "data/chunk-*/file-*.parquet")))], ignore_index=True)
    tasks = pd.read_parquet(os.path.join(base, "meta/tasks.parquet"))
    tmap = {int(r["task_index"]): str(t) for t, r in tasks.iterrows()}
    vids = sorted(os.listdir(os.path.join(base, "videos"))) if os.path.isdir(os.path.join(base, "videos")) else []
    cams = []
    if os.path.isdir(os.path.join(base, "videos", "observation.images.realsense 243622061421")) or True:
        for c in ["observation.images.realsense 243622061421", "observation.images.realsense 243722064316", "observation.images.usb2_0_cam1 external camera"]:
            if os.path.isdir(os.path.join(base, "videos", c)):
                cams.append(c)
    datasets[d] = dict(
        total_episodes=info.get("total_episodes"), total_frames=info.get("total_frames"),
        fps=info.get("fps"), meta_rows=len(meta), data_rows=len(data),
        episodes_in_data=int(data.episode_index.nunique()),
        tasks=list(tmap.values()), cameras=cams, video_dirs=vids,
        joint_names=info.get("features", {}).get("action", {}).get("names"),
    )
    for ep, g in data.sort_values(["episode_index", "frame_index"]).groupby("episode_index"):
        act = np.stack([np.asarray(x, dtype=np.float64) for x in g["action"].values])
        st = np.stack([np.asarray(x, dtype=np.float64) for x in g["observation.state"].values])
        fi = g["frame_index"].values
        ts = g["timestamp"].values
        m = meta[meta.episode_index == ep]
        tindex = int(g["task_index"].iloc[0])
        eps.append(dict(
            dataset=d, uuid_prefix=d[:8], episode_index=int(ep), n=int(len(g)),
            frame_index_contiguous=bool(np.array_equal(fi, np.arange(len(fi)))),
            t0=round(float(ts[0]), 3), t1=round(float(ts[-1]), 3),
            meta_length=int(m["length"].iloc[0]),
            task_index=tindex, task_text=tmap.get(tindex),
            act_sha=hashlib.sha256(act.tobytes()).hexdigest()[:16],
            act_min=act.min(axis=0).round(2).tolist(), act_max=act.max(axis=0).round(2).tolist(),
            act=act,
        ))

# containment: an episode is a copy/window if its action rows equal a window of another (longer or equal, earlier) episode
order = sorted(range(len(eps)), key=lambda i: (-eps[i]["n"], eps[i]["dataset"], eps[i]["episode_index"]))
rank = {i: k for k, i in enumerate(order)}
for i in range(len(eps)):
    eps[i]["contained_in"] = None
    eps[i]["window"] = None
for i in order:
    a = eps[i]["act"]
    for j in order:
        if rank[j] >= rank[i]:
            break
        b = eps[j]["act"]
        if len(b) < len(a):
            continue
        for off in range(0, len(b) - len(a) + 1):
            if np.array_equal(b[off:off + len(a)], a):
                eps[i]["contained_in"] = f'{eps[j]["dataset"][:8]}:{eps[j]["episode_index"]}'
                eps[i]["window"] = [off, off + len(a)]
                break
        if eps[i]["contained_in"]:
            break

rows = [{k: v for k, v in e.items() if k != "act"} for e in eps]
phys = [r for r in rows if r["contained_in"] is None]
out = dict(snapshot_utc="2026-11-14T20:45Z", n_datasets=len(datasets), n_episodes=len(rows),
           physical_candidates=[f'{r["dataset"][:8]}:{r["episode_index"]}' for r in phys],
           datasets=datasets, episodes=rows)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"), indent=1)
print("datasets", len(datasets), "episodes", len(rows), "uncontained", len(phys))
for r in rows:
    print(r["dataset"][:8], r["episode_index"], r["n"], r["t1"], r["act_sha"], "copy_of=" + str(r["contained_in"]), r["window"], r["task_text"])
