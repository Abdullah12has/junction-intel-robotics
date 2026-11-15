import json
import os

import numpy as np
import pyarrow.parquet as pq

ROOT = "/home/intel/.local/share/physicalai/datasets"
W = "/home/intel/intel-junction-2026/work/P2-B-OVRUN2/E1"
OUT = f"{W}/data"
SETS = [
    ("trainBlueV1", "d2fd0a80-5364-4f38-bc81-d82d348c5a06"),
    ("heldBlueV2", "e2eae6c8-fe98-4cfb-82eb-e5e59c7cc8af"),
    ("heldGreenV2", "ba9eafba-a9c3-47a0-b941-34c20a2bbf55"),
]
CAMS = ["realsense 243722064316", "realsense 243622061421"]
FPS = 30


def main():
    os.makedirs(OUT, exist_ok=True)
    man = {"sets": {}, "episodes": []}
    for tag, dsid in SETS:
        base = f"{ROOT}/{dsid}"
        data = pq.read_table(f"{base}/data/chunk-000/file-000.parquet").to_pandas()
        eps = pq.read_table(f"{base}/meta/episodes/chunk-000/file-000.parquet").to_pandas()
        man["sets"][tag] = {"dataset_id": dsid, "rows": int(len(data)), "episodes": int(len(eps))}
        vids_set = set()
        for _, r in eps.iterrows():
            e = int(r["episode_index"])
            L = int(r["length"])
            rows = data[data["episode_index"] == e]
            assert len(rows) == L, (tag, e, len(rows), L)
            S = np.stack(rows["observation.state"].to_numpy()).astype(np.float32)
            A = np.stack(rows["action"].to_numpy()).astype(np.float32)
            g0s, vids = {}, {}
            for cam in CAMS:
                g0s[cam] = int(round(float(r[f"videos/observation.images.{cam}/from_timestamp"]) * FPS))
                ci = int(r[f"videos/observation.images.{cam}/chunk_index"])
                fi = int(r[f"videos/observation.images.{cam}/file_index"])
                vids[cam] = f"{base}/videos/observation.images.{cam}/chunk-{ci:03d}/file-{fi:03d}.mp4"
                assert os.path.exists(vids[cam]), vids[cam]
            assert len(set(g0s.values())) == 1, (tag, e, g0s)
            vids_set.add(tuple(vids[c] for c in CAMS))
            npz = f"{OUT}/{tag}_ep{e}.npz"
            np.savez(npz, state=S, action=A)
            man["episodes"].append({"tag": tag, "dataset_id": dsid, "episode_index": e, "length": L,
                                    "g0": g0s[CAMS[0]], "tasks_recorded": [str(x) for x in r["tasks"]],
                                    "videos": [vids[c] for c in CAMS], "npz": npz})
        man["sets"][tag]["video_files"] = [list(v) for v in vids_set]
    with open(f"{OUT}/prep2.json", "w") as fh:
        json.dump(man, fh, indent=1)
    for ep in man["episodes"]:
        print(ep["tag"], ep["dataset_id"][:8], "ep", ep["episode_index"], "len", ep["length"], "g0", ep["g0"], ep["tasks_recorded"])
    print(json.dumps(man["sets"]))


if __name__ == "__main__":
    main()
