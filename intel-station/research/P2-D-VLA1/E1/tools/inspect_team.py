import glob
import inspect
import json
import sys

import pyarrow.parquet as pq

for root in sys.argv[1:]:
    print("=== DATASET", root)
    info = json.load(open(root + "/meta/info.json", encoding="utf-8"))
    print("info keys", list(info.keys()))
    dfiles = sorted(glob.glob(root + "/data/chunk-*/file-*.parquet"))
    mfiles = sorted(glob.glob(root + "/meta/episodes/chunk-*/file-*.parquet"))
    print("data files", [p.split(root)[1] for p in dfiles])
    print("episode meta files", [p.split(root)[1] for p in mfiles])
    dt = pq.read_table(dfiles[0])
    print("DATA schema:")
    for f in dt.schema:
        print("  ", f.name, str(f.type))
    print("DATA schema metadata keys", list((dt.schema.metadata or {}).keys()))
    row0 = dt.slice(0, 1).to_pylist()[0]
    print("DATA row0 keys/values:", {k: (v if not isinstance(v, list) else (len(v), v[:3])) for k, v in row0.items()})
    mt = pq.read_table(mfiles[0])
    print("META schema:")
    for f in mt.schema:
        print("  ", f.name, str(f.type))
    print("META schema metadata keys", list((mt.schema.metadata or {}).keys()))
    mrow = mt.slice(0, 1).to_pylist()[0]
    print("META row0:", mrow)
    tk = pq.read_table(root + "/meta/tasks.parquet")
    print("TASKS schema:", [(f.name, str(f.type)) for f in tk.schema])
    print("TASKS rows:", tk.to_pylist())
    print("TASKS metadata keys", list((tk.schema.metadata or {}).keys()))
    st = json.load(open(root + "/meta/stats.json", encoding="utf-8"))
    print("STATS keys", list(st.keys()))
    for k in st:
        print("  STATS", k, {kk: (len(vv) if isinstance(vv, list) else vv) for kk, vv in st[k].items()})
    print("video dirs", sorted(p.split(root)[1] for p in glob.glob(root + "/videos/*/chunk-*/*.mp4")))

print("=== LEROBOT API")
from lerobot.datasets.lerobot_dataset import LeRobotDataset
print("LeRobotDataset.__init__", inspect.signature(LeRobotDataset.__init__))
print("LeRobotDataset.create", inspect.signature(LeRobotDataset.create))
print("LeRobotDataset.add_frame", inspect.signature(LeRobotDataset.add_frame))
print("LeRobotDataset.save_episode", inspect.signature(LeRobotDataset.save_episode))
import lerobot
print("lerobot path", lerobot.__file__)
try:
    from lerobot.datasets import video_utils as vu
    print("encode_video_frames", inspect.signature(vu.encode_video_frames))
except Exception as exc:
    print("video_utils import failed", type(exc).__name__, str(exc)[:200])
