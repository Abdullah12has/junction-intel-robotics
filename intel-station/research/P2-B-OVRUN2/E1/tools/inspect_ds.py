import json, sys
import pyarrow.parquet as pq

ROOT = "/home/intel/.local/share/physicalai/datasets"
for d in sys.argv[1:]:
    base = f"{ROOT}/{d}"
    try:
        info = json.load(open(f"{base}/meta/info.json"))
        print("DS", d, "total_episodes", info.get("total_episodes"), "total_frames", info.get("total_frames"), "fps", info.get("fps"))
        print("features", list(info.get("features", {}).keys()))
    except Exception as exc:
        print("DS", d, "info.json error", type(exc).__name__, str(exc)[:200])
    eps = pq.read_table(f"{base}/meta/episodes/chunk-000/file-000.parquet").to_pandas()
    print("episode_columns", list(eps.columns))
    for _, r in eps.iterrows():
        tk = list(r["tasks"]) if "tasks" in r else None
        print("ep", int(r["episode_index"]), "len", int(r["length"]), "tasks", tk, "dfrom", int(r["dataset_from_index"]), "dto", int(r["dataset_to_index"]))
    tasks = pq.read_table(f"{base}/meta/tasks.parquet").to_pandas().reset_index()
    print("tasks", tasks.to_dict(orient="records"))
