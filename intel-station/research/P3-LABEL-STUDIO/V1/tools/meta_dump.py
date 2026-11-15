import json, os, sys, glob
import pyarrow.parquet as pq

ROOTS = ["/home/intel/intel-junction-2026/data/studio-datasets",
         "/home/intel/intel-junction-2026/data/curated"]
OUT = "/home/intel/intel-junction-2026/work/P3-LABEL-STUDIO/V1/out"


def plain(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, dict):
        return {k: plain(x) for k, x in v.items()}
    if hasattr(v, "isoformat"):
        return str(v)
    return v


def table_rows(path):
    t = pq.read_table(path)
    cols = t.column_names
    rows = t.to_pylist()
    return cols, [plain(r) for r in rows]


def main():
    result = {}
    for root in ROOTS:
        for ds in sorted(os.listdir(root)):
            d = os.path.join(root, ds)
            if not os.path.isdir(d):
                continue
            entry = {"path": d}
            info = os.path.join(d, "meta", "info.json")
            if os.path.exists(info):
                with open(info) as f:
                    entry["info"] = json.load(f)
            tasks = os.path.join(d, "meta", "tasks.parquet")
            if os.path.exists(tasks):
                cols, rows = table_rows(tasks)
                entry["tasks_cols"] = cols
                entry["tasks"] = rows
            eps = sorted(glob.glob(os.path.join(d, "meta", "episodes", "*", "*.parquet")))
            entry["episode_files"] = eps
            ep_rows = []
            ep_cols = None
            for p in eps:
                cols, rows = table_rows(p)
                ep_cols = cols
                ep_rows.extend(rows)
            entry["episode_cols"] = ep_cols
            entry["episodes"] = ep_rows
            data = sorted(glob.glob(os.path.join(d, "data", "*", "*.parquet")))
            entry["data_files"] = data
            entry["data_rows"] = sum(pq.ParquetFile(p).metadata.num_rows for p in data)
            if data:
                schema = pq.ParquetFile(data[0]).schema_arrow
                entry["data_schema"] = [f"{f.name}:{f.type}" for f in schema]
            vids = sorted(glob.glob(os.path.join(d, "videos", "*", "*", "*.mp4")))
            entry["videos"] = [{"path": v, "bytes": os.path.getsize(v)} for v in vids]
            result[ds] = entry
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "meta_dump.json"), "w") as f:
        json.dump(result, f, indent=1, default=str)
    print("datasets", len(result))
    for k, v in result.items():
        print(k, "episodes", len(v.get("episodes", [])), "data_rows", v.get("data_rows"), "videos", len(v.get("videos", [])))


if __name__ == "__main__":
    main()
