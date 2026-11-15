import sys, os, glob
import pandas as pd
import pyarrow.parquet as pq

root = sys.argv[1]
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_colwidth", 70)

t = pd.read_parquet(os.path.join(root, "meta", "tasks.parquet"))
print("TASKS columns:", list(t.columns), "index name:", t.index.name, "index type:", type(t.index[0]).__name__)
print(t.reset_index().to_string())

for pat in ["meta/episodes/*/*.parquet", "data/*/*.parquet"]:
    files = sorted(glob.glob(os.path.join(root, pat)))
    print("FILES", pat, len(files))
    for f in files:
        md = pq.ParquetFile(f).metadata
        print("  ", os.path.relpath(f, root), "rows", md.num_rows, "rowgroups", md.num_row_groups)
    if files:
        print(pq.ParquetFile(files[0]).schema_arrow)

ep = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(root, "meta/episodes/*/*.parquet")))], ignore_index=True)
print("EPISODES shape", ep.shape)
print("EPISODE COLUMNS:", list(ep.columns))
keep = [c for c in ep.columns if not c.startswith("stats/")]
print(ep[keep].to_string())

d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(root, "data/*/*.parquet")))], ignore_index=True)
print("DATA shape", d.shape)
print("DATA dtypes:\n", d.dtypes.to_string())
print(d.head(3).to_string(max_colwidth=120))
print("rows per episode:\n", d.groupby("episode_index").size().to_string())
print("task_index values:", sorted(d["task_index"].unique().tolist()))
