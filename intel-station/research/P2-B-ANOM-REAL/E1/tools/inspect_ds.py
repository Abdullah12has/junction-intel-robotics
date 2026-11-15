import glob, os, sys
import pyarrow.parquet as pq
import pandas as pd
ROOT = "/home/intel/intel-junction-2026/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af"
ep_files = sorted(glob.glob(f"{ROOT}/meta/episodes/chunk-*/file-*.parquet"))
print("episode files", ep_files)
ep = pd.concat([pq.read_table(f).to_pandas() for f in ep_files], ignore_index=True)
print("episodes rows", len(ep))
print("columns", list(ep.columns))
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
show = [c for c in ep.columns if not c.startswith("stats/")]
print(ep[show].to_string())
data_files = sorted(glob.glob(f"{ROOT}/data/chunk-*/file-*.parquet"))
d = pd.concat([pq.read_table(f).to_pandas() for f in data_files], ignore_index=True)
print("data rows", len(d), "columns", list(d.columns))
print(d.head(3).to_string())
print("tasks:")
print(pq.read_table(f"{ROOT}/meta/tasks.parquet").to_pandas().to_string())
