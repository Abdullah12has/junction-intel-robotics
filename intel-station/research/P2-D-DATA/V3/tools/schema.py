import sys
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

DS = Path(sys.argv[1])
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 120)

print("== meta/episodes files")
for p in sorted(DS.glob("meta/episodes/chunk-*/file-*.parquet")):
    t = pq.read_table(p)
    print(p.name, t.num_rows, t.schema.names)
print("== tasks")
t = pd.read_parquet(DS / "meta/tasks.parquet")
print(t.dtypes.to_dict(), "index:", t.index.name)
print(t.reset_index().to_string())
print("== data files")
for p in sorted(DS.glob("data/chunk-*/file-*.parquet")):
    t = pq.read_table(p)
    print(p.name, t.num_rows, t.schema.names)
print(pq.read_schema(sorted(DS.glob("data/chunk-*/file-000.parquet"))[0]))
print("== episodes table")
ep = pd.concat([pd.read_parquet(p) for p in sorted(DS.glob("meta/episodes/chunk-*/file-*.parquet"))], ignore_index=True)
print(ep.shape)
for c in ep.columns:
    print(c, "=>", repr(ep[c].iloc[0])[:300])
print(ep[[c for c in ep.columns if c in ("episode_index", "length", "tasks")]].to_string())
print("== data table")
d = pd.concat([pd.read_parquet(p) for p in sorted(DS.glob("data/chunk-*/file-*.parquet"))], ignore_index=True)
print(d.shape, list(d.columns))
print(d.iloc[0].to_dict())
print(d.groupby("episode_index").size().to_string())
print("state0", d["observation.state"].iloc[0])
print("action0", d["action"].iloc[0])
print("index monotone:", bool((d["index"].diff().dropna() == 1).all()))
print("V3_SCHEMA_OK")
