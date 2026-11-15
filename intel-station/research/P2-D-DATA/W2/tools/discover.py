import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

root = Path(sys.argv[1])
info = json.loads((root / "meta" / "info.json").read_text())
print("INFO total_episodes", info["total_episodes"], "total_frames", info["total_frames"], "total_tasks", info["total_tasks"], "fps", info["fps"], "robot", info["robot_type"])
for f in sorted(root.rglob("*.parquet")):
    t = pq.read_table(f)
    print("FILE", f.relative_to(root), "rows", t.num_rows)
    print("  SCHEMA", [(fld.name, str(fld.type)) for fld in t.schema])
    df = t.to_pandas()
    keep = [c for c in df.columns if not c.startswith("stats/")]
    df = df[keep]
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        if "observation.state" in df.columns:
            print(df.head(3).to_string(max_colwidth=70))
        else:
            print(df.to_string(max_colwidth=70))
tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
print("TASKS index", list(tasks.index), "cols", list(tasks.columns))
print(tasks.reset_index().to_string())
