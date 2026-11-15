import glob
import json
import os

RUNS = "/home/intel/intel-junction-2026/logs/sort"
print("runs:", [(os.path.basename(r), len(glob.glob(os.path.join(r, "ep_*")))) for r in sorted(glob.glob(os.path.join(RUNS, "*")))])
for run in ("any_1731", "any_1740"):
    for ep in sorted(glob.glob(os.path.join(RUNS, run, "ep_*"))):
        mp = os.path.join(ep, "episode.json")
        jp = os.path.join(ep, "joints.csv")
        meta = {}
        if os.path.isfile(mp):
            with open(mp, encoding="utf-8") as fh:
                meta = json.load(fh)
        rows = -1
        if os.path.isfile(jp):
            with open(jp, encoding="utf-8") as fh:
                rows = sum(1 for _ in fh) - 1
        task = str(meta.get("task", ""))[:50]
        print(run, os.path.basename(ep), "rows", rows, "success", meta.get("success"), "plate", meta.get("plate"),
              "task", json.dumps(task), "t_start", meta.get("t_start"), "t_end", meta.get("t_end"))
