"""Self-check (R12): recount key facts directly from raw files, not from out/rows.json."""
import glob
import json
import math

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SIM = "/home/intel/intel-junction-2026/data/sim/2026-11-14/dataset-150-raw"
DS = "/home/intel/intel-junction-2026/data/studio-datasets"
RAW = DS + "/6d6bdce3-b382-478d-9697-fb9e573b97af"
CUR = DS + "/a4ba6ede-092c-450f-8c76-559f66991dbd"
PLATE = {"blue": (-0.105, -0.13), "green": (0.235, -0.16)}

metas = [json.load(open(f)) for f in sorted(glob.glob(SIM + "/attempt-*/metadata.json"))]
print("RC_N", len(metas), "ACC", sum(m["accepted"] for m in metas), "GRASP", sum(m["physical_grasp_verified"] for m in metas))
print("RC_BRICK", sum(m["brick_type"] == "2x2" for m in metas), sum(m["brick_type"] == "2x4" for m in metas))
print("RC_DEST", sum(m["destination"] == "blue" for m in metas), sum(m["destination"] == "green" for m in metas))
print("RC_OD10", sum((m["destination"] != ("blue" if m["brick_type"] == "2x2" else "green")) for m in metas))
errs = [math.dist(m["final_object_xyz"][:2], PLATE[m["destination"]]) for m in metas]
print("RC_MAX_XY_ERR", round(max(errs), 4), "ALL_BELOW_0.015", all(e < 0.015 for e in errs))
print("RC_FRAMES", sum(m["frames"] for m in metas), "TASK_TEXT_MATCH", all(m["task"] == f"move the {m['target_color']} brick to the {m['destination']} plate" for m in metas))
print("RC_SEEDS_UNIQUE", len({m["seed"] for m in metas}))
K_lift = 200 * 4095 / (2 * math.pi * (3090 - 822))
print("RC_K_LIFT", round(K_lift, 3), "K_PAN", round(200 * 4095 / (2 * math.pi * (3212 - 891)), 3))
# URDF limits converted to normalized units at the calibrated K (sanity of midpoint convention)
print("RC_URDF_LIFT_LIMIT_N", round(1.74533 * K_lift, 2), "URDF_ELBOW_LIMIT_N", round(1.69 * 200 * 4095 / (2 * math.pi * (3102 - 898)), 2))
# real raw: ep7 video timing and ep count
tab = pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(RAW + "/meta/episodes/chunk-000/file-*.parquet"))], ignore_index=True)
keys = [c for c in tab.columns if "243622061421" in c]
r7 = tab.loc[tab["episode_index"] == 7].iloc[0]
print("RC_EP7_LEN", int(r7["length"]), "TOP_FILE", int(r7[[c for c in keys if "file_index" in c][0]]), "TOP_FROM_TS", float(r7[[c for c in keys if "from_timestamp" in c][0]]), "TOP_TO_TS", float(r7[[c for c in keys if "to_timestamp" in c][0]]))
print("RC_EPISODES_IN_META", len(tab), "ep_idx_range", int(tab["episode_index"].min()), int(tab["episode_index"].max()))
# real data: lengths, curated==raw eps 3-6 check
data = pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(RAW + "/data/chunk-000/file-*.parquet"))], ignore_index=True)
cdata = pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(CUR + "/data/chunk-000/file-*.parquet"))], ignore_index=True)
lens = data.groupby("episode_index").size()
print("RC_REAL_FRAMES_TOTAL", int(len(data)), "RAW_EPS", int(lens.shape[0]), "DUR_MEDIAN_S", round(float(lens.median() / 30), 2))
for cur_ep, raw_ep in ((0, 3), (1, 4), (2, 5), (3, 6)):
    a = np.stack(cdata[cdata["episode_index"] == cur_ep].sort_values("frame_index")["observation.state"].values)
    b = np.stack(data[data["episode_index"] == raw_ep].sort_values("frame_index")["observation.state"].values)
    print("RC_CUR_EQ_RAW", cur_ep, raw_ep, a.shape == b.shape and bool(np.allclose(a, b)))
# real home pose from raw first frame of ep 3 and sim HOME
h = np.stack(data[data["episode_index"] == 3].sort_values("frame_index")["observation.state"].values)[0]
print("RC_REAL_HOME_EP3", np.round(h, 2).tolist(), "SIM_HOME_RAD", [0, -0.5, 0.8, 0.6, 0, 0.5])
# grasp radius from base for sim (python re-derivation of K and sign) is in fk_check.py; here just report
print("RC_DONE")
