#!/usr/bin/env python3
# V3 patch: home pose spread computed on eps 0-7 (eps 8-10 start away from home); fixes the INVENTORY line.
# usage: python3 -I v3_patch.py AUTO_DIR REPORT_DIR
import sys
import json
from pathlib import Path

import numpy as np

AUTO = Path(sys.argv[1])
REP = Path(sys.argv[2])
recs = json.loads((AUTO / "episodes_v3.json").read_text())
calib = json.loads((REP / "calib.json").read_text())
e07 = [r for r in recs if r["episode_index"] <= 7]
P = np.array([r["start_pose"] for r in e07], float)
home_like = [r for r in recs if r["start_dist_home"] < 6.0]
Q = np.array([r["start_pose"] for r in home_like], float)
A = np.array([r["start_pose"] for r in recs], float)
calib["home_pose_v3"] = dict(
    joints=calib["home_start_pose"]["joints"],
    definition="median of start poses of all 11 episodes; spread over eps 0-7 only (eps 8-10 start holding a brick away from home)",
    median_all11=np.round(np.median(A, axis=0), 2).tolist(),
    eps_0_7_median=np.round(np.median(P, axis=0), 2).tolist(),
    eps_0_7_min=np.round(P.min(0), 2).tolist(),
    eps_0_7_max=np.round(P.max(0), 2).tolist(),
    eps_0_7_std=np.round(P.std(0), 2).tolist(),
    start_within_6_units_of_median=[r["episode_index"] for r in home_like],
    start_within_6_units_median=np.round(np.median(Q, axis=0), 2).tolist(),
)
(REP / "calib.json").write_text(json.dumps(calib, indent=1, default=str))
md = (REP / "INVENTORY.md").read_text().splitlines()
new_md = []
for line in md:
    if line.startswith("- Home (start median, 11 eps)"):
        line = (f"- Home (start median, 11 eps): {calib['home_start_pose']['median']}; "
                f"eps 0-7 start min {calib['home_pose_v3']['eps_0_7_min']}, max {calib['home_pose_v3']['eps_0_7_max']}; "
                f"eps 8-10 start away from home (holding a brick, 116-135 units from home).")
    new_md.append(line)
(REP / "INVENTORY.md").write_text("\n".join(new_md) + "\n")
print("V3_PATCH_OK", json.dumps(calib["home_pose_v3"]))
