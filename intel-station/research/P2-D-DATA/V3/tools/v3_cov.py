#!/usr/bin/env python3
# V3: add a coverage checklist (R11) to the JSON deliverables in the report folder.
# usage: python3 -I v3_cov.py REPORT_DIR
import sys
import json
from pathlib import Path

REP = Path(sys.argv[1])
COV_GAPS = [
    "[x] needs computed for S1 (OD-9), S2 (BLUE), S3 (GREEN) from the 11 episodes (verdicts in episodes.csv)",
    "[x] recording requests written from the QA (spot instability, task text split, ep 0-2 not S1, ep 7 no return)",
    "[ ] recording requests not yet checked against Hassan's plans (Opus relays)",
]
COV_CALIB = [
    "[x] gripper open, closed on brick, closed on nothing, open after release, closing direction",
    "[x] closure rule and stationary thresholds (speed 1.0 units/s, idle 3.0 units/s, dwell 0.5 s)",
    "[x] home start pose (all 11 and eps 0-7) and hold pose (start of eps 8-10)",
    "[x] blue_drop (ep 7), green_drop (eps 8-10), spot put-down poses (eps 3-6) with spread",
    "[x] approach poses 0.5 s and 1.0 s before each release; release distance from home",
    "[x] duration stats by type and suggested max_s = 1.5 x p95",
    "[ ] placed-brick pixel centres per RealSense view: not measured numerically",
    "[ ] hover pose 8 cm above spot: not derivable without forward kinematics",
    "[ ] per-episode stationary dwell check: only frac_stationary reported",
]
gaps = json.loads((REP / "gaps.json").read_text())
gaps["coverage_checklist"] = COV_GAPS
(REP / "gaps.json").write_text(json.dumps(gaps, indent=1))
calib = json.loads((REP / "calib.json").read_text())
calib["coverage_checklist"] = COV_CALIB
(REP / "calib.json").write_text(json.dumps(calib, indent=1))
print("V3_COV_OK")
