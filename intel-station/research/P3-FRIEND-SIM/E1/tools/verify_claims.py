#!/usr/bin/env python3
"""Checks three report claims against the data (read-only). Usage: python3 -I verify_claims.py OUT"""
import glob, json, sys
import numpy as np
R = "/home/intel/intel-junction-2026"
OUT = sys.argv[1].rstrip("/")
C = json.load(open(OUT + "/compare.json"))
b = C["brightness"]
print("saturation S (mean HSV) top sim/real:", b["top"]["sim"]["S"], b["top"]["real"]["S"], " close sim/real:", b["close"]["sim"]["S"], b["close"]["real"]["S"])
K = np.array(C["joints"]["K_units_per_rad"])
first = []
for d in sorted(glob.glob(R + "/data/sim/live/2026-11-14/dataset-150-raw/attempt-*")):
    st = np.load(d + "/trajectory.npz")["state"].astype(float)
    first.append(st[0] * K)
F = np.array(first)
print("sim frame-0 normalized min:", F.min(0).round(2).tolist())
print("sim frame-0 normalized max:", F.max(0).round(2).tolist())
print("sim frame-0 normalized std:", F.std(0).round(3).tolist())
m = C["motion"]
ratio = np.array(m["sim"]["speed_p95"]) / np.maximum(np.array(m["real_joints_csv"]["speed_p95"]), 1e-9)
print("p95 speed ratio sim/real (pan..gripper):", ratio.round(2).tolist())
