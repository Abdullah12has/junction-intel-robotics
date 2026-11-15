#!/usr/bin/env python3
"""Per-episode release pose check on the real Studio dataset (data only, read-only). Usage: python3 -I real_release_check.py OUT"""
import glob, json, sys
import numpy as np
import pyarrow.parquet as pq
REAL = "/home/intel/intel-junction-2026/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af"
OUT = sys.argv[1].rstrip("/")
S, E, F = [], [], []
for f in sorted(glob.glob(REAL + "/data/chunk-*/file-*.parquet")):
    tb = pq.read_table(f, columns=["observation.state", "episode_index", "frame_index"]).to_pydict()
    S += tb["observation.state"]
    E += tb["episode_index"]
    F += tb["frame_index"]
S, E, F = np.array(S, float), np.array(E), np.array(F)


def release(e):
    m = np.where(E == e)[0]
    m = m[np.argsort(F[m])]
    st = S[m]
    g = st[:, 5]
    tmin = int(np.argmin(g))
    r = g[tmin + 5:] - g[tmin:-5]
    k = int(np.argmax(r))
    return st[tmin + k, :5], int(F[m][tmin + k]), float(r[k])


gate_sets = {"blue": [7, 11, 12], "green": [8, 10, 14, 15]}
rows = []
for e in sorted(set(E.tolist())):
    p, frame, rise = release(e)
    side = "blue" if p[0] > 0 else "green"
    gate_label = [k for k, v in gate_sets.items() if e in v]
    rows.append({"episode": e, "release_frame": frame, "gripper_rise": round(rise, 2), "pan_norm": round(float(p[0]), 2),
                 "lift_norm": round(float(p[1]), 2), "elbow_norm": round(float(p[2]), 2), "side_by_pan_sign": side,
                 "gate_REAL_SETS_label": gate_label[0] if gate_label else "not in gate sets"})
blue_pan = [r["pan_norm"] for r in rows if r["side_by_pan_sign"] == "blue"]
green_pan = [r["pan_norm"] for r in rows if r["side_by_pan_sign"] == "green"]
mismatch = [r["episode"] for r in rows if r["gate_REAL_SETS_label"] != "not in gate sets" and r["gate_REAL_SETS_label"] != r["side_by_pan_sign"]]
out = {"episodes": rows, "pan_median_blue_side": float(np.median(blue_pan)) if blue_pan else None,
       "pan_median_green_side": float(np.median(green_pan)) if green_pan else None,
       "gate_sets_disagree_with_pan_side_for_episodes": mismatch, "gate_sets": gate_sets,
       "note": "side is read from the sign of the release pan (physical location), not from labels; D-067 excluded eps 0-2, 8-10, 13 from v2 sets"}
with open(OUT + "/real_release_check.json", "w") as fh:
    json.dump(out, fh, indent=1)
for r in rows:
    print(r["episode"], r["release_frame"], r["pan_norm"], r["lift_norm"], r["elbow_norm"], r["side_by_pan_sign"], r["gate_REAL_SETS_label"])
print("pan median blue side", out["pan_median_blue_side"], "green side", out["pan_median_green_side"], "mismatch", mismatch)
