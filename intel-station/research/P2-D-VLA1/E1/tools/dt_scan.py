import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/intel/intel-junction-2026/app/vla")
import build_dataset as B  # the builder's own segment clock, candidate choice and nearest-frame pass (decode only)

RUN = sys.argv[1]
t0 = time.time()
skipped = []
episodes = B.load_episodes([RUN], False, None, skipped)
logs = os.path.join(B.ROOT, "logs/monitor")
wrist = os.path.join(B.ROOT, "data/wrist/raw")
media = {"top": B.list_media(logs, "top"), "close": B.list_media(logs, "close"), "wrist": B.list_media(wrist, "w")}
span = {"top": None, "close": None, "wrist": B.WRIST_SEG_S}
per_cam = {c: [] for c in media}
missing = []
for ep in episodes:
    t_first, t_last = float(ep["t"][0]), float(ep["t"][-1])
    for cam in media:
        cand = B.candidates(media[cam], t_first - B.PAD, t_last + B.PAD, span[cam])
        if not cand or cand[0][1] > t_first:
            missing.append((ep["name"], cam, "no segment at start"))
            continue
        dts = B.nearest_dt(cand, ep["t"])
        if dts is None:
            missing.append((ep["name"], cam, "does not cover rows"))
            continue
        per_cam[cam].append((ep["name"], float(dts.max()), int(np.sum(dts > 2.0 / B.FPS + 1e-9)), len(dts)))
summary = {}
for cam, rows in per_cam.items():
    summary[cam] = {
        "episodes": len(rows),
        "max_s": round(max(r[1] for r in rows), 6),
        "rows": sum(r[3] for r in rows),
        "rows_over_2_frames": sum(r[2] for r in rows),
        "episodes_over_2_frames": sum(1 for r in rows if r[1] > 2.0 / B.FPS + 1e-9),
        "worst_episode": max(rows, key=lambda r: r[1])[0],
    }
print("RUN", RUN, "episodes loaded", len(episodes), "skipped at load", len(skipped))
print("SUMMARY " + json.dumps(summary))
print("MISSING %d %s" % (len(missing), json.dumps(missing[:6])))
print("seconds %.1f" % (time.time() - t0))
