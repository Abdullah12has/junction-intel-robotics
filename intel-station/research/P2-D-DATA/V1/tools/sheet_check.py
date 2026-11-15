import json
import os
import sys

sys.path.insert(0, "/home/intel/intel-junction-2026/work/P2-D-DATA/V1/app")
import cv2
import numpy as np
from data import inventory as inv

OUT = "/home/intel/intel-junction-2026/data/studio-datasets-report/V1/sheets/top_start_release_end.png"
EPS = [int(x) for x in sys.argv[1].split(",")]
dump = json.load(open("/home/intel/intel-junction-2026/data/studio-datasets-report/V1/inventory.json"))
rel = {}
for ep in dump["episodes"]:
    pr = ep["primary"]
    if pr and pr["release"] is not None:
        rel[ep["episode_index"]] = pr["release"]
info, tasks, eps_meta, by = inv.read_dataset(inv.ROOT_DEFAULT)
K = "observation.images." + inv.CAM_TOP
plan = {}
for em in eps_meta:
    e = int(em["episode_index"])
    if e not in EPS:
        continue
    ch = int(em["videos/%s/chunk_index" % K])
    fx = int(em["videos/%s/file_index" % K])
    s0 = int(round(float(em["videos/%s/from_timestamp" % K]) * 30))
    n = int(em["length"])
    path = os.path.join(inv.ROOT_DEFAULT, "videos", K, "chunk-%03d" % ch, "file-%03d.mp4" % fx)
    P = plan.setdefault(path, {"exp": 0, "wanted": {}})
    P["exp"] = max(P["exp"], s0 + n)
    tags = {"start": 0, "end": n - 1}
    if e in rel:
        tags["rel5"] = rel[e] + 5
    for tag, t in tags.items():
        P["wanted"].setdefault(s0 + t, []).append((e, tag))
panels, report = {}, []
for path, P in sorted(plan.items()):
    got, cnt, rc, err = inv.decode_file(path, set(P["wanted"].keys()))
    report.append("%s rc=%s frames=%d expected=%d" % (os.path.basename(path), rc, cnt, P["exp"]))
    for loc, items in P["wanted"].items():
        if loc in got:
            for e, tag in items:
                panels[(e, tag)] = got[loc]
cw, ch = 240, 180
canvas = np.full((len(EPS) * (ch + 16), 3 * cw, 3), 25, np.uint8)
for r, e in enumerate(EPS):
    for c, tag in enumerate(["start", "rel5", "end"]):
        y, x = r * (ch + 16), c * cw
        f = panels.get((e, tag))
        if f is not None:
            canvas[y + 16:y + 16 + ch, x:x + cw] = cv2.resize(f, (cw, ch), interpolation=cv2.INTER_AREA)
        cv2.putText(canvas, "ep%d %s" % (e, tag), (x + 3, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
cv2.imwrite(OUT, canvas[:, :, ::-1])
print("\n".join(report))
print("WROTE", OUT)
