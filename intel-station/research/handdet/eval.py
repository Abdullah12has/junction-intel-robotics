"""Old vs new hand detector (Opus I-61): (1) the 41 labelled e-stop images; (2) every 2nd second of the monitor
recordings since 17:03Z, with contact sheets of the frames where the two detectors disagree (for visual review)."""
import csv
import glob
import os
import subprocess
import sys

import cv2
import numpy as np

R = "/home/intel/intel-junction-2026"
OUT = f"{R}/work/handdet"
sys.path.insert(0, f"{R}/app/tools")
import handdet  # noqa: E402

OLD_MIN = {"top": 900, "close": 5000, "wrist": 20000}  # hand_watch.py thresholds at 18:12Z
LABEL_FIX = {("close", 2): "hand", ("close", 6): "nohand"}  # full-resolution re-check 18:40Z


def old_area(img, view):
    x0, y0, x1, y1 = handdet.ROI[view]
    m = handdet.skin_mask(img[y0:y1, x0:x1])
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    return int(st[1:, 4].max()) if n > 1 else 0


# (1) labelled e-stop images
rows = list(csv.DictReader(open(f"{OUT}/labels_features.tsv"), delimiter="\t"))
conf = {"old": {}, "new": {}}
lines = []
for r in rows:
    v, i = r["view"], int(r["idx"])
    lab = LABEL_FIX.get((v, i), r["label"])
    if lab == "unknown":
        continue
    img = cv2.imread(f"{R}/logs/monitor/{r['file']}")
    old = old_area(img, v) >= OLD_MIN[v]
    new, f = handdet.is_hand(img, v)
    for k, d in (("old", old), ("new", new)):
        key = (lab, "hand" if d else "nohand")
        conf[k][key] = conf[k].get(key, 0) + 1
    lines.append(f"{v}\t{i}\t{lab}\told={'hand' if old else 'no'}\tnew={'hand' if new else 'no'}\t{f}")
with open(f"{OUT}/eval_labelled.tsv", "w") as fh:
    fh.write("\n".join(lines) + "\n")
for k in ("old", "new"):
    c = conf[k]
    print(k, "hands caught", c.get(("hand", "hand"), 0), "/", c.get(("hand", "hand"), 0) + c.get(("hand", "nohand"), 0),
          "| false stops", c.get(("nohand", "hand"), 0), "/", c.get(("nohand", "hand"), 0) + c.get(("nohand", "nohand"), 0))

# (2) recordings: one frame every 2 s
dis = {"old_only": [], "new_only": []}
counts = {}
for view in ("top", "close"):
    for mkv in sorted(glob.glob(f"{R}/logs/monitor/{view}_*.mkv")):
        tmp = f"{OUT}/frames_{view}"
        os.makedirs(tmp, exist_ok=True)
        for f in glob.glob(f"{tmp}/*.jpg"):
            os.remove(f)
        subprocess.run(["nice", "-n", "19", "ffmpeg", "-v", "error", "-threads", "2", "-i", mkv, "-vf", "fps=0.5",
                        "-q:v", "4", f"{tmp}/f_%05d.jpg"], check=False)
        for f in sorted(glob.glob(f"{tmp}/*.jpg")):
            img = cv2.imread(f)
            if img is None:
                continue
            if img.shape[1] != 640:
                img = cv2.resize(img, (640, 480))
            old = old_area(img, view) >= OLD_MIN[view]
            new, ft = handdet.is_hand(img, view)
            c = counts.setdefault(view, {"n": 0, "old": 0, "new": 0})
            c["n"] += 1
            c["old"] += int(old)
            c["new"] += int(new)
            if old != new:
                tag = "old_only" if old else "new_only"
                t = cv2.resize(img, (320, 240))
                cv2.putText(t, f"{view} {os.path.basename(mkv)[-20:-4]} {os.path.basename(f)} {ft['area']}{ft['edge']}",
                            (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)
                dis[tag].append(t)
print("recordings", counts)
for tag, tiles in dis.items():
    print(tag, len(tiles))
    for k in range(0, min(len(tiles), 36), 12):
        chunk = tiles[k:k + 12]
        while len(chunk) % 4:
            chunk.append(np.zeros((240, 320, 3), np.uint8))
        sheet = np.vstack([np.hstack(chunk[j:j + 4]) for j in range(0, len(chunk), 4)])
        cv2.imwrite(f"{OUT}/dis_{tag}_{k // 12}.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
