"""Shape features of the skin blobs in every labelled estop image (Opus I-61). Labels from Opus's visual review."""
import csv
import cv2
import numpy as np

R = "/home/intel/intel-junction-2026"
OUT = f"{R}/work/handdet"
ROI = {"top": (175, 30, 640, 345), "close": (0, 0, 640, 480), "wrist": (0, 0, 640, 480)}
# Opus visual labels 18:36Z (sheets sheet_*.jpg): hand / nohand / unknown
LAB = {
    "top": ["nohand", "nohand", "hand", "hand", "hand", "hand", "hand", "hand", "hand", "hand", "hand", "unknown",
            "hand", "unknown", "hand", "hand", "hand"],
    "close": ["nohand", "hand", "nohand", "nohand", "nohand", "nohand", "nohand", "nohand", "nohand"],
    "wrist": ["nohand"] * 15,
}
WHY = {
    ("close", 2): "orange brick", ("close", 3): "gripper mount + held white brick", ("close", 4): "warm speckles on mat",
    ("close", 5): "mat speckles + cream brick", ("close", 6): "cream taped brick", ("close", 7): "cream brick",
    ("close", 8): "cream brick", ("wrist", 0): "white brick", ("top", 0): "no hand visible",
    ("top", 1): "no hand visible",
}


def skin(img):
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((ycc[..., 1] >= 138) & (ycc[..., 1] <= 170) & (ycc[..., 2] >= 90) & (ycc[..., 2] <= 125)
         & (h >= 5) & (h <= 20) & (s >= 45) & (s <= 150) & (v >= 90)).astype(np.uint8) * 255
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def feats(img, view):
    x0, y0, x1, y1 = ROI[view]
    m = skin(img)[y0:y1, x0:x1]
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return dict(area=0, fill=0, solidity=0, border="", hue=0, sat=0, val=0)
    i = 1 + int(np.argmax(st[1:, 4]))
    a = int(st[i, 4])
    x, y, w, h = (int(v) for v in st[i, :4])
    comp = (lab == i).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    ha = float(cv2.contourArea(hull)) or 1.0
    H, W = m.shape
    border = "".join(s for s, c in (("T", y <= 1), ("B", y + h >= H - 1), ("L", x <= 1), ("R", x + w >= W - 1)) if c)
    hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    px = hsv[comp > 0]
    return dict(area=a, fill=round(a / float(w * h), 3), solidity=round(a / ha, 3), border=border,
                hue=int(np.median(px[:, 0])), sat=int(np.median(px[:, 1])), val=int(np.median(px[:, 2])))


rows = []
with open(f"{OUT}/estops.tsv") as fh:
    for r in csv.DictReader(fh, delimiter="\t"):
        v, i = r["view"], int(r["idx"])
        img = cv2.imread(f"{R}/logs/monitor/{r['file']}")
        f = feats(img, v)
        rows.append(dict(view=v, idx=i, file=r["file"], utc=r["utc"], label=LAB[v][i], why=WHY.get((v, i), ""), **f))
keys = list(rows[0].keys())
with open(f"{OUT}/labels_features.tsv", "w") as fh:
    fh.write("\t".join(keys) + "\n")
    for r in rows:
        fh.write("\t".join(str(r[k]) for k in keys) + "\n")
for r in rows:
    print(r["view"], r["idx"], r["label"], r["area"], r["fill"], r["solidity"], r["border"], r["hue"], r["sat"], r["val"], r["why"])
