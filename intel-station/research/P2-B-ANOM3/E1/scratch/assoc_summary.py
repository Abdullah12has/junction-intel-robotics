"""Read-only summary of R/work/eyeinhand/assoc.json and the jaw mask (anom3 E1 scratch)."""
import collections
import json
import os

import cv2

R = "/home/intel/intel-junction-2026"
with open(os.path.join(R, "work/eyeinhand/assoc.json"), encoding="utf-8") as fh:
    A = json.load(fh)
print("type", type(A).__name__)
if isinstance(A, dict):
    print("keys", list(A.keys())[:20])
    for k, v in A.items():
        print("key", k, "type", type(v).__name__, "len", len(v) if hasattr(v, "__len__") else "-")
    entries = None
    for k, v in A.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            entries = v
            print("entries under key", k)
            break
else:
    entries = A
print("n entries", len(entries))
print("sample first", json.dumps(entries[0])[:700])
print("sample last", json.dumps(entries[-1])[:700])
ep_c = collections.Counter()
scan_c = collections.Counter()
exist = missing = 0
frames_root = collections.Counter()
for e in entries:
    ep = str(e.get("ep"))
    ep_c[ep] += 1
    scan_c[(ep, bool(e.get("scan")))] += 1
    p = str(e.get("frame", ""))
    full = p if os.path.isabs(p) else os.path.join(R, p)
    frames_root[os.path.dirname(p)] += 1
    if os.path.isfile(full):
        exist += 1
    else:
        missing += 1
print("per ep", dict(ep_c))
print("per ep scan", sorted(scan_c.items()))
print("frame dirs", dict(frames_root.most_common(5)))
print("frames exist", exist, "missing", missing)
ts = [float(e["t"]) for e in entries if "t" in e]
print("t range", min(ts) if ts else None, max(ts) if ts else None)
m = cv2.imread(os.path.join(R, "app/tools/wrist_jaw_mask.png"), cv2.IMREAD_GRAYSCALE)
print("wrist mask", None if m is None else (m.shape, int(m.min()), int(m.max()), round(float((m > 0).mean()), 4)))
c = cv2.imread(os.path.join(R, "app/tools/close_jaw_mask_hover.png"), cv2.IMREAD_GRAYSCALE)
print("close mask", None if c is None else (c.shape, int(c.min()), int(c.max())))
