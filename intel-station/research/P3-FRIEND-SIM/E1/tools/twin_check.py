#!/usr/bin/env python3
"""Read-only status of the twin data (another session's folder R/twin). Never writes under R/twin.
Writes twin_status.json and merges a 'twin' block into compare.json (in OUT). Usage: python3 -I twin_check.py OUT"""
import json, os, sys, time
R = "/home/intel/intel-junction-2026"
OUT = sys.argv[1].rstrip("/")
T = R + "/twin"
DIRS = ["calib", "model", "captures", "episodes/raw", "episodes/lerobot"]
st = {"checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "root": T + "/data", "dirs": {}}
for d in DIRS:
    p = os.path.join(T, "data", d)
    files = []
    if os.path.isdir(p):
        for dp, dn, fn in os.walk(p):
            for f in fn:
                q = os.path.join(dp, f)
                files.append([os.path.relpath(q, T + "/data"), os.path.getsize(q)])
    st["dirs"][d] = {"exists": os.path.isdir(p), "files": len(files), "bytes": sum(x[1] for x in files), "examples": sorted(files)[:40]}
st["code_py_files_under_app_twin"] = sum(1 for dp, dn, fn in os.walk(T + "/app/twin") for f in fn if f.endswith(".py"))
n = {d: st["dirs"][d]["files"] for d in DIRS}
st["status"] = "PRESENT" if any(n.values()) else "EMPTY at %s: no files in any twin data dir" % st["checked_utc"]
st["columns"] = {
    "camera_views": "available" if n["captures"] else "not available (captures empty)",
    "plate_positions": "available" if (n["calib"] or n["model"]) else "not available (calib and model empty)",
    "joint_units": "available" if (n["episodes/raw"] or n["episodes/lerobot"]) else "not available (episodes empty)",
    "episode_recordings": "available" if n["episodes/lerobot"] else "not available (episodes/lerobot empty)",
}
with open(OUT + "/twin_status.json", "w") as fh:
    json.dump(st, fh, indent=1)
cmp_path = OUT + "/compare.json"
C = json.load(open(cmp_path))
C["twin"] = st
with open(cmp_path, "w") as fh:
    json.dump(C, fh, indent=1, default=str)
print(json.dumps({"status": st["status"], "files": n, "columns": st["columns"], "py_files": st["code_py_files_under_app_twin"]}, indent=1))
