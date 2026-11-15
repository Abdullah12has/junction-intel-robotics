#!/usr/bin/env python3
"""P2-D-SIM E1 snapshot gate (stdlib only). Read-only on the team folder SRC.
usage: snapshot.py status | ready
status: one JSON line with manifest coverage and incoming rsync processes (no hashing).
ready : returns WAIT if an incoming rsync exists; otherwise waits 120 s with no incoming
        rsync, verifies every SHA256.json entry, re-checks rsync, then copies SRC into DST
        (never writes to SRC) and verifies the copy.
"""
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

SRC = "/home/intel/junction-cell/simulation-data/2026-11-14"
DST = "/home/intel/intel-junction-2026/data/sim/2026-11-14"
LOG = "/home/intel/intel-junction-2026/logs/P2-D-SIM-E1/snapshot.log"


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%SZ")


def incoming():
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return [l.strip()[:160] for l in out.splitlines()
            if l.strip().startswith("rsync --server") and "simulation-data" in l]


def manifest():
    with open(os.path.join(SRC, "SHA256.json")) as f:
        return json.load(f)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def verify(root):
    man = manifest()
    res = {"entries": len(man), "ok": 0, "mismatch": [], "missing": []}
    for rel, want in man.items():
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            res["missing"].append(rel)
        elif sha256(p) == want:
            res["ok"] += 1
        else:
            res["mismatch"].append(rel)
    return res


def log(obj):
    obj["utc"] = utc()
    line = json.dumps(obj, sort_keys=True)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def ready():
    if incoming():
        return log({"event": "WAIT", "reason": "incoming rsync running", "rsync": incoming()})
    start = time.time()
    while time.time() - start < 120:
        if incoming():
            return log({"event": "WAIT", "reason": "rsync appeared in quiet window"})
        time.sleep(10)
    res = verify(SRC)
    if res["mismatch"] or res["missing"]:
        return log({"event": "WAIT", "reason": "manifest incomplete",
                    "ok": res["ok"], "missing": len(res["missing"]),
                    "mismatch": len(res["mismatch"]),
                    "first_missing": res["missing"][:3], "first_mismatch": res["mismatch"][:3]})
    if incoming():
        return log({"event": "WAIT", "reason": "rsync appeared during verification"})
    shutil.copytree(SRC, DST, dirs_exist_ok=True)
    post = verify(DST)
    return log({"event": "COPIED", "dst": DST, "entries": post["entries"], "ok": post["ok"],
                "mismatch": len(post["mismatch"]), "missing": len(post["missing"])})


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "status"
    if mode == "status":
        man = manifest()
        present = sum(1 for k in man if os.path.isfile(os.path.join(SRC, k)))
        log({"event": "STATUS", "entries": len(man), "present": present, "incoming_rsync": incoming()})
    elif mode == "ready":
        ready()
    else:
        sys.exit("usage: snapshot.py status | ready")
