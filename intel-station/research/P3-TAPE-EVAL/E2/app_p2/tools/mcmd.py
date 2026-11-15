#!/usr/bin/env python3
"""Send one command to motion_server.py and wait for its state. usage: mcmd.py '{"op":"move","q":[...],"speed":5}'"""
import json
import os
import sys
import time

D = "/home/intel/intel-junction-2026/run/mserver"
c = json.loads(sys.argv[1])
timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 60
try:
    last = json.load(open(f"{D}/state.json")).get("last_id", 0)
except Exception:
    last = 0
c["id"] = int(last) + 1 + int(time.time()) % 1000 * 1000
with open(f"{D}/cmd.jsonl", "a") as f:
    f.write(json.dumps(c) + "\n")
t0 = time.time()
while time.time() - t0 < timeout:
    try:
        st = json.load(open(f"{D}/state.json"))
        if st.get("last_id") == c["id"]:
            print(json.dumps(st))
            sys.exit(0 if st.get("status") in ("ok", "ready") or c["op"] in ("state", "exit", "release") else 2)
    except Exception:
        pass
    if not os.path.exists(f"{D}/pid") and c["op"] not in ("exit", "release"):
        print("server not running")
        sys.exit(3)
    time.sleep(0.1)
print("timeout")
sys.exit(4)
