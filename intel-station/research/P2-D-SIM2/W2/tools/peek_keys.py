import json, sys
import numpy as np
R = "/home/intel/intel-junction-2026/data/sim/2026-11-14/dataset-150-raw"
meta = json.load(open(R + "/attempt-0000/metadata.json"))
print("META_KEYS", sorted(meta.keys()))
print("META_0", json.dumps(meta)[:900])
tr = json.load(open(R + "/attempt-0000/trace.json"))
print("TRACE_LEN", len(tr), "TRACE_KEYS", sorted(tr[0].keys()))
print("TRACE_PHASES", [ (t["phase"]) for t in tr[::30]][:60])
z = np.load(R + "/attempt-0000/trajectory.npz")
print("NPZ", {k: z[k].shape for k in z.files}, z["state"].dtype)
s = json.load(open(R + "/summary.json"))
print("SUMMARY_TYPE", type(s).__name__, len(s))
if isinstance(s, list):
    print("SUMMARY_ITEM_KEYS", sorted(s[0].keys()))
    print("SUMMARY_ITEM0", json.dumps(s[0])[:700])
elif isinstance(s, dict):
    print("SUMMARY_KEYS", list(s.keys())[:40])
v = json.load(open(R + "/verification.json"))
print("VERIF_TYPE", type(v).__name__)
if isinstance(v, dict):
    print("VERIF_KEYS", list(v.keys())[:40])
    print("VERIF_HEAD", json.dumps(v)[:1200])
rc = json.load(open(R + "/recording_context.json"))
print("RC", rc)
