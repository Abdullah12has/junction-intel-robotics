#!/usr/bin/env python3
"""Collect the variant metrics into metrics.json (stage tape-eval-v2) with deltas against the base run."""
import sys, os, json, hashlib
E2 = sys.argv[1]
V = {}
for v in ["base", "p1", "p2", "p12"]:
    V[v] = json.load(open(os.path.join(E2, v, "metrics.json")))


def pick(m):
    t = m["tape_ours_close"]["overall"]
    s = m["tape_studio_close_per_brick"]["overall"]
    r = m["routing_ours"]
    h = m["shape_ours_2x2_vs_not"]
    return {"ours_close_decided": t["decided"], "ours_close_correct": t["correct"], "ours_close_uncertain": t["uncertain"],
            "ours_close_accuracy_ci95": t["accuracy_ci95"],
            "studio_close_decided": s["decided"], "studio_close_correct": s["correct"], "studio_close_uncertain": s["uncertain"],
            "studio_close_not_located": m["tape_studio_close_per_brick"]["not_located"],
            "routing_truth_known_replicated": r["truth_known_and_replicated"], "routing_correct": r["correct"],
            "routing_accuracy_ci95": r["accuracy_ci95"], "routing_not_replicable": r["not_replicable"],
            "shape_n": h["n"], "shape_correct": h["correct"]}


def cmp(a, b):
    out = {}
    for k in a:
        if isinstance(a[k], (int, float)) and isinstance(b.get(k), (int, float)):
            out[k] = b[k] - a[k]
    return out


base = pick(V["base"])
res = {"stage": "tape-eval-v2", "utc": "2026-11-14T21:00Z",
       "app_head_at_copy": "bbaf99a (working tree copied 20:48Z)",
       "scope": "close-view tape (ours 23 views, Studio close 115 frames per brick), routing + shape (ours, 6 replicable episodes), wrist used for shape only; top view not used by the decision code",
       "variants": {v: pick(V[v]) for v in V},
       "previous": {"stage": "tape-eval-v1", "ours_close_decided": 7, "ours_close_correct": 7},
       "delta_vs_base": {v: cmp(base, pick(V[v])) for v in V if v != "base"},
       "patches": {"p1": "patches/p1_sort_loop_jaw_anchored.patch (jaw-anchored fallback in sort_loop.close_verdict)",
                   "p2": "patches/p2_inspector_merged_tape.patch (merged blob tape not attributable in inspector.triage)",
                   "p12": "p1 + p2 (app_p12)"},
       "leave_one_episode_out": "not required: no threshold or parameter was fitted; p1 radius 130 px chosen once, not tuned on this set"}
json.dump(res, open(os.path.join(E2, "metrics.json"), "w"), indent=1, default=str)
print(json.dumps(res["variants"], default=str)[:1500])
print(json.dumps(res["delta_vs_base"], default=str))
# duplicate check for the Studio gallery rows
a = os.path.join("/home/intel/intel-junction-2026/work/P3-LABEL-STUDIO/E1/frames", "U20_064316_start_f0000.png")
b = os.path.join("/home/intel/intel-junction-2026/work/P3-LABEL-STUDIO/E1/frames", "U20_064316_x_f0000.png")
if os.path.exists(b):
    print("U20 start vs x md5 equal:", hashlib.md5(open(a, "rb").read()).hexdigest() == hashlib.md5(open(b, "rb").read()).hexdigest())
else:
    print("U20 x frame file not present")
