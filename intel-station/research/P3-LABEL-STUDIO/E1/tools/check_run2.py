import json
from collections import Counter

BASE = "/home/intel/intel-junction-2026/work/P3-LABEL-STUDIO/E1/"
m = json.load(open(BASE + "metrics.json", encoding="utf-8"))
print("stage", m["stage"], "utc", m["utc"])
print("delta", json.dumps(m["delta"], sort_keys=True))
rows = [json.loads(l) for l in open(BASE + "labels_episodes.jsonl", encoding="utf-8")]
print("rows", len(rows), "trust", dict(Counter(r["trust"] for r in rows)))
print("verdicts", dict(Counter((r.get("frame_check_vs_rep") or {}).get("frame_verdict", "rep") for r in rows)))
for r in rows:
    if r["entry"] in ("18468a09:6", "e2eae6c8:2", "a4ba6ede:1", "a4ba6ede:0", "7e99a9e4:0"):
        fc = r.get("frame_check_vs_rep") or {}
        po = fc.get("pixel_offset_check") or {}
        print(r["entry"], r["outcome"], "|", r["failure_category"][:70], "|", r["task_text_ok"],
              "| dec", r["decision_plate"], r["decision_correct"], "| trust", r["trust"],
              "| verdict", fc.get("frame_verdict"), "| best", po.get("best_offset_frames"))
print("notes a4ba6ede:0 ->", [r["notes"] for r in rows if r["entry"] == "a4ba6ede:0"][0][-260:])
print("notes 6d6bdce3:7 ->", [r["notes"] for r in rows if r["entry"] == "6d6bdce3:7"][0])
print("mismatch kinds", dict(Counter(l.split("\t")[0] for l in open(BASE + "mismatches.tsv", encoding="utf-8").read().splitlines()[1:])))
