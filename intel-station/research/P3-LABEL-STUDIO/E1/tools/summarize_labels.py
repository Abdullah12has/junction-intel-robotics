import json
from collections import Counter

BASE = "/home/intel/intel-junction-2026/work/P3-LABEL-STUDIO/E1/"
rows = [json.loads(l) for l in open(BASE + "labels_episodes.jsonl", encoding="utf-8")]
print("rows", len(rows), "unique entries", len({r["entry"] for r in rows}))
print("trust", dict(Counter(r.get("trust") for r in rows)))
print("usable_for", dict(Counter(tuple(r.get("usable_for", [])) for r in rows)))
print("keys", sorted(rows[0].keys()))
print("entry\tP\tunit\trelation\ttask_text_ok\toutcome\tdecision_correct\ttrust\tsame_view\tmae_top\tmae_close")
for r in rows:
    fc = r.get("frame_check_vs_rep") or {}
    mae = fc.get("mae_by_cam") or {}
    print("\t".join(str(x) for x in [
        r["entry"], r.get("physical_episode"), r.get("unit_group"),
        (r.get("relation") or "")[:38], r.get("task_text_ok"), r.get("outcome"),
        r.get("decision_correct"), r.get("trust"), fc.get("same_view", "-"),
        mae.get("top", "-"), mae.get("close", "-")]))
