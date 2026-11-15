import json
from collections import Counter

BASE = "/home/intel/intel-junction-2026/work/P3-LABEL-STUDIO/E1/"
rows = [json.loads(l) for l in open(BASE + "labels_episodes.jsonl", encoding="utf-8")]
census = {}
for line in open(BASE + "out/episodes_census.jsonl", encoding="utf-8"):
    e = json.loads(line)
    census["%s:%d" % (e["dataset"][:8], e["episode_index"])] = e
print("TASK TEXT PER ENTRY")
for r in rows:
    print(r["entry"], "|", r["task_text"], "|", r["task_text_ok"])
print("ALIGNMENT / JOINT FLAGS")
bad_align = [r["entry"] for r in rows if not all(r["timestamp_frame_alignment"].values())]
bad_range = [r["entry"] for r in rows if not r["joint_sanity"]["action_range_in_bounds"]]
sat = [(r["entry"], r["joint_sanity"]["saturated_at_normalized_limit"]) for r in rows
       if r["joint_sanity"]["saturated_at_normalized_limit"]]
flags = [(r["entry"], r["joint_sanity"]["census_flags"]) for r in rows if r["joint_sanity"]["census_flags"]]
lags = Counter(r["joint_sanity"]["lag_best_frames"] for r in rows)
print("alignment_not_all_true", bad_align)
print("action_out_of_bounds", bad_range)
print("saturated", sat)
print("census_flags", flags)
print("lag_best_counts", dict(lags))
print("PHYSICAL REPRESENTATIVES")
for r in rows:
    if r["relation"].startswith("representative"):
        c = census[r["entry"]]
        print(r["physical_episode"], r["entry"], r["target_brick"]["colour"], r["target_brick"]["shape"],
              r["target_brick"]["tape"], "decision", r["decision_plate"], "od10", r["od10_correct_plate"],
              "dc", r["decision_correct"], "outcome", r["outcome"], "rel", r["release_s_in_rep"],
              "grip_events", c.get("grip_events_s"), "dur", r["dur_s"], "trust", r["trust"])
