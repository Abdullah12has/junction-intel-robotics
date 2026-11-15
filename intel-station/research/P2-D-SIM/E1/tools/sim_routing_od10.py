#!/usr/bin/env python3
"""Routing check of the simulation dataset against OD-10 (read-only).
OD-10: GOOD (clean 2x2) -> BLUE plate; DEFECTIVE (tape or not 2x2) -> GREEN plate.
Expected plate is derived from the recorded brick_type (geometry), never from the task text.
The task text and the metadata destination are only compared with the physical destination.
usage: /usr/bin/python3 -I -B sim_routing_od10.py RAW_DIR OUT_DIR
"""
import ast
import collections
import csv
import json
import math
import os
import re
import sys

RAW, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

# plate centres as defined in the simulation source (station_scene.py), read at run time
src = open(os.path.join(RAW, "source", "station_scene.py"), encoding="utf-8").read()
m = re.search(r"BOWL_CENTERS = (\{.*?\})", src)
PLATES = ast.literal_eval(m.group(1))

rows = []
for name in sorted(os.listdir(RAW)):
    if not re.fullmatch(r"attempt-\d{4}", name):
        continue
    with open(os.path.join(RAW, name, "metadata.json"), encoding="utf-8") as f:
        md = json.load(f)
    fin = md["final_object_xyz"]
    dists = {p: math.hypot(fin[0] - c[0], fin[1] - c[1]) for p, c in PLATES.items()}
    phys = min(dists, key=dists.get)
    txt = re.search(r"to the (\w+) plate", md["task"])
    txt_dest = txt.group(1) if txt else None
    expected = "blue" if md["brick_type"] == "2x2" else "green"  # OD-10 by geometry only
    rows.append({
        "attempt": name, "seed": md["seed"], "target_color": md["target_color"],
        "brick_type": md["brick_type"], "destination_meta": md["destination"],
        "destination_text": txt_dest, "destination_physical": phys,
        "final_dist_to_plate_m": round(dists[phys], 4), "expected_od10": expected,
        "conforms_od10": phys == expected, "accepted": md["accepted"], "success": md["success"],
        "physical_grasp_verified": md["physical_grasp_verified"], "frames": md["frames"], "fps": md["fps"],
        "meta_eq_text": md["destination"] == txt_dest, "meta_eq_physical": md["destination"] == phys,
    })

with open(os.path.join(OUT, "sim_episode_routing.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

cross = collections.Counter((r["brick_type"], r["destination_physical"]) for r in rows)
summary = {
    "episodes": len(rows),
    "plates_from_station_scene": PLATES,
    "brick_type_counts": dict(collections.Counter(r["brick_type"] for r in rows)),
    "color_counts": dict(collections.Counter(r["target_color"] for r in rows)),
    "destination_meta_counts": dict(collections.Counter(r["destination_meta"] for r in rows)),
    "destination_physical_counts": dict(collections.Counter(r["destination_physical"] for r in rows)),
    "crosstab_brick_type_x_physical_destination": {f"{a}->{b}": n for (a, b), n in sorted(cross.items())},
    "conforms_od10": sum(r["conforms_od10"] for r in rows),
    "contradicts_od10": sum(not r["conforms_od10"] for r in rows),
    "contradicts_od10_2x2_to_green": sum(1 for r in rows if r["brick_type"] == "2x2" and not r["conforms_od10"]),
    "contradicts_od10_2x4_to_blue": sum(1 for r in rows if r["brick_type"] != "2x2" and not r["conforms_od10"]),
    "meta_destination_eq_task_text": sum(r["meta_eq_text"] for r in rows),
    "meta_destination_eq_physical": sum(r["meta_eq_physical"] for r in rows),
    "physical_dist_to_plate_m_max": max(r["final_dist_to_plate_m"] for r in rows),
    "accepted_all": all(r["accepted"] for r in rows),
    "success_all": all(r["success"] for r in rows),
    "grasp_verified_all": all(r["physical_grasp_verified"] for r in rows),
    "frames_min_max_mean": [min(r["frames"] for r in rows), max(r["frames"] for r in rows),
                            round(sum(r["frames"] for r in rows) / len(rows), 2)],
    "fps_set": sorted(set(r["fps"] for r in rows)),
    "contradicting_episodes": [r["attempt"] for r in rows if not r["conforms_od10"]][:200],
}
with open(os.path.join(OUT, "od10_routing_summary.json"), "w") as f:
    json.dump(summary, f, indent=1, sort_keys=True)
print(json.dumps({k: v for k, v in summary.items() if k != "contradicting_episodes"}, sort_keys=True))
