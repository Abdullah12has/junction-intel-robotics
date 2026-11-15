import json

BASE = "/home/intel/intel-junction-2026/work/P3-LABEL-STUDIO/E1/"
d = json.load(open(BASE + "datasets.json", encoding="utf-8"))
for x in d["datasets"]:
    if x["group"] == "curated":
        print("CURATED", x["name"], "fps", x["fps"], "info_eps", x["info_total_episodes"], "meta", x["episodes_meta_rows"],
              "data_eps", x["episodes_with_data"], "rows", x["data_rows"], "matched", x["matched_to_studio_entries"],
              "unmatched", x["unmatched"], "issues_eps", x["episodes_with_issues"], "phys", x["physical_episodes"],
              "tasks", len(x["tasks"]), "video_keys", len(x["video_keys"]))
        for e in x["episode_detail"]:
            if e["issues"]:
                print("   ", e["episode_index"], e["entry"], e["physical"], e["issues"])
    elif x["group"] == "studio":
        print("STUDIO", x["uuid"][:8], "eps", x["episodes_info_json"], "rows", x["data_rows"],
              "frames_total_s", round(sum(x["episode_frames"]) / 30.0, 1),
              "min_s", x["duration_s_min"], "max_s", x["duration_s_max"], "orig", len(x["originals_in_dataset"]),
              "trust", x["trust_counts"])
    elif x["group"] == "v1_hint":
        print("V1", json.dumps(x)[:300])
