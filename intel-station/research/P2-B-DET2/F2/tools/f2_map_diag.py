"""Lists labelled Opus keys that did not map, with the raw close bricks of the same frame (analysis only)."""
import json
import sys

RUN = sys.argv[1]
d = json.load(open(RUN))
frames = d["frames"]
for row in d["label_rows"]:
    if row["mapped"]:
        continue
    sample = "6d6bdce3_" + row["key"].split("/")[1].split("_")[0] + "_" + row["key"].split("/")[1].split("_")[1]
    print("UNMAPPED", row["key"], row["label"], row["colour"], "sample=" + sample)
    for b in frames.get(sample, {}).get("raw", {}).get("bricks", []):
        if b["view"] == "close":
            print("   raw close", b["id"], b["colour"], "aspect", b["aspect"], "tape", b["tape_score"],
                  "area", b["area"], "centre", b["centre_px"], "verdict", b["verdict"], b["reasons"])
