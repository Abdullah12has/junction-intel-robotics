"""Lists top-view CONFIDENT_DEFECT calls of a run with their geometry, and the live 155631 top brick of other runs."""
import json
import sys

FINAL = json.load(open(sys.argv[1]))
print("FINAL top CD calls (raw and jpg), sample, side, id, colour, area, aspect, fill, shape_conf, split, truth")
for key, res in sorted(FINAL["frames"].items()):
    for side in ("raw", "jpg"):
        r = res.get(side, {})
        for b in r.get("bricks", []):
            if b["view"] == "top" and b["verdict"] == "CONFIDENT_DEFECT":
                print(side, key, b["id"], b["colour"], b["area"], b["aspect"], b.get("fill"), b["shape_conf"],
                      b.get("split"), b.get("truth"), b["reasons"])
for path in sys.argv[2:]:
    run = json.load(open(path))
    for b in run["live_full"]["155631"].get("bricks", []):
        if b["view"] == "top" and b["colour"] == "white":
            print("LIVE155631", path.rsplit("/", 1)[-1], b["id"], b["shape"], b["shape_conf"], b["aspect"],
                  b.get("fill"), b["area"], b["verdict"])
