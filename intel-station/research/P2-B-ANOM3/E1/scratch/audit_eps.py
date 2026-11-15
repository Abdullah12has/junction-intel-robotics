import glob
import json
import os

R = "/home/intel/intel-junction-2026"
rows = []
for ep_json in sorted(glob.glob(os.path.join(R, "logs", "sort", "*", "ep_*", "episode.json"))):
    d = os.path.dirname(ep_json)
    run = os.path.basename(os.path.dirname(d))
    name = run + "/" + os.path.basename(d)
    try:
        with open(ep_json, encoding="utf-8") as fh:
            ep = json.load(fh)
    except (OSError, ValueError) as exc:
        rows.append((name, "UNREADABLE " + str(exc)[:60]))
        continue
    cdir = os.path.join(d, "crops")
    pngs = sorted(c for c in os.listdir(cdir) if c.endswith(".png")) if os.path.isdir(cdir) else []
    views = [str(v.get("verdict")) if isinstance(v, dict) else str(v) for v in (ep.get("views") or [])]
    station = ep.get("station") or {}
    pre = ep.get("pre_close") or {}
    usable = ep.get("success") is True and ep.get("label") in ("GOOD", "DEFECT")
    mismatch = ""
    if ep.get("label") == "GOOD" and ep.get("plate") not in (None, "blue"):
        mismatch = " PLATE_MISMATCH"
    if ep.get("label") == "DEFECT" and ep.get("plate") not in (None, "green"):
        mismatch = " PLATE_MISMATCH"
    rows.append((name,
                 "USABLE" if usable else "skip",
                 "success=%s" % ep.get("success"),
                 "label=%s" % ep.get("label"),
                 "src=%s" % ep.get("label_source"),
                 "plate=%s" % ep.get("plate"),
                 "conf=%s" % ep.get("decision_conf"),
                 "colour=%s" % ep.get("colour"),
                 "shape=%s" % ep.get("shape"),
                 "views=%s" % ",".join(views),
                 "station=%s" % station.get("verdict"),
                 "pre_bbox=%s" % bool(pre.get("bbox")),
                 "pngs=%s" % " ".join(pngs),
                 "fail=%s" % str(ep.get("failure"))[:70] + mismatch))
for r in rows:
    print(" | ".join(str(x) for x in r))
print("usable:", sum(1 for r in rows if len(r) > 1 and r[1] == "USABLE"), "of", len(rows))
