"""Compact summary of R/work/P2-B-ANOM3/E1/multi/manifest.tsv (read-only, nothing is written)."""
import collections
import csv
import os

OUT = "/home/intel/intel-junction-2026/work/P2-B-ANOM3/E1/multi"
with open(os.path.join(OUT, "manifest.tsv"), encoding="utf-8") as fh:
    rows = [r for r in csv.DictReader(fh, delimiter="\t") if r["status"] != "skip_episode"]
print("rows for usable episodes", len(rows))
groups = collections.OrderedDict()
for r in rows:
    groups.setdefault((r["view"], r["key"], r["label"]), []).append(r)
for (view, key, label), rs in groups.items():
    st = collections.Counter(r["status"] for r in rs)
    print("%-12s %-18s %-6s %s" % (view, key, label, dict(sorted(st.items()))))
print()
print("wrist_multi ok rows: colour", dict(collections.Counter(r["blob_colour"] for r in rows
                                                              if r["view"] == "wrist_multi" and r["status"] == "ok")))
ok_w = [r for r in rows if r["view"] == "wrist_multi" and r["status"] == "ok"]
if ok_w:
    sides = [float(r["side_px"]) for r in ok_w]
    dists = [float(r["blob_dist"]) for r in ok_w]
    jaws = [float(r["jaw_frac"]) for r in ok_w]
    print("wrist side_px min %.1f max %.1f; blob_dist max %.1f; jaw_frac max %.3f; clipped %d"
          % (min(sides), max(sides), max(dists), max(jaws), sum(int(r["clipped"]) for r in ok_w)))
print()
print("close rows (not scan):")
for r in rows:
    if r["view"] != "close_multi" or r["status"] == "skip_scan":
        continue
    print("  %-18s idx %3s %-22s verdict %-18s hue_diff %-4s blob %-7s hue %-4s reason %s"
          % (r["key"], r["idx"], r["status"], r["close_verdict"] or "-", r["close_hue_diff"] or "-",
             r["blob_colour"] or "-", r["blob_hue"] or "-", (r["reason"] or "")[:70]))
