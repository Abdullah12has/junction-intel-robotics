import json, glob, os
R = "/home/intel/intel-junction-2026"
E1 = R + "/work/P3-DATA-CENSUS/E1"
P = {}
for line in open(E1 + "/raw/probe_all.jsonl"):
    r = json.loads(line)
    P[r["path"]] = r
units = [json.loads(l) for l in open(E1 + "/raw/unit_detail.jsonl")]
rows = []
for u in units:
    if u["kind"] == "video" and u["source"] != "ours/twin-chat" and u.get("unique", True):
        rows.append((u["id"], u["source"], "video", u["path"]))
for att in sorted(glob.glob(R + "/data/sim/2026-11-14/dataset-150-raw/attempt-*")):
    a = os.path.basename(att)
    n = a.split("-")[1]
    for cam in ("overhead", "wrist"):
        rows.append(("SIM%s-%s" % (n, cam[:2]), "friend sim", "sim_attempt", "data/sim/2026-11-14/dataset-150-raw/%s/%s.mp4" % (a, cam)))
with open(E1 + "/raw/spot_plan.tsv", "w") as out:
    out.write("key\tsource\tkind\tpath\tdur_s\tt_a_s\tt_b_s\n")
    for key, src, kind, path in rows:
        pr = P.get(path)
        d = float(pr["true_duration_s"]) if pr and pr.get("true_duration_s") not in (None, "", "None") else None
        if d is None:
            out.write("%s\t%s\t%s\t%s\t-\t-\t-\n" % (key, src, kind, path))
        else:
            out.write("%s\t%s\t%s\t%s\t%.3f\t%.3f\t%.3f\n" % (key, src, kind, path, d, 0.25 * d, 0.75 * d))
print("plan rows", len(rows))
