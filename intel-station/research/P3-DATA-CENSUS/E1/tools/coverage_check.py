import csv, collections, re
R = "/home/intel/intel-junction-2026"
E1 = R + "/work/P3-DATA-CENSUS/E1"
rows = [r for r in csv.DictReader(open(E1 + "/catalogue.csv")) if r["source"] != "ours/twin-chat"]
exact, pref = set(), []
for r in rows:
    p = re.sub(r"\s*\[[^\]]*\]$", "", r["path"]).rstrip("/")
    exact.add(p)
    pref.append(p + "/")
files = []
for line in open(E1 + "/raw/files_all.tsv"):
    p, s = line.rstrip("\n").split("\t")[:2]
    files.append((p, int(s)))
cov = 0
unc = []
for p, s in files:
    if p in exact or any(p.startswith(x) for x in pref):
        cov += 1
    else:
        unc.append((p, s))
print("files under logs/data/run", len(files), "bytes", sum(s for _, s in files))
print("covered by a unit path (exact or folder prefix)", cov)
print("not covered", len(unc), "bytes", sum(s for _, s in unc))
grp = collections.Counter(); gb = collections.Counter()
for p, s in unc:
    k = "/".join(p.split("/")[:3]) if p.count("/") >= 2 else p
    grp[k] += 1; gb[k] += s
for k, v in grp.most_common(30):
    print(v, round(gb[k] / 1e6, 2), "MB", k)
with open(E1 + "/raw/coverage_uncovered.tsv", "w") as f:
    for p, s in unc:
        f.write("%s\t%d\n" % (p, s))
