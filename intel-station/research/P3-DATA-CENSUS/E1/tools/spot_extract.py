import os, subprocess, concurrent.futures as cf
R = "/home/intel/intel-junction-2026"
E1 = R + "/work/P3-DATA-CENSUS/E1"
FR = E1 + "/frames"
os.makedirs(FR, exist_ok=True)
plan = [l.rstrip("\n").split("\t") for l in open(E1 + "/raw/spot_plan.tsv")][1:]
jobs = []
for key, src, kind, path, dur, ta, tb in plan:
    if dur == "-":
        continue
    jobs.append((key, path, "a", ta))
    jobs.append((key, path, "b", tb))
def run(job):
    key, path, tag, t = job
    out = os.path.join(FR, "%s_%s.png" % (key, tag))
    cmd = ["nice", "-n", "19", "/usr/bin/ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-threads", "2",
           "-ss", t, "-i", os.path.join(R, path), "-an", "-frames:v", "1", "-vf", "scale=192:144", "-y", out]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    ok = p.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 0
    err = (p.stderr or "").strip().replace("\n", " | ").replace("\t", " ")[:200]
    return (key, tag, t, p.returncode, ok, os.path.getsize(out) if ok else 0, err)
with cf.ThreadPoolExecutor(max_workers=4) as ex:
    res = list(ex.map(run, jobs))
with open(E1 + "/raw/extract_log.tsv", "w") as f:
    f.write("key\ttag\tt_s\trc\tok\tbytes\tstderr\n")
    for r in res:
        f.write("%s\t%s\t%s\t%d\t%s\t%d\t%s\n" % r)
print("jobs", len(jobs), "ok", sum(1 for r in res if r[4]), "failed", sum(1 for r in res if not r[4]))
