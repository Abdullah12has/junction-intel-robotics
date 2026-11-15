import json, collections, os
OUT = "/home/intel/intel-junction-2026/work/P3-LABEL-STUDIO/V1/out"
M = json.load(open(os.path.join(OUT, "meta_dump.json")))

lines = []
def p(*a):
    lines.append(" ".join(str(x) for x in a))

# sha groups
sha = {}
for ln in open(os.path.join(OUT, "sha256.txt")):
    h, path = ln.rstrip("\n").split("  ", 1)
    sha.setdefault(h, []).append(path)
p("== duplicate groups (sha256 identical) ==")
for h, paths in sha.items():
    if len(paths) > 1:
        p(h[:12], len(paths))
        for x in paths:
            p("   ", x.replace("/home/intel/intel-junction-2026/data/", ""))

p("")
p("== per dataset ==")
for ds, e in M.items():
    info = e.get("info", {})
    feats = info.get("features", {})
    p("##", ds, "| fps", info.get("fps"), "| total_episodes", info.get("total_episodes"),
      "| total_frames", info.get("total_frames"), "| codebase", info.get("codebase_version"))
    p("  tasks:", [t for t in e.get("tasks", [])][:6])
    cams = [k for k in feats if k.startswith("observation.images")]
    p("  cams:", cams)
    if "observation.state" in feats:
        p("  state names:", feats["observation.state"].get("names"))
    if "action" in feats:
        p("  action names:", feats["action"].get("names"))
    for ep in e.get("episodes", []):
        keep = {k: ep.get(k) for k in ep if not k.startswith("stats/") and not k.startswith("meta/") and k not in ("tasks",)}
        p("  ep", json.dumps(keep, default=str)[:600])
        p("     tasks:", ep.get("tasks"))

open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(lines) + "\n")
print("lines", len(lines))
