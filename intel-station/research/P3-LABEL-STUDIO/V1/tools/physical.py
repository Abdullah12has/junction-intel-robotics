import json, os
OUT = "/home/intel/intel-junction-2026/work/P3-LABEL-STUDIO/V1/out"
M = json.load(open(os.path.join(OUT, "meta_dump.json")))
SHA = {}
for ln in open(os.path.join(OUT, "sha256.txt")):
    h, path = ln.rstrip("\n").split("  ", 1)
    SHA[path.replace("/home/intel/intel-junction-2026/data/", "")] = h[:12]

CAMS = ["243722064316", "243622061421"]
rows = []
for ds, e in M.items():
    root = e["path"].replace("/home/intel/intel-junction-2026/data/", "")
    for ep in e.get("episodes", []):
        r = {"ds": ds[:8], "src": root.split("/")[0][:7], "ep": ep["episode_index"],
             "len": ep["length"], "task": (ep.get("tasks") or [""])[0],
             "dq": ep["dataset_from_index"], "dt": ep["dataset_to_index"]}
        for c in CAMS:
            cam = "observation.images.realsense " + c
            fk = f"videos/{cam}/file_index"
            fromk = f"videos/{cam}/from_timestamp"
            tok = f"videos/{cam}/to_timestamp"
            ci = f"videos/{cam}/chunk_index"
            vpath = f"{root}/videos/{cam}/chunk-{int(ep.get(ci, 0)):03d}/file-{int(ep.get(fk, 0)):03d}.mp4"
            r[c + "_file"] = f"{ep.get(fk)}"
            r[c + "_sha"] = SHA.get(vpath, "MISSING")
            r[c + "_from"] = round(float(ep.get(fromk, -1)), 3)
            r[c + "_to"] = round(float(ep.get(tok, -1)), 3)
        r["data_sha"] = SHA.get(f"{root}/data/chunk-000/file-000.parquet", "?")
        rows.append(r)

# physical identity: same cam1 file sha + same from timestamp (+ same length)
groups = {}
for r in rows:
    key = (r[CAMS[0] + "_sha"], r[CAMS[0] + "_from"], r["len"])
    groups.setdefault(key, []).append(r)

lines = ["gid\tsrc\tds\tep\tlen\ttask\tcam1_sha\tcam1_from\tcam1_to\tcam2_sha\tcam2_from\tcam2_to\tdata_file_sha"]
for gid, (key, members) in enumerate(sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])), 1):
    for m in members:
        lines.append("\t".join(str(x) for x in [f"P{gid:02d}", m["src"], m["ds"], m["ep"], m["len"], m["task"],
                      m[CAMS[0] + "_sha"], m[CAMS[0] + "_from"], m[CAMS[0] + "_to"],
                      m[CAMS[1] + "_sha"], m[CAMS[1] + "_from"], m[CAMS[1] + "_to"], m["data_sha"]]))
open(os.path.join(OUT, "physical_map.tsv"), "w").write("\n".join(lines) + "\n")
print("entries", len(rows), "physical groups", len(groups))
