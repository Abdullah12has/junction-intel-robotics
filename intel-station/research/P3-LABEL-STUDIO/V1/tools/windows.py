import json, os, subprocess
ROOT = "/home/intel/intel-junction-2026/data"
OUT = "/home/intel/intel-junction-2026/work/P3-LABEL-STUDIO/V1/out"
FR = "/home/intel/intel-junction-2026/work/P3-LABEL-STUDIO/V1/frames"
M = json.load(open(os.path.join(OUT, "meta_dump.json")))
SHA = {}
for ln in open(os.path.join(OUT, "sha256.txt")):
    h, path = ln.rstrip("\n").split("  ", 1)
    SHA[path] = (h[:12], path)
CAMS = ["243722064316", "243622061421"]  # cam1=243722064316 (close/top?), cam2=243622061421

wins = {}
for ds, e in M.items():
    root = e["path"]
    for ep in e.get("episodes", []):
        key = []
        paths = []
        for c in CAMS:
            cam = "observation.images.realsense " + c
            vp = f"{root}/videos/{cam}/chunk-{int(ep.get(f'videos/{cam}/chunk_index', 0)):03d}/file-{int(ep.get(f'videos/{cam}/file_index', 0)):03d}.mp4"
            h = SHA.get(vp, ("MISSING", vp))[0]
            key.append((h, round(float(ep[f"videos/{cam}/from_timestamp"]), 1), round(float(ep[f"videos/{cam}/to_timestamp"]), 1)))
            paths.append(vp)
        k = (key[0][0], key[0][1], key[0][2])
        w = wins.setdefault(k, {"cam1_path": paths[0], "cam2_path": paths[1], "cam1_from": key[0][1], "cam1_to": key[0][2],
                                "cam2_from": key[1][1], "cam2_to": key[1][2], "cam2_sha": key[1][0], "entries": []})
        w["entries"].append(f"{ds[:8]}:{ep['episode_index']}:{ep['length']}:{(ep.get('tasks') or [''])[0]}")

rows = []
for i, (k, w) in enumerate(sorted(wins.items(), key=lambda kv: (kv[0][0], kv[0][1])), 1):
    w["wid"] = f"W{i:02d}"
    rows.append(w)

# probe resolution once
probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate,codec_name",
                        "-of", "json", rows[0]["cam1_path"]], capture_output=True, text=True).stdout
print("probe cam1", probe.replace("\n", ""))
probe2 = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate,codec_name",
                         "-of", "json", rows[0]["cam2_path"]], capture_output=True, text=True).stdout
print("probe cam2", probe2.replace("\n", ""))

with open(os.path.join(OUT, "windows.tsv"), "w") as f:
    f.write("wid\tcam1_from\tcam1_to\tcam2_from\tcam2_to\tcam1_path\tentries\n")
    for w in rows:
        f.write(f"{w['wid']}\t{w['cam1_from']}\t{w['cam1_to']}\t{w['cam2_from']}\t{w['cam2_to']}\t{w['cam1_path'].replace(ROOT + '/', '')}\t{' | '.join(w['entries'])}\n")
print("windows", len(rows))
