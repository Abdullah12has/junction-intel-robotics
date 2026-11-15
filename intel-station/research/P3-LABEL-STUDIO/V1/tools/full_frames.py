import json, os, subprocess
ROOT = "/home/intel/intel-junction-2026/data"
OUT = "/home/intel/intel-junction-2026/work/P3-LABEL-STUDIO/V1/out"
FR = "/home/intel/intel-junction-2026/work/P3-LABEL-STUDIO/V1/frames"
TMP = "/tmp/v1full"
os.makedirs(TMP, exist_ok=True)
os.makedirs(FR, exist_ok=True)
M = json.load(open(os.path.join(OUT, "meta_dump.json")))
C1 = "observation.images.realsense 243722064316"
C2 = "observation.images.realsense 243622061421"

wins = {}
for ds, e in M.items():
    root = e["path"]
    for ep in e.get("episodes", []):
        rec = {}
        for tag, cam in (("c1", C1), ("c2", C2)):
            ci = int(ep.get(f"videos/{cam}/chunk_index", 0))
            fi = int(ep.get(f"videos/{cam}/file_index", 0))
            rec[tag + "_path"] = f"{root}/videos/{cam}/chunk-{ci:03d}/file-{fi:03d}.mp4"
            rec[tag + "_from"] = float(ep[f"videos/{cam}/from_timestamp"])
            rec[tag + "_to"] = float(ep[f"videos/{cam}/to_timestamp"])
        key = (rec["c1_path"], round(rec["c1_from"], 1), round(rec["c1_to"], 1), round(rec["c2_from"], 1), round(rec["c2_to"], 1))
        w = wins.setdefault(key, dict(rec, entries=[]))
        w["entries"].append(f"{ds[:8]}:{ep['episode_index']}:{ep['length']}:{(ep.get('tasks') or [''])[0]}")

rows = []
for i, (k, w) in enumerate(sorted(wins.items(), key=lambda kv: (kv[0][0], kv[0][1])), 1):
    w["wid"] = f"W{i:02d}"
    rows.append(w)

with open(os.path.join(OUT, "windows2.tsv"), "w") as f:
    f.write("wid\tc1_from\tc1_to\tc2_from\tc2_to\tc1_path\tc2_path\tentries\n")
    for w in rows:
        f.write("\t".join([w["wid"], str(round(w["c1_from"], 3)), str(round(w["c1_to"], 3)), str(round(w["c2_from"], 3)), str(round(w["c2_to"], 3)),
                           w["c1_path"].replace(ROOT + "/", ""), w["c2_path"].replace(ROOT + "/", ""), " | ".join(w["entries"])]) + "\n")

def grab(path, t, out):
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1", out])
    return r.returncode == 0

FRACS = [0.05, 0.25, 0.45, 0.65, 0.85, 0.97]
report = []
for w in rows:
    ok = True
    for tag in ("c1", "c2"):
        d = w[f"{tag}_to"] - w[f"{tag}_from"]
        for j, fr in enumerate(FRACS):
            t = w[f"{tag}_from"] + d * fr
            ok = grab(w[f"{tag}_path"], t, f"{TMP}/{w['wid']}_{tag}_{j}.png") and ok
    if not ok:
        report.append(f"{w['wid']} extraction FAILED")
        continue
    # row1 = close view (serial 243722064316), row2 = top view (serial 243622061421), native 640x480 each
    ins = []
    for tag in ("c1", "c2"):
        for j in range(len(FRACS)):
            ins += ["-i", f"{TMP}/{w['wid']}_{tag}_{j}.png"]
    fc = ("[0][1][2]hstack=inputs=3[a];[3][4][5]hstack=inputs=3[b];"
          "[6][7][8]hstack=inputs=3[c];[9][10][11]hstack=inputs=3[d];[a][b][c][d]vstack=inputs=4")
    r = subprocess.run(["ffmpeg", "-v", "error", "-y"] + ins + ["-filter_complex", fc, "-frames:v", "1", f"{FR}/F{w['wid'][1:]}.png"])
    report.append(f"{w['wid']} montage {'ok' if r.returncode == 0 else 'FAILED'}")
open(os.path.join(OUT, "full_frames.log"), "w").write("\n".join(report) + "\n")
print("windows", len(rows))
print("\n".join(report))
