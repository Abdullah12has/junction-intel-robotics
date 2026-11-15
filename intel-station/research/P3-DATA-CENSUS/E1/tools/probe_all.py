import json, os, subprocess, statistics, sys, time
R = "/home/intel/intel-junction-2026"
E1 = os.path.join(R, "work/P3-DATA-CENSUS/E1")
OUT = os.path.join(E1, "raw/probe_all.jsonl")

def collect():
    out = []
    def add(d, exts, recursive=False):
        if not os.path.isdir(d):
            return
        for root, dirs, files in os.walk(d):
            for f in sorted(files):
                if f.endswith(exts):
                    out.append(os.path.join(root, f))
            if not recursive:
                break
    add(os.path.join(R, "twin/work"), (".mp4", ".mkv"), True)
    add(os.path.join(R, "logs/monitor"), (".mkv",))
    add(os.path.join(R, "logs/gm"), (".mkv",))
    add(os.path.join(R, "data/wrist/raw"), (".mkv",))
    add(os.path.join(R, "data/studio-datasets"), (".mp4",), True)
    add(os.path.join(R, "data/curated"), (".mp4",), True)
    add(os.path.join(R, "data/sim/2026-11-14/extract-lerobot"), (".mp4",), True)
    add(os.path.join(R, "data/sim/2026-11-14/dataset-150-raw"), (".mp4",), True)
    add(os.path.join(R, "data/sim/live/2026-11-14/dataset-150-raw"), (".mp4",), True)
    return sorted(set(out))

def ffjson(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
        "format=start_time,duration,size,format_name:stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,start_time,duration",
        "-of", "json", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    d = json.loads(p.stdout) if p.stdout.strip() else {}
    return p.returncode, d, p.stderr.strip()

def packets(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "packet=pts_time", "-of", "csv=p=0", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    vals, na = [], 0
    for line in p.stdout.splitlines():
        s = line.strip().rstrip(",")
        if not s:
            continue
        if s == "N/A":
            na += 1
            continue
        vals.append(float(s))
    return p.returncode, vals, na, p.stderr.strip()

def pstats(vals, na):
    rec = {"n_packets": len(vals) + na, "n_pts_na": na}
    if len(vals) < 2:
        return rec
    diffs = [b - a for a, b in zip(vals, vals[1:])]
    med = statistics.median(diffs)
    gaps = [d for d in diffs if d > 1.5 * med]
    rec.update({
        "first_pts": vals[0], "last_pts": vals[-1], "median_dt": med,
        "implied_fps": (1.0 / med) if med > 0 else None,
        "n_gaps": len(gaps), "max_gap_s": max(gaps) if gaps else 0.0,
        "missing_s": sum(g - med for g in gaps),
        "n_nonmono": sum(1 for d in diffs if d <= 0),
        "true_duration_s": (vals[-1] - vals[0]) + med,
    })
    return rec

def main():
    files = collect()
    print("files", len(files), flush=True)
    t0 = time.time()
    with open(OUT, "w") as fo:
        for i, f in enumerate(files):
            st = os.stat(f)
            rec = {"path": os.path.relpath(f, R), "size": st.st_size, "mtime": st.st_mtime}
            rc, d, err = ffjson(f)
            rec["ffprobe_json_rc"] = rc
            rec["ffprobe_err"] = err[:300]
            fmt = d.get("format", {})
            rec["format"] = {k: fmt.get(k) for k in ("start_time", "duration", "format_name")}
            ss = d.get("streams", [])
            rec["stream"] = {k: ss[0].get(k) for k in ("codec_name", "width", "height", "r_frame_rate", "avg_frame_rate", "nb_frames")} if ss else None
            prc, vals, na, perr = packets(f)
            rec["packets_rc"] = prc
            rec["packets_err"] = perr[:300]
            rec.update(pstats(vals, na))
            fo.write(json.dumps(rec) + "\n")
            if i % 50 == 0:
                print("progress", i, round(time.time() - t0, 1), flush=True)
    print("done", len(files), round(time.time() - t0, 1), flush=True)

main()
