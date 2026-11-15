import collections
import csv
import datetime
import glob
import json
import os
import shlex
import subprocess
import sys
import traceback

R = "/home/intel/intel-junction-2026"
SORT = os.path.join(R, "logs", "sort")
MON = os.path.join(R, "logs", "monitor")
WRIST = os.path.join(R, "data", "wrist", "raw")
STUDIO = os.path.join(R, "data", "studio-datasets")
TAGS = ("T17", "T18", "T19", "T20", "T21", "T22")
MONS = []
WMS = []


def p(*a):
    print(*a, flush=True)


def sh(cmd, timeout=120):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return "ERR " + type(e).__name__ + " " + str(e)


def run(args, timeout=120):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def utc(t):
    if t is None:
        return None
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def section(name, fn):
    p("=====", name)
    try:
        fn()
    except Exception:
        p("SECTION ERROR", name)
        p(traceback.format_exc())


def s_env():
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    p("UTC", now, "host", os.uname().nodename, "cpus", os.cpu_count())
    p("PYTHON", sys.version.split()[0], sys.executable)
    p("R", sorted(os.listdir(R)))
    for m in ["pyarrow", "pandas", "numpy", "av", "torchcodec", "cv2", "PIL", "lerobot", "torch"]:
        try:
            mod = __import__(m)
            p("MOD", m, getattr(mod, "__version__", "ok"))
        except Exception as e:
            p("MOD", m, "MISSING", type(e).__name__)
    p("ENVS", sh("ls -1 " + R + "/envs 2>&1 | tr '\\n' ' '"))
    p(sh("which ffmpeg ffprobe; ffmpeg -hide_banner -version 2>&1 | head -1; ffmpeg -hide_banner -encoders 2>/dev/null | grep -E 'svtav1|libx264|libaom' | cut -c1-100"))
    p("PROC", sh("ps -eo pid,etime,args | grep -E 'sort_any|monitor|wrist|ffmpeg|record' | grep -v grep | cut -c1-150 | head -20"))


def probe(path):
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
               "format=start_time,duration,size:stream=codec_name,pix_fmt,width,height,r_frame_rate,"
               "avg_frame_rate,nb_frames,time_base,has_b_frames", "-of", "json", path], timeout=120)
    try:
        j = json.loads(out)
    except Exception:
        return {"err": out[:160]}
    f = j.get("format", {})
    s = (j.get("streams") or [{}])[0]
    info = {}
    for k in ("start_time", "duration", "size"):
        try:
            info[k] = float(f.get(k))
        except (TypeError, ValueError):
            info[k] = None
    for k in ("codec_name", "pix_fmt", "width", "height", "r_frame_rate", "avg_frame_rate", "nb_frames",
              "time_base", "has_b_frames"):
        info[k] = s.get(k)
    return info


def packets(path, n=3000):
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+#" + str(n),
               "-show_entries", "packet=pts_time", "-of", "csv=p=0", path], timeout=300)
    pts = []
    for line in out.splitlines():
        try:
            pts.append(float(line.strip().rstrip(",")))
        except ValueError:
            pass
    if len(pts) < 3:
        return {"n": len(pts)}
    dts = [b - a for a, b in zip(pts, pts[1:])]
    srt = sorted(dts)
    return {"n": len(pts), "first": round(pts[0], 4), "last": round(pts[-1], 4),
            "nonpositive_dt": sum(1 for x in dts if x <= 0),
            "dt_ms_p05": round(srt[len(srt) // 20] * 1000, 3),
            "dt_ms_median": round(srt[len(srt) // 2] * 1000, 3),
            "dt_ms_p95": round(srt[(len(srt) * 19) // 20] * 1000, 3),
            "dt_ms_max": round(max(dts) * 1000, 2),
            "frac_30fps": round(sum(1 for x in dts if 0.028 < x < 0.038) / len(dts), 3),
            "frac_15fps": round(sum(1 for x in dts if 0.060 < x < 0.070) / len(dts), 3),
            "frac_backward": round(sum(1 for x in dts if x < 0) / len(dts), 3)}


def s_monitor():
    names = sorted(x for x in os.listdir(MON) if x.endswith(".mkv") and (x.startswith("top_") or x.startswith("close_")))
    p("MON mkv", len(names), "first", names[:2], "last", names[-3:])
    for nm in names:
        info = probe(os.path.join(MON, nm))
        info["name"] = nm
        MONS.append(info)
        p("MON", nm, "start", info.get("start_time"), "dur_or_end", info.get("duration"),
          "MB", round((info.get("size") or 0) / 1e6, 1), info.get("codec_name"), info.get("width"),
          info.get("height"), "rfr", info.get("r_frame_rate"), "avg", info.get("avg_frame_rate"),
          "nb", info.get("nb_frames"), "tb", info.get("time_base"), "bfr", info.get("has_b_frames"),
          info.get("err", ""))
    for m in [m for m in MONS if any(t in m["name"] for t in TAGS)][:12]:
        p("PKT", m["name"], packets(os.path.join(MON, m["name"])))


def s_wrist():
    if not os.path.isdir(WRIST):
        p("WRIST dir missing")
        return
    names = sorted(os.listdir(WRIST))
    p("WRIST count", len(names), "first", names[:2], "last", names[-2:])
    sel = [x for x in names if any(t in x for t in TAGS)] or names[:400]
    p("WRIST sel", len(sel), sel[:2], sel[-2:])
    for nm in sel:
        info = probe(os.path.join(WRIST, nm))
        info["name"] = nm
        WMS.append(info)
    for info in (WMS[:1] + WMS[-1:]):
        p("WRIST", info["name"], "start", info.get("start_time"), "dur_or_end", info.get("duration"),
          "codec", info.get("codec_name"), info.get("width"), info.get("height"), "rfr", info.get("r_frame_rate"),
          "avg", info.get("avg_frame_rate"), "nb", info.get("nb_frames"), "tb", info.get("time_base"),
          "bfr", info.get("has_b_frames"), info.get("err", ""))
        p("WRIST PKT", info["name"], packets(os.path.join(WRIST, info["name"]), 300))


def s_clock():
    p(sh("grep -n -i -E 'lag|sign|positive|negative|convention|offset' " +
         shlex.quote(R + "/work/P3-VLA-CLOCK/E1/method.md") + " | cut -c1-260 | head -60"))
    p(sh("head -4 " + shlex.quote(R + "/work/P3-VLA-CLOCK/E1/offsets.csv") + " | cut -c1-300"))


def s_code():
    p("APP vla", sh("ls -1 " + shlex.quote(R + "/app/vla") + " 2>&1 | tr '\\n' ' '"))
    p("APP data", sh("ls -1 " + shlex.quote(R + "/app/data") + " 2>&1 | tr '\\n' ' '"))
    p("LEROBOT refs", sh("grep -l -i lerobot " + shlex.quote(R + "/app/vla") + "/*.py " +
                         shlex.quote(R + "/app/data") + "/*.py 2>/dev/null | head -20"))
    p(sh("grep -n -E 'episode.json|cycles.jsonl|joints.csv|success|colour|plate|decision|shape|t_start|t_end|mkv|time.time' " +
         shlex.quote(R + "/app/tools/sort_any.py") + " | cut -c1-200 | head -70"))
    p("OURS-LEROBOT", sh("ls -ld " + shlex.quote(R + "/data/ours-lerobot") + " 2>&1"))
    p("DATA dirs", sh("ls -1 " + shlex.quote(R + "/data") + " 2>&1 | tr '\\n' ' '"))


def s_studio():
    import pyarrow.parquet as pq
    infos = sorted(glob.glob(STUDIO + "/**/meta/info.json", recursive=True))
    p("INFO_JSON count", len(infos))
    parsed = []
    for f in infos:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            p("BAD", f, e)
            continue
        parsed.append((f, d))
        vids = [k for k, v in d.get("features", {}).items() if v.get("dtype") == "video"]
        p("DS", os.path.relpath(os.path.dirname(os.path.dirname(f)), R), "codebase", d.get("codebase_version"),
          "robot", d.get("robot_type"), "eps", d.get("total_episodes"), "frames", d.get("total_frames"),
          "tasks", d.get("total_tasks"), "fps", d.get("fps"), "video_keys", vids)
    if not parsed:
        return
    f, d = next(((f, d) for f, d in parsed if d.get("total_episodes") == 16), parsed[0])
    root = os.path.dirname(os.path.dirname(f))
    p("PICK", os.path.relpath(root, R))
    p("INFO_TOP", json.dumps({k: v for k, v in d.items() if k != "features"})[:1500])
    p("FEATURES", json.dumps(d.get("features", {}))[:3500])
    for pat in ["data/chunk-*/file-*.parquet", "meta/episodes/chunk-*/file-*.parquet", "meta/tasks.parquet"]:
        hits = sorted(glob.glob(os.path.join(root, pat)))
        p("GLOB", pat, len(hits))
        if not hits:
            continue
        t = pq.read_table(hits[0])
        p(" SCHEMA", [(x.name, str(x.type)) for x in t.schema])
        p(" ROWS", t.num_rows, os.path.relpath(hits[0], root))
        if pat.startswith("meta/tasks"):
            p(" TASKS", t.to_pylist())
        elif t.num_rows:
            row = t.slice(0, 1).to_pylist()[0]
            p(" ROW0", {k: str(v)[:90] for k, v in row.items() if not k.startswith("stats/")})
    stp = os.path.join(root, "meta", "stats.json")
    if os.path.exists(stp):
        st = json.load(open(stp, encoding="utf-8"))
        p("STATS keys", list(st.keys()))
        for k in ("observation.state", "action", "timestamp"):
            if k in st:
                p("STATS", k, {kk: str(vv)[:70] for kk, vv in st[k].items()})
    vids = sorted(glob.glob(os.path.join(root, "videos", "**", "*.mp4"), recursive=True))
    p("VIDEOS", len(vids), [os.path.relpath(v, root) for v in vids[:4]])
    if vids:
        p(sh("ffprobe -v error -show_entries stream=codec_name,profile,pix_fmt,width,height,r_frame_rate,"
             "avg_frame_rate,nb_frames,time_base,start_time:format=start_time,duration -of compact=p=0 " +
             shlex.quote(vids[0])))


def covering(lst, t0, t1):
    hits = []
    for m in lst:
        st, du = m.get("start_time"), m.get("duration")
        if st is None or du is None:
            continue
        end = du if du > 1e9 else st + du
        if st <= t0 and end >= t1:
            hits.append(m["name"])
    return hits


def overlapping(lst, t0, t1):
    hits = []
    for m in lst:
        st, du = m.get("start_time"), m.get("duration")
        if st is None or du is None:
            continue
        end = du if du > 1e9 else st + du
        if st <= t1 and end >= t0:
            hits.append(m["name"])
    return hits


def s_sort():
    runs = sorted(x for x in os.listdir(SORT) if os.path.isdir(os.path.join(SORT, x)))
    p("RUNS", runs)
    allj = []
    keyfreq = collections.Counter()
    tops = [m for m in MONS if m["name"].startswith("top_")]
    closes = [m for m in MONS if m["name"].startswith("close_")]
    for run_name in runs:
        rd = os.path.join(SORT, run_name)
        cyc = os.path.join(rd, "cycles.jsonl")
        if os.path.exists(cyc):
            lines = [x for x in open(cyc, encoding="utf-8").read().splitlines() if x.strip()]
            p("CYCLES", run_name, "lines", len(lines))
            if lines:
                p("CYC0", lines[0][:700])
                p("CYCN", lines[-1][:700])
        eps = sorted(x for x in os.listdir(rd) if x.startswith("ep_") and os.path.isdir(os.path.join(rd, x)))
        p("RUN", run_name, "episodes", len(eps), eps[:1], eps[-1:])
        for ep in eps:
            ed = os.path.join(rd, ep)
            files = sorted(os.listdir(ed))
            js = {}
            ej = os.path.join(ed, "episode.json")
            if os.path.exists(ej):
                try:
                    js = json.load(open(ej, encoding="utf-8"))
                except Exception as e:
                    js = {"_err": str(e)}
            if not isinstance(js, dict):
                js = {"_value": str(js)[:200]}
            keyfreq.update(js.keys())
            nrows, t0, t1 = 0, None, None
            jc = os.path.join(ed, "joints.csv")
            if os.path.exists(jc):
                with open(jc, encoding="utf-8") as fh:
                    rdr = csv.reader(fh)
                    hdr = next(rdr, None)
                    if run_name == runs[0] and ep == eps[0]:
                        p("JOINTS header", hdr)
                    for row in rdr:
                        try:
                            tt = float(row[0])
                        except (ValueError, IndexError):
                            continue
                        nrows += 1
                        if t0 is None:
                            t0 = tt
                        t1 = tt
            allj.append({"run": run_name, "ep": ep, "js": js, "t0": t0, "t1": t1, "rows": nrows})
            keep = {k: js.get(k) for k in ("success", "decision", "plate", "colour", "color", "shape", "outcome",
                                           "reason", "failure", "interrupted", "tape") if k in js}
            tim = json.dumps(js.get("timings"))[:180] if "timings" in js else ""
            cov, wov = "", ""
            if t0 is not None:
                cov = "top=%s close=%s" % (covering(tops, t0, t1), covering(closes, t0, t1))
                wov = len(overlapping(WMS, t0, t1))
            p("EP", run_name, ep, "files", len(files), "jpg", sum(1 for x in files if x.endswith(".jpg")),
              "crops" if "crops" in files else "nocrops", "rows", nrows,
              "t0", None if t0 is None else round(t0, 3), "dur", None if t0 is None else round(t1 - t0, 2),
              "utc0", utc(t0), keep, tim, cov, "wrist_overlaps", wov)
        p("FILES first ep of run", run_name, sorted(os.listdir(os.path.join(rd, eps[0]))) if eps else None)
    p("SUCCESS counts", collections.Counter(str(r["js"].get("success")) for r in allj))
    p("KEYS", keyfreq.most_common(60))
    succ = [r for r in allj if r["js"].get("success") is True]
    if succ:
        p("FIRST_SUCCESS", succ[0]["run"], succ[0]["ep"], json.dumps(succ[0]["js"])[:2600])
    if allj:
        p("NEWEST", allj[-1]["run"], allj[-1]["ep"], json.dumps(allj[-1]["js"])[:2600])
    for run_name in runs:
        rr = [r for r in allj if r["run"] == run_name and r["t0"] is not None]
        if rr:
            p("RUNSPAN", run_name, len(rr), utc(min(r["t0"] for r in rr)), utc(max(r["t1"] for r in rr)))


if __name__ == "__main__":
    section("env", s_env)
    section("monitor", s_monitor)
    section("wrist", s_wrist)
    section("clock", s_clock)
    section("code", s_code)
    section("studio", s_studio)
    section("sort", s_sort)
    p("DISCOVERY_DONE", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
