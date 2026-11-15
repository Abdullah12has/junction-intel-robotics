#!/usr/bin/env python3
# census_build.py: builds the P3-DATA-CENSUS unit catalogue (run as envs/data-env/bin/python -I).
# Reads only. Writes only under R/work/P3-DATA-CENSUS/E1/.
import csv, datetime, glob, hashlib, json, os, re, statistics, sys, time, zipfile
import numpy as np
import pyarrow.parquet as pq

R = "/home/intel/intel-junction-2026"
E1 = os.path.join(R, "work/P3-DATA-CENSUS/E1")
RAW = os.path.join(E1, "raw")
NOW = time.time()
SNAP = datetime.datetime.fromtimestamp(NOW, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def A(rel):
    return os.path.join(R, rel)

def rel(p):
    return os.path.relpath(p, R)

def hms(t):
    if t is None:
        return "-"
    return datetime.datetime.fromtimestamp(float(t), datetime.timezone.utc).strftime("%H:%M:%S")

def stamp_epoch(name):
    m = re.search(r"(\d{8})T(\d{6})", name)
    if not m:
        return None
    d = datetime.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return d.replace(tzinfo=datetime.timezone.utc).timestamp()

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

_SHA = {}
def sha_cached(path):
    if path not in _SHA:
        _SHA[path] = sha256(path)
    return _SHA[path]

def walk_size(d):
    total, n = 0, 0
    for root, _, files in os.walk(d):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
                n += 1
            except OSError:
                pass
    return total, n

def clean(s):
    return str(s).replace(",", ";").replace("\n", " ").replace("\r", " ")

UNITS = []
def add(kind, source, path, **kw):
    u = {"kind": kind, "source": source, "path": path, "t0": None, "t1": None,
         "dur": None, "size": None, "cams": "-", "frames": "-", "fps": "-",
         "joint": "-", "labels": "-", "links": [], "flags": [], "notes": "-",
         "mtime": None, "dsrc": None, "unique": True, "copy_of": None}
    u.update(kw)
    UNITS.append(u)
    return u

PROBE = {}
with open(os.path.join(RAW, "probe_all.jsonl")) as f:
    for line in f:
        r = json.loads(line)
        PROBE[r["path"]] = r

def cam_and_source(rp):
    name = os.path.basename(rp)
    if rp.startswith("logs/monitor/"):
        return "ours", ("top" if name.startswith("top_") else "close")
    if rp.startswith("logs/gm/"):
        return "ours", ("close" if "close_" in name else "wrist")
    if rp.startswith("data/wrist/raw/"):
        return "ours", "wrist"
    if rp.startswith("data/studio-datasets/") or rp.startswith("data/curated/"):
        m = re.search(r"realsense (\d+)", rp)
        return "team/friend dataset", ("rs" + m.group(1)) if m else "-"
    if rp.startswith("data/sim/2026-11-14/extract-lerobot/"):
        return "friend sim", ("camera1" if "camera1" in rp else "camera2")
    if rp.startswith("twin/work/"):
        m = re.search(r"realsense (\d+)", rp)
        if m:
            return "ours/twin-chat", "rs" + m.group(1)
        return "ours/twin-chat", "overlay"
    return None, None

def build_videos():
    for rp in sorted(PROBE):
        if rp.startswith("data/sim/") and not rp.startswith("data/sim/2026-11-14/extract-lerobot/"):
            continue  # friend sim raw and live mirror MP4s are carried inside the episode rows
        src, cam = cam_and_source(rp)
        if src is None:
            continue
        r = PROBE[rp]
        name = os.path.basename(rp)
        absolute = r.get("first_pts") is not None and r["first_pts"] > 1e8
        dur = r.get("true_duration_s")
        t0 = r.get("first_pts") if absolute else stamp_epoch(name)
        t1 = (t0 + dur) if (t0 is not None and dur is not None) else None
        fps = r.get("implied_fps")
        st = r.get("stream") or {}
        flags = []
        if r.get("ffprobe_json_rc") or r.get("packets_rc") or not r.get("n_packets"):
            flags.append("CORRUPT")
        flags.append("PTS_ABS" if absolute else "PTS_REL")
        if r.get("n_gaps"):
            flags.append("GAPS%d(max%.2fs;miss%.1fs)" % (r["n_gaps"], r["max_gap_s"], r["missing_s"]))
        if r.get("n_nonmono"):
            flags.append("NONMONO%d" % r["n_nonmono"])
        if st.get("r_frame_rate") not in (None, "30/1"):
            flags.append("RFR" + st["r_frame_rate"].replace("/", "_"))
        if fps is not None and abs(fps - 30.0) > 0.5:
            flags.append("FPS%.1f" % fps)
        if (not absolute) and stamp_epoch(name) is None:
            flags.append("NO_TIME")
        if rp.startswith("data/wrist/") and dur is not None and dur < 55:
            flags.append("SHORT_SEGMENT")
        if NOW - r["mtime"] < 900:
            flags.append("LIVE_MTIME<15min")
        codec = st.get("codec_name", "?")
        notes = "%s %sx%s" % (codec, st.get("width"), st.get("height"))
        if not absolute and rp.startswith(("logs/monitor/", "data/wrist/")):
            notes += "; relative pts"
        vds = None
        m = re.match(r"(data/(?:studio-datasets/[^/]+|curated/[^/]+|sim/2026-11-14/extract-lerobot/dataset-150)|twin/work/TW-REC/V\d/[^/]+(?:/export/[^/]+)?)/videos/", rp)
        if m:
            vds = m.group(1)
        add("video", src, rp, t0=t0, t1=t1, dur=dur, size=r["size"], cams=cam,
            frames=r.get("n_packets"), fps=(round(fps, 2) if fps else "-"), flags=flags,
            mtime=r["mtime"], notes=notes, dsrc=vds)


def joint_stats(path):
    if not os.path.exists(path):
        return None
    ts = []
    with open(path) as f:
        rd = csv.reader(f)
        next(rd, None)
        for row in rd:
            if row and row[0].strip():
                try:
                    ts.append(float(row[0]))
                except ValueError:
                    pass
    if len(ts) < 2:
        return {"rows": len(ts), "hz": None, "first": ts[0] if ts else None, "last": ts[-1] if ts else None, "maxgap": None}
    dts = [b - a for a, b in zip(ts, ts[1:])]
    med = statistics.median(dts)
    return {"rows": len(ts), "hz": (1.0 / med) if med > 0 else None, "first": ts[0], "last": ts[-1], "maxgap": max(dts)}

def build_sort_episodes():
    for run in ("any_1731", "any_1740"):
        base = A("logs/sort/" + run)
        for ep in sorted(glob.glob(os.path.join(base, "ep_*"))):
            ejp = os.path.join(ep, "episode.json")
            has_ej = os.path.exists(ejp)
            ej = json.load(open(ejp)) if has_ej else {}
            t0, t1 = ej.get("t_start"), ej.get("t_end")
            js = joint_stats(os.path.join(ep, "joints.csv"))
            if not has_ej and js is not None:
                t0, t1 = js["first"], js["last"]
            size, n = walk_size(ep)
            names = [x for x in os.listdir(ep) if x.endswith((".jpg", ".png"))]
            cams = [c for c in ("top", "close", "wrist") if any(x.endswith("_%s.jpg" % c) for x in names)]
            keys = [k for k in ("label", "colour", "plate", "shape", "success", "failure") if ej.get(k) is not None]
            flags = [] if has_ej else ["NO_EPISODE_JSON"]
            if js is None:
                flags.append("JOINTS_MISSING")
            elif t1 is not None and js["last"] is not None and (t1 - js["last"]) > 2.0:
                flags.append("JOINTS_END_EARLY%.1fs" % (t1 - js["last"]))
            if t0 is None:
                flags.append("NO_T_START")
            if "label" not in keys:
                flags.append("NO_LABEL")
            if ej.get("success") is False:
                flags.append("FAILED")
            for k in ("top", "close", "wrist_dir"):
                ref = (ej.get("cams") or {}).get(k)
                if ref and not os.path.exists(A(ref)):
                    flags.append("MISSING_REF:" + ref)
            jtxt = ("joints.csv %d rows %.1fHz" % (js["rows"], js["hz"])) if (js and js["hz"]) else "joints.csv missing"
            note = "%s ep%s" % (run, ej.get("episode", os.path.basename(ep)[3:]))
            if not has_ej:
                note += "; no episode.json, times from joints.csv"
            if ej.get("failure"):
                note += "; failure: " + clean(ej["failure"])[:60]
            dur = (t1 - t0) if (t0 is not None and t1 is not None) else None
            u = add("episode", "ours", rel(ep), t0=t0, t1=t1, dur=dur, size=size,
                    cams=("+".join(cams) if cams else "-"), frames="-", joint=jtxt,
                    labels=("+".join(keys) if keys else "-"), flags=flags, notes=clean(note))
            u["run"] = run
            u["nimg"] = len(names)

def npz_shapes(path):
    with np.load(path) as z:
        return {k: "x".join(str(d) for d in z[k].shape) for k in z.files}

SIM_FILES = ["metadata.json", "overhead.mp4", "trace.json", "trajectory.npz", "wrist.mp4"]
SIM_STATS = {"attempts": 0, "mirror_ok": 0, "mirror_diff": 0, "mirror_missing": 0}

def build_sim_episodes():
    base = A("data/sim/2026-11-14/dataset-150-raw")
    live = A("data/sim/live/2026-11-14/dataset-150-raw")
    for att in sorted(glob.glob(os.path.join(base, "attempt-*"))):
        a = os.path.basename(att)
        md = json.load(open(os.path.join(att, "metadata.json")))
        miss = [f for f in SIM_FILES if not os.path.exists(os.path.join(att, f))]
        ov = PROBE.get(rel(os.path.join(att, "overhead.mp4")), {})
        wr = PROBE.get(rel(os.path.join(att, "wrist.mp4")), {})
        shapes = npz_shapes(os.path.join(att, "trajectory.npz")) if os.path.exists(os.path.join(att, "trajectory.npz")) else {}
        diffs = []
        for f in SIM_FILES:
            p1, p2 = os.path.join(att, f), os.path.join(live, a, f)
            if not (os.path.exists(p1) and os.path.exists(p2)):
                diffs.append(f + "(missing)")
                SIM_STATS["mirror_missing"] += 1
            elif os.path.getsize(p1) != os.path.getsize(p2) or sha_cached(p1) != sha_cached(p2):
                diffs.append(f)
        SIM_STATS["attempts"] += 1
        SIM_STATS["mirror_ok" if not diffs else "mirror_diff"] += 1
        flags = []
        if md.get("success") is False:
            flags.append("FAILED")
        if md.get("error"):
            flags.append("SIM_ERROR")
        if miss:
            flags.append("MISSING:" + "+".join(miss))
        flags.append("MIRROR_OK" if not diffs else "MIRROR_DIFF:" + "+".join(diffs))
        if ov.get("n_packets") != wr.get("n_packets"):
            flags.append("FRAMES_MISMATCH")
        keys = [k for k in ("success", "task", "target_color", "destination") if md.get(k) is not None]
        fps = ov.get("implied_fps")
        size, n = walk_size(att)
        jtxt = ("trajectory.npz state %s action %s" % (shapes.get("state", "-"), shapes.get("action", "-"))) if shapes else "trajectory.npz missing"
        u = add("episode", "friend sim", rel(att), t0=None, t1=None,
                dur=ov.get("true_duration_s"), size=size, cams="overhead+wrist",
                frames=ov.get("n_packets", "-"), fps=(round(fps, 2) if fps else "-"),
                joint=jtxt, labels=("+".join(keys) if keys else "-"), flags=flags,
                notes=clean("seed %s; %s" % (md.get("seed"), md.get("task")))[:80])
        u["dsrc"] = "sim-raw-primary"

TWIN_RAW = {
    "V1": ("twin/work/TW-REC/V1/episodes", "V1"),
    "V2": ("twin/work/TW-REC/V2/data/episodes/raw", "V2"),
    "V3": ("twin/work/TW-REC/V3/raw", "V3"),
}
def build_twin_raw():
    for ver, (base, _) in TWIN_RAW.items():
        for ep in sorted(glob.glob(os.path.join(A(base), "*/"))):
            ep = ep.rstrip("/")
            pf = os.path.join(ep, "steps.parquet")
            if not os.path.exists(pf):
                continue
            pfile = pq.ParquetFile(pf)
            nrows = pfile.metadata.num_rows
            cols = pfile.schema_arrow.names
            lab = [c for c in cols if re.search(r"label|success|outcome|reward|task", c)]
            t0 = stamp_epoch(os.path.basename(ep))
            size, n = walk_size(ep)
            nimg = sum(1 for x in os.listdir(ep) if x.endswith((".jpg", ".png")))
            flags = ["FPS_ASSUMED30"]
            if not lab:
                flags.append("NO_LABEL_COLUMN")
            u = add("episode", "ours/twin-chat", rel(ep), t0=t0,
                    t1=(t0 + nrows / 30.0) if t0 else None, dur=nrows / 30.0, size=size,
                    cams="-", frames=nrows, fps=30.0,
                    joint="steps.parquet %d rows" % nrows, labels=("+".join(lab) if lab else "-"),
                    flags=flags, notes=clean("%s; %d image files; cols %d" % (ver, nimg, len(cols))))
            u["nimg"] = nimg
            u["dsrc"] = "twin-raw-" + ver

build_sort_episodes()
build_sim_episodes()
build_twin_raw()

def name_epoch(hhmmss):
    return datetime.datetime(2026, 11, 14, int(hhmmss[0:2]), int(hhmmss[2:4]), int(hhmmss[4:6]),
                             tzinfo=datetime.timezone.utc).timestamp()

def cam_key(k):
    return re.sub(r"observation\.images\.realsense (\d+)", r"rs\1", k).replace("observation.images.", "")

def dataset_row(dirpath, source):
    if source == "ours/twin-chat" and not os.path.isdir(dirpath):
        return None
    info = json.load(open(os.path.join(dirpath, "meta", "info.json")))
    fps, total_eps, total_frames = info.get("fps"), info.get("total_episodes"), info.get("total_frames")
    data_files = sorted(glob.glob(os.path.join(dirpath, "data", "**", "*.parquet"), recursive=True))
    ep_files = sorted(glob.glob(os.path.join(dirpath, "meta", "episodes", "**", "*.parquet"), recursive=True))
    data_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in data_files)
    ep_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in ep_files)
    tasks = pq.read_table(os.path.join(dirpath, "meta", "tasks.parquet")).to_pylist()
    cams = "+".join(cam_key(k) for k in info.get("features", {}) if k.startswith("observation.images."))
    vids = sorted(glob.glob(os.path.join(dirpath, "videos", "**", "*.mp4"), recursive=True))
    vframes = sum((PROBE.get(rel(v)) or {}).get("n_packets", 0) for v in vids)
    size, nfiles = walk_size(dirpath)
    flags = []
    if data_rows != total_frames:
        flags.append("FRAME_MISMATCH(parquet%d;info%s)" % (data_rows, total_frames))
    if ep_rows != total_eps:
        flags.append("EP_MISMATCH(parquet%d;info%s)" % (ep_rows, total_eps))
    dur = (total_frames / float(fps)) if (fps and total_frames) else None
    u = add("dataset", source, rel(dirpath), dur=dur, size=size, cams=cams or "-",
            frames=total_frames, fps=fps, joint="parquet action+state %sHz" % fps,
            labels="task-text(%d)" % len(tasks), flags=flags,
            notes=clean("episodes %s; tasks %s; videos %d (%d frames); files %d; codebase %s; robot %s" % (
                total_eps, info.get("total_tasks"), len(vids), vframes, nfiles,
                info.get("codebase_version"), info.get("robot_type"))))
    u["n_eps"] = total_eps
    u["dsrc"] = rel(dirpath)
    u["vids"] = [rel(v) for v in vids]
    return u

def zip_row(zp, folder, source):
    if source == "ours/twin-chat" and not os.path.isfile(zp):
        return None
    z = zipfile.ZipFile(zp)
    infos = [i for i in z.infolist() if not i.is_dir()]
    names = {i.filename: i.file_size for i in infos}
    ff = []
    for root, _, files in os.walk(folder):
        for f in files:
            fp = os.path.join(root, f)
            ff.append((os.path.relpath(fp, folder), os.path.getsize(fp)))
    matched = 0
    for r_, sz in ff:
        cand = [n for n in names if n == r_ or n.endswith("/" + r_)]
        if cand and names[cand[0]] == sz:
            matched += 1
    extra = len(names) - matched
    flags = ["ZIP_MATCH_FOLDER"] if (matched == len(ff) and extra == 0) else ["ZIP_DIFF(match%d/%d;extra%d)" % (matched, len(ff), extra)]
    u = add("dataset", source, rel(zp), size=os.path.getsize(zp), flags=flags,
            notes=clean("zip archive of %s; entries %d; folder files %d" % (rel(folder), len(infos), len(ff))))
    u["copy_of"] = rel(folder)
    u["unique"] = False
    return u

def manifest_check(root):
    man = json.load(open(os.path.join(A("data/sim/2026-11-14"), "SHA256.json")))
    ok, bad, missing, badl = 0, 0, 0, []
    for k, v in man.items():
        p = os.path.join(root, k)
        if not os.path.exists(p):
            missing += 1
        elif sha_cached(p) == v:
            ok += 1
        else:
            bad += 1
            badl.append(k)
    return {"entries": len(man), "ok": ok, "bad": bad, "missing": missing, "bad_first": badl[:5]}

MANIFEST = {}

def build_datasets():
    for uuid in sorted(os.listdir(A("data/studio-datasets"))):
        dataset_row(A("data/studio-datasets/" + uuid), "team/friend dataset")
    for n in ["MT-ALL-v2", "PP-BLUE", "PP-BLUE-v2", "PP-BOTH", "PP-GREEN", "PP-GREEN-v2", "S1_v0"]:
        dataset_row(A("data/curated/" + n), "team/friend dataset")
    dataset_row(A("data/sim/2026-11-14/extract-lerobot/dataset-150"), "friend sim")
    dataset_row(A("twin/work/TW-REC/V1/lerobot_studio"), "ours/twin-chat")
    dataset_row(A("twin/work/TW-REC/V2/export/studio"), "ours/twin-chat")
    dataset_row(A("twin/work/TW-REC/V3/export/studio_v3"), "ours/twin-chat")
    for zp, folder, src in [
        ("data/curated/MT-ALL-v2.zip", "data/curated/MT-ALL-v2", "team/friend dataset"),
        ("data/curated/PP-BLUE.zip", "data/curated/PP-BLUE", "team/friend dataset"),
        ("data/curated/PP-BLUE-v2.zip", "data/curated/PP-BLUE-v2", "team/friend dataset"),
        ("data/curated/PP-BOTH.zip", "data/curated/PP-BOTH", "team/friend dataset"),
        ("data/curated/PP-GREEN.zip", "data/curated/PP-GREEN", "team/friend dataset"),
        ("data/curated/PP-GREEN-v2.zip", "data/curated/PP-GREEN-v2", "team/friend dataset"),
        ("data/curated/S1_v0.zip", "data/curated/S1_v0", "team/friend dataset"),
        ("twin/work/TW-REC/V1/lerobot_studio_studio_import.zip", "twin/work/TW-REC/V1/lerobot_studio", "ours/twin-chat"),
        ("twin/work/TW-REC/V2/export/studio.zip", "twin/work/TW-REC/V2/export/studio", "ours/twin-chat"),
        ("twin/work/TW-REC/V3/export/studio_v3.zip", "twin/work/TW-REC/V3/export/studio_v3", "ours/twin-chat"),
    ]:
        zip_row(A(zp), A(folder), src)
    # friend sim: raw dataset (primary), live mirror, tar.gz copies
    base = A("data/sim/2026-11-14/dataset-150-raw")
    live = A("data/sim/live/2026-11-14/dataset-150-raw")
    MANIFEST["primary"] = manifest_check(A("data/sim/2026-11-14"))
    MANIFEST["live"] = manifest_check(A("data/sim/live/2026-11-14"))
    sz, nf = walk_size(base)
    u = add("dataset", "friend sim", rel(base), size=sz,
            frames=sum((PROBE.get(rel(os.path.join(a, "overhead.mp4"))) or {}).get("n_packets", 0)
                       for a in glob.glob(os.path.join(base, "attempt-*"))),
            fps=30, labels="success+task+target_color+destination",
            flags=["MANIFEST_OK%d/%d" % (MANIFEST["primary"]["ok"], MANIFEST["primary"]["entries"])],
            notes=clean("attempts %d; files %d; sidecars summary.json verification.json recording_context.json source/ (7 code files)" % (SIM_STATS["attempts"], nf)))
    u["dsrc"] = "sim-raw-primary"
    sz2, nf2 = walk_size(live)
    u2 = add("dataset", "friend sim", rel(live), size=sz2,
             frames=u["frames"], fps=30, labels="success+task+target_color+destination",
             flags=["MIRROR_OF_PRIMARY(ok%d;diff%d)" % (SIM_STATS["mirror_ok"], SIM_STATS["mirror_diff"]),
                    "MANIFEST_OK%d/%d" % (MANIFEST["live"]["ok"], MANIFEST["live"]["entries"])],
             notes=clean("live mirror, synced every 120 s (sync_sim.sh running); files %d" % nf2))
    u2["copy_of"] = rel(base)
    u2["unique"] = False
    for root_, tag in [(A("data/sim/2026-11-14"), "primary"), (A("data/sim/live/2026-11-14"), "live")]:
        p = os.path.join(root_, "dataset-150-lerobot.tar.gz")
        same = sha_cached(os.path.join(A("data/sim/2026-11-14"), "dataset-150-lerobot.tar.gz")) == sha_cached(os.path.join(A("data/sim/live/2026-11-14"), "dataset-150-lerobot.tar.gz"))
        fl = ["MANIFEST_OK" if (MANIFEST[tag]["ok"] and sha_cached(p) == json.load(open(os.path.join(A("data/sim/2026-11-14"), "SHA256.json")))["dataset-150-lerobot.tar.gz"]) else "MANIFEST_CHECK"]
        ta = add("dataset", "friend sim", rel(p), size=os.path.getsize(p), flags=fl + (["SHA_EQUAL_TO_PRIMARY"] if same else ["SHA_DIFF"]),
                 notes="LeRobot tar.gz of raw attempts (archive)")
        if tag == "live":
            ta["copy_of"] = rel(os.path.join(A("data/sim/2026-11-14"), "dataset-150-lerobot.tar.gz"))
            ta["unique"] = False
    # anomaly sets
    def tsv_stats(p):
        rows = list(csv.DictReader(open(p), delimiter="\t"))
        return len(rows), sum(1 for r_ in rows if (r_.get("label") or "").strip())
    tl = [tsv_stats(A("data/anomaly/real/label_sheets/labels_template.tsv")),
          tsv_stats(A("data/anomaly/real/label_sheets/held_labels_template.tsv"))]
    rs, rf = walk_size(A("data/anomaly/real"))
    n_rows = sum(t[0] for t in tl)
    n_fill = sum(t[1] for t in tl)
    dup = 0
    for p in [A("data/anomaly/real/held/episodes_held.csv"), A("data/anomaly/real/spot/episodes.csv")]:
        dup += sum(1 for r_ in csv.DictReader(open(p)) if "duplicate" in (r_.get("status") or ""))
    add("dataset", "team/friend dataset", "data/anomaly/real", size=rs,
        labels="defect-label(%d/%d filled)" % (n_fill, n_rows), flags=["DERIVED_FROM_FRIEND_DATASETS", "LABELS_BLANK" if n_fill < n_rows else "LABELS_FILLED"],
        notes=clean("held/spot crops and label sheets; files %d; duplicate-episode rows %d" % (rf, dup)))
    ss, sf = walk_size(A("data/anomaly/synthetic"))
    add("dataset", "ours", "data/anomaly/synthetic", size=ss,
        labels="good/defect (by folder)", notes=clean("synthetic train/test sets; files %d; manifest.json" % sf))

def img_row(folder, pattern, source, cams="-", labels="-", flags=None, notes="-", epoch_names=False, rec=False):
    if not os.path.isdir(folder):
        return None
    fl = [x for x in os.listdir(folder) if re.search(pattern, x) and os.path.isfile(os.path.join(folder, x))]
    if not fl:
        return None
    size = sum(os.path.getsize(os.path.join(folder, x)) for x in fl)
    t0 = t1 = None
    if epoch_names:
        ep = [int(re.search(r"_(\d{10})\.", x).group(1)) for x in fl if re.search(r"_(\d{10})\.", x)]
        if ep:
            t0, t1 = float(min(ep)), float(max(ep))
    mt = max(os.path.getmtime(os.path.join(folder, x)) for x in fl)
    fl2 = list(flags or [])
    if NOW - mt < 900:
        fl2.append("LIVE_MTIME<15min")
    return add("image_set", source, rel(folder), t0=t0, t1=t1, size=size, cams=cams, frames=len(fl),
               labels=labels, flags=fl2, notes=clean(notes), mtime=mt)

def build_image_sets():
    img_row(A("logs/monitor"), r"^estop_top_\d+\.jpg$", "ours", "top", epoch_names=True, notes="e-stop frames (name = epoch s)")
    img_row(A("logs/monitor"), r"^estop_close_\d+\.jpg$", "ours", "close", epoch_names=True, notes="e-stop frames (name = epoch s)")
    img_row(A("logs/monitor"), r"^estop_wrist_\d+\.jpg$", "ours", "wrist", epoch_names=True, notes="e-stop frames (name = epoch s)")
    img_row(A("logs/gm"), r"\.jpg$", "ours", "close+wrist", notes="gm stills and sheet")
    img_row(A("run"), r"^live_(top|close|wrist)\.jpg$", "ours", "top+close+wrist", flags=["LIVE_SNAPSHOT"], notes="overwritten by a running process")
    img_row(A("data/inspection/views"), r"\.jpg$", "ours", notes="inspection views")
    for sub in ("held/crops/gw", "held/crops/mat", "held/crops/raw", "spot/crops/gw", "spot/crops/mat", "spot/crops/raw"):
        img_row(A("data/anomaly/real/" + sub), r"\.png$", "team/friend dataset", labels="see anomaly/real",
                flags=["DERIVED_FROM_FRIEND_DATASETS"], notes="crops named <studio uuid8>_e<ep>_<serial>_<k>")
    img_row(A("data/anomaly/real/spot/overview"), r"\.jpg$", "team/friend dataset", flags=["DERIVED_FROM_FRIEND_DATASETS"], notes="pre/post overviews")
    img_row(A("data/anomaly/real/label_sheets"), r"\.jpg$", "ours", labels="sheets for hand labelling", notes="label sheets")
    img_row(A("data/anomaly/synthetic/train/good"), r"\.png$", "ours", notes="synthetic good")
    img_row(A("data/anomaly/synthetic/test/defect"), r"\.png$", "ours", notes="synthetic defect")
    img_row(A("data/anomaly/synthetic/test/good"), r"\.png$", "ours", notes="synthetic good")
    img_row(A("data/studio-datasets-report/V1/sheets"), r"\.png$", "ours", notes="report contact sheets")
    img_row(A("data/studio-datasets-report/V2"), r"\.png$", "ours", notes="report check images")
    for ep in sorted(glob.glob(A("twin/work/TW-REC/V*/episodes/*/")) + glob.glob(A("twin/work/TW-REC/V2/data/episodes/raw/*/")) + glob.glob(A("twin/work/TW-REC/V3/raw/*/"))):
        ep = ep.rstrip("/")
        if os.path.exists(os.path.join(ep, "steps.parquet")):
            img_row(ep, r"\.(jpg|png)$", "ours/twin-chat", flags=["TWIN_LIVE_SESSION?"] if False else [],
                    notes="raw episode frames: 2700 jpg + 900 png (3 cams x 900)")
    for sub in ("V1", "V3", "W1", "W2"):
        img_row(A("twin/work/TW-CAM/%s/results/stills" % sub), r"\.(jpg|png)$", "ours/twin-chat", notes="camera-test stills")
    img_row(A("twin/work/TW-PERC/V2/out"), r"\.png$", "ours/twin-chat", notes="perception result image")
    img_row(A("twin/work/ENV-FIX/F1"), r"\.png$", "ours/twin-chat", notes="env-fix result image")

def build_reports():
    for sub in ("live", "V1", "V2", "V3"):
        d = A("data/studio-datasets-report/" + sub)
        sz, n = walk_size(d)
        mt = max(os.path.getmtime(os.path.join(root, f)) for root, _, fs in os.walk(d) for f in fs)
        fl = ["LIVE_MTIME<15min"] if NOW - mt < 900 else []
        add("report", "ours", rel(d), size=sz, flags=fl, notes="%d files; studio-datasets report bundle" % n, mtime=mt)
    p = A("data/curated/episodes_v1.csv")
    add("report", "team/friend dataset", rel(p), size=os.path.getsize(p),
        notes="curated episode list; %d lines" % sum(1 for _ in open(p)))
    p = A("data/sim/gate/2026-11-14-95fa40dbf3.json")
    add("report", "friend sim", rel(p), size=os.path.getsize(p), notes="sim gate verdict json")

build_videos()
build_datasets()
build_image_sets()
build_reports()

def build_sessions():
    groups = {}
    for u in UNITS:
        if u["kind"] == "video" and u["path"].startswith("logs/monitor/"):
            key = re.search(r"(\d{8}T\d{6})Z", u["path"]).group(1)
            groups.setdefault(key, []).append(u)
    for key, us in sorted(groups.items()):
        t0 = min(x["t0"] for x in us if x["t0"] is not None)
        t1 = max(x["t1"] for x in us if x["t1"] is not None)
        add("capture_session", "ours", "logs/monitor [session %s]" % key, t0=t0, t1=t1,
            size=sum(x["size"] for x in us), cams="top+close", frames="-",
            notes=clean("%d MKV files; recorder stamp %s" % (len(us), key)))
    wr = sorted([u for u in UNITS if u["kind"] == "video" and u["path"].startswith("data/wrist/raw/")],
                key=lambda x: x["t0"] or 0)
    grp, cur = [], []
    for u in wr:
        if cur and (u["t0"] - cur[-1]["t1"]) > 120:
            grp.append(cur)
            cur = []
        cur.append(u)
    if cur:
        grp.append(cur)
    for i, g in enumerate(grp, 1):
        gaps = [b["t0"] - a["t1"] for a, b in zip(g, g[1:])]
        big = [x for x in gaps if x > 2.0]
        flags = ["GAPS_BETWEEN_SEGMENTS(n%d;max%.1fs)" % (len(big), max(big))] if big else []
        dur = sum((x["dur"] or 0) for x in g)
        add("capture_session", "ours", "data/wrist/raw [session %d %s]" % (i, hms(g[0]["t0"])),
            t0=g[0]["t0"], t1=g[-1]["t1"], size=sum(x["size"] for x in g), cams="wrist",
            frames=sum((x["frames"] or 0) for x in g), flags=flags,
            notes=clean("%d segments; %.1f min of video" % (len(g), dur / 60.0)))
    paths = ["logs/depth/g1_162903", "logs/depth/jprobe/164223"] + sorted(rel(p) for p in glob.glob(A("logs/depth/servo/*")) if os.path.isdir(p))
    for rp in paths:
        d = A(rp)
        size, nf = walk_size(d)
        files = [os.path.join(root, f) for root, _, fs in os.walk(d) for f in fs]
        ts0, ts1 = [], []
        for f in files:
            if f.endswith(".json"):
                try:
                    j = json.load(open(f))
                except Exception:
                    continue
                if isinstance(j, dict) and j.get("t_start_unix") is not None:
                    ts0.append(float(j["t_start_unix"]))
                if isinstance(j, dict) and j.get("t_end_unix") is not None:
                    ts1.append(float(j["t_end_unix"]))
        flags = []
        if ts0:
            t0, t1 = min(ts0), (max(ts1) if ts1 else None)
        else:
            m = re.search(r"(?:_|/)(\d{6})(?:_|$)", rp)
            t0 = name_epoch(m.group(1)) if m else None
            t1 = None
            flags.append("NAME_TIME")
        names = [os.path.basename(f) for f in files]
        cams = "+".join(c for c in ("top", "close", "wrist") if any(c in n for n in names))
        cnt = lambda ext: sum(1 for n in names if n.endswith(ext))
        add("capture_session", "ours", rp, t0=t0, t1=t1, size=size, cams=cams or "-", flags=flags,
            notes=clean("depth capture; npy %d; jpg %d; png %d; json %d; files %d" % (cnt(".npy"), cnt(".jpg"), cnt(".png"), cnt(".json"), nf)),
            mtime=(max(os.path.getmtime(f) for f in files) if files else None))
    loose = [p for p in glob.glob(A("logs/depth/servo/*")) if os.path.isfile(p)]
    if loose:
        lnames = sorted(os.path.basename(p) for p in loose)
        add("image_set", "ours", rel(A("logs/depth/servo")) + " [loose files]", size=sum(os.path.getsize(p) for p in loose),
            flags=["NO_TIMESTAMP"],
            notes=clean("depth servo root loose files: %d (jpg %d, tsv %d); names: %s ..." % (len(loose), sum(1 for n in lnames if n.endswith(".jpg")), sum(1 for n in lnames if n.endswith(".tsv")), ";".join(lnames[:5]))))
    for d in sorted(glob.glob(A("logs/ovrun/*"))):
        if os.path.islink(d) or not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        t0 = datetime.datetime.strptime(name, "%Y%m%d-%H%M%S").replace(tzinfo=datetime.timezone.utc).timestamp()
        size, n = walk_size(d)
        add("capture_session", "ours", rel(d), t0=t0, size=size, flags=["NAME_TIME"],
            notes=clean("ovrun csv+jsonl; %d files" % n))
    def logs_only(rp, note, flags=None, t0=None):
        d = A(rp)
        if os.path.isdir(d):
            size, n = walk_size(d)
            mt = max(os.path.getmtime(os.path.join(r_, f)) for r_, _, fs in os.walk(d) for f in fs)
        else:
            size, n = os.path.getsize(d), 1
            mt = os.path.getmtime(d)
        fl = list(flags or [])
        if NOW - mt < 900:
            fl.append("LIVE_MTIME<15min")
        add("capture_session", "ours", rp, t0=t0, size=size, flags=fl,
            notes=clean("%s; %d files" % (note, n)), mtime=mt)
    logs_only("logs/telemetry", "telemetry jsonl and sampler log")
    logs_only("logs/runs", "analysis-run logs (no sensor data)", ["LOG_ONLY"])
    logs_only("logs/runs-attempt1", "analysis-run logs attempt1 (no sensor data)", ["LOG_ONLY"])
    logs_only("logs/attempt2", "analysis-run logs attempt2 incl. model exports (no sensor data)", ["LOG_ONLY"])
    logs_only("logs/P2-D-SIM-E1", "sim gate logs", ["LOG_ONLY"])
    logs_only("run/mserver", "motion server logs and state")
    logs_only("logs/detlive", "detection live scans (jpg, json)")
    logs_only("logs/gm", "gm session: 2 MKV, jpg stills, logs")
    logs_only("logs/grasp_auto/calib_171531", "grasp calibration stills", ["NAME_TIME"], t0=name_epoch("171531"))
    mon = [A("logs/monitor/" + x) for x in ("hand_watch.log", "hand_watch.out", "ffmpeg_top.log", "ffmpeg_close.log")]
    empty = sum(1 for x in mon if os.path.getsize(x) == 0)
    add("capture_session", "ours", "logs/monitor [hand_watch + ffmpeg logs]", size=sum(os.path.getsize(x) for x in mon),
        flags=(["EMPTY_FILES%d" % empty] if empty else []), notes="hand-watch log and out; ffmpeg logs",
        mtime=max(os.path.getmtime(x) for x in mon))
    top = [x for x in os.listdir(A("logs")) if os.path.isfile(A("logs/" + x))]
    add("capture_session", "ours", "logs/ [top-level misc logs]", size=sum(os.path.getsize(A("logs/" + x)) for x in top),
        notes=clean("%d top-level log files" % len(top)))
    for ver, base in [("V1", "twin/work/TW-REC/V1"), ("V2", "twin/work/TW-REC/V2"), ("V3", "twin/work/TW-REC/V3")]:
        size, n = walk_size(A(base))
        fl = ["INCLUDES_HF_CACHE"] if os.path.isdir(A(base + "/cache/hf")) or os.path.isdir(A(base + "/hf_home")) else []
        stamp = {"V1": "20261114T175023794356Z", "V2": "20261114T174102Z", "V3": "20261114T173453Z"}[ver]
        add("capture_session", "ours/twin-chat", base, t0=stamp_epoch(stamp), size=size, flags=fl,
            notes=clean("%s recording: raw episode + LeRobot export + zip (%d files incl. caches)" % (ver, n)))
    add("dataset", "ours/twin-chat", "twin/data (+ twin/logs)", size=0, flags=["EMPTY_PLACEHOLDER_DIRS"],
        notes="0 files at census time; dirs exist: calib, model, captures, episodes, episodes/raw, episodes/lerobot; twin/logs empty")

build_sessions()

def link_copies():
    studio = [u for u in UNITS if u["kind"] == "video" and u["path"].startswith("data/studio-datasets/")]
    by_size = {}
    for s in studio:
        by_size.setdefault(s["size"], []).append(s)
    for c in [u for u in UNITS if u["kind"] == "video" and u["path"].startswith("data/curated/")]:
        for s in by_size.get(c["size"], []):
            if sha_cached(A(c["path"])) == sha_cached(A(s["path"])):
                c["copy_of"] = s["path"]
                c["unique"] = False
                c["flags"].append("COPY_OF:" + s["path"].split("/")[2][:8])
                break
    for d in [u for u in UNITS if u["kind"] == "dataset" and u["path"].startswith("data/curated/") and u.get("vids") is not None]:
        vids = [u for u in UNITS if u["kind"] == "video" and u["dsrc"] == d["path"]]
        m = [v for v in vids if v["copy_of"]]
        d["flags"].append("VIDEO_COPY_OF_STUDIO(%d/%d)" % (len(m), len(vids)))
        if vids and len(m) == len(vids):
            d["unique"] = False
            d["copy_of"] = "studio videos (sha256 identical)"

link_copies()

# ---- twin scope: R/twin was deleted at 19:48Z. Fresh twin rows are dropped (the folder is gone); the
# pre-deletion rows from the 19:47:59Z snapshot are restored and marked deleted_19:48Z (outside unique totals).
TWIN_SNAP = os.path.join(RAW, "unit_detail_run3_prededeletion.jsonl")
TWIN_GONE = not os.path.exists(A("twin"))
if TWIN_GONE:
    UNITS[:] = [u for u in UNITS if u["source"] != "ours/twin-chat"]
    with open(TWIN_SNAP) as _f:
        for _line in _f:
            _d = json.loads(_line)
            if _d["source"] != "ours/twin-chat":
                continue
            _d["links"] = []
            _d["id"] = None
            _d["flags"] = list(_d["flags"]) + ["deleted_19:48Z"]
            _d["unique"] = False
            _note = "" if _d["notes"] in (None, "-") else str(_d["notes"]) + "; "
            _d["notes"] = _note + "R/twin deleted 19:48Z; pre-deletion facts from the 19:47:59Z snapshot; record R/work/deleted/twin_manifest_20261114T194800Z.txt"
            _u = add(_d["kind"], _d["source"], _d["path"])
            _u.update(_d)

# ---- sim manifest and status files (friend's sim tree, data only): catalogued as report rows
for _p in ["data/sim/2026-11-14/SHA256.json", "data/sim/2026-11-14/TRAINING-STATUS.md", "data/sim/live/2026-11-14/SHA256.json", "data/sim/live/2026-11-14/TRAINING-STATUS.md"]:
    if os.path.isfile(A(_p)):
        add("report", "friend sim", _p, size=os.path.getsize(A(_p)), mtime=os.path.getmtime(A(_p)),
            flags=(["LIVE_MIRROR"] if "/live/" in _p else []),
            notes=clean("sim %s (friend's; data only)%s" % ("hash manifest" if _p.endswith("SHA256.json") else "training status", "; live copy changes with sync_sim.sh every 120 s" if "/live/" in _p else "")))

KORDER = {"episode": 0, "video": 1, "capture_session": 2, "dataset": 3, "image_set": 4, "report": 5}
PFX = {"episode": "E", "video": "V", "capture_session": "C", "dataset": "D", "image_set": "I", "report": "R"}
UNITS.sort(key=lambda u: (KORDER[u["kind"]], u["path"]))
cnt = {}
for u in UNITS:
    cnt[u["kind"]] = cnt.get(u["kind"], 0) + 1
    u["id"] = "%s%03d" % (PFX[u["kind"]], cnt[u["kind"]])
BY_PATH = {u["path"]: u for u in UNITS}
D_BY_DSRC = {u["dsrc"] if u.get("dsrc") else u["path"]: u for u in UNITS if u["kind"] == "dataset"}

def overlap(a0, a1, b0, b1):
    if a0 is None or b0 is None:
        return False
    a1 = a1 if a1 is not None else a0 + 60
    b1 = b1 if b1 is not None else b0 + 60
    return a0 <= b1 and b0 <= a1

EPS = [u for u in UNITS if u["kind"] == "episode" and u["source"] == "ours"]
VMON = [u for u in UNITS if u["kind"] == "video" and (u["path"].startswith("logs/monitor/") or u["path"].startswith("data/wrist/raw/"))]
for e in EPS:
    e["links"] = [v["id"] for v in VMON if overlap(e["t0"], e["t1"], v["t0"], v["t1"])]
for v in VMON:
    v["links"] = [e["id"] for e in EPS if overlap(e["t0"], e["t1"], v["t0"], v["t1"])]
for u in UNITS:
    if u["kind"] in ("image_set", "capture_session") and u["t0"] is not None and u["path"].split(" ")[0] in ("logs/depth/g1_162903", "logs/depth/jprobe/164223") or (u["kind"] == "capture_session" and u["path"].startswith("logs/depth/servo/")) or (u["kind"] == "image_set" and u["path"].startswith("logs/monitor")):
        u["links"] = [e["id"] for e in EPS if overlap(e["t0"], e["t1"], u["t0"], u["t1"])]
for u in UNITS:
    if u["kind"] == "video" and u["dsrc"] and u["dsrc"] in D_BY_DSRC:
        u["links"] = [D_BY_DSRC[u["dsrc"]]["id"]]
    if u["kind"] == "episode" and u.get("dsrc") == "sim-raw-primary":
        u["links"] = [D_BY_DSRC["sim-raw-primary"]["id"]] if "sim-raw-primary" in D_BY_DSRC else []
for u in UNITS:
    if u["kind"] == "video" and u.get("copy_of") is None and u["path"].startswith("data/curated/") and False:
        pass

# ---- fix-ups: distinct paths for set rows that share a folder; dataset keys for videos
for u in UNITS:
    if u["source"] == "ours/twin-chat":
        u["links"] = []
for u in UNITS:
    if u["kind"] == "video" and "/videos/" in u["path"]:
        u["dsrc"] = u["path"].split("/videos/")[0]
    if u["kind"] == "image_set" and u["path"] == "logs/monitor":
        u["path"] = "logs/monitor [estop_%s]" % u["cams"]
    if u["kind"] == "image_set" and u["path"].startswith("twin/work/TW-REC/") and not u["path"].endswith(" [frames]"):
        u["path"] += " [frames]"

def is_unique(u):
    return bool(u.get("unique", True))

def hour_of(t):
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).hour

T = {"snapshot_utc": SNAP}
T["rows_by_kind_source"] = {}
for u in UNITS:
    k = u["kind"] + " | " + u["source"]
    d = T["rows_by_kind_source"].setdefault(k, {"rows": 0, "unique_rows": 0, "bytes": 0, "unique_bytes": 0})
    d["rows"] += 1
    d["bytes"] += u["size"] or 0
    if is_unique(u):
        d["unique_rows"] += 1
        d["unique_bytes"] += u["size"] or 0
T["video_unique"] = {}
for u in UNITS:
    if u["kind"] == "video" and is_unique(u):
        d = T["video_unique"].setdefault(u["source"], {"files": 0, "hours": 0.0, "frames": 0, "gb": 0.0})
        d["files"] += 1
        d["hours"] += (u["dur"] or 0) / 3600.0
        d["frames"] += int(u["frames"] or 0)
        d["gb"] += (u["size"] or 0) / 1e9
T["episodes_unique"] = {}
for u in UNITS:
    if u["kind"] == "episode" and is_unique(u):
        T["episodes_unique"][u["source"]] = T["episodes_unique"].get(u["source"], 0) + 1
T["lerobot_datasets_unique"] = {}
for u in UNITS:
    if u["kind"] == "dataset" and is_unique(u) and u.get("n_eps") is not None:
        d = T["lerobot_datasets_unique"].setdefault(u["source"], {"datasets": 0, "episodes": 0, "frames": 0})
        d["datasets"] += 1
        d["episodes"] += int(u["n_eps"])
        d["frames"] += int(u["frames"] or 0)
T["sim_raw_attempt_frames"] = sum(int(u["frames"]) for u in UNITS if u["kind"] == "episode" and u["source"] == "friend sim" and str(u["frames"]).isdigit())
T["sim_mirror"] = dict(SIM_STATS)
T["manifest"] = MANIFEST
MAN_TWIN = A("work/deleted/twin_manifest_20261114T194800Z.txt")
T["twin_scope"] = {"twin_dir_exists_at_census": os.path.exists(A("twin")),
    "rows_marked_deleted_19_48Z": sum(1 for u in UNITS if "deleted_19:48Z" in u["flags"]),
    "deletion_manifest": rel(MAN_TWIN), "manifest_exists": os.path.exists(MAN_TWIN),
    "manifest_bytes": os.path.getsize(MAN_TWIN) if os.path.exists(MAN_TWIN) else None,
    "manifest_sha256": sha256(MAN_TWIN) if os.path.exists(MAN_TWIN) else None}

hours = {}
def hbump(t, key):
    if t is None:
        return
    h = hour_of(t)
    hours.setdefault(h, {})
    hours[h][key] = hours[h].get(key, 0) + 1
for e in EPS:
    hbump(e["t0"], "sort_episodes_started")
for u in UNITS:
    if u["kind"] == "video" and u["path"].startswith("data/wrist/raw/"):
        hbump(u["t0"], "wrist_segments_started")
    if u["kind"] == "video" and u["path"].startswith("logs/monitor/"):
        hbump(u["t0"], "monitor_mkv_started")
    if u["kind"] == "capture_session" and u["path"].startswith("logs/depth/"):
        hbump(u["t0"], "depth_sessions_started")
    if u["kind"] == "capture_session" and u["path"].startswith("logs/ovrun/"):
        hbump(u["t0"], "ovrun_sessions_started")
for f in glob.glob(A("logs/monitor/estop_*.jpg")):
    m = re.search(r"_(\d{10})\.jpg$", f)
    if m:
        hbump(float(m.group(1)), "estop_jpgs")
T["timeline_utc_hours"] = {str(h): hours[h] for h in sorted(hours)}

T["gaps_and_ends"] = {
    "sort_run_spans": {run: [hms(min(e["t0"] for e in EPS if e["run"] == run and e["t0"] is not None)), hms(max(e["t1"] for e in EPS if e["run"] == run and e["t1"] is not None))] for run in ("any_1731", "any_1740")},
    "wrist_sessions": [[u["path"], u["t0"] and hms(u["t0"]), u["t1"] and hms(u["t1"]), u["notes"], ";".join(u["flags"])] for u in UNITS if u["kind"] == "capture_session" and u["path"].startswith("data/wrist/raw [session")],
    "monitor_sessions": [[u["path"], hms(u["t0"]), hms(u["t1"]), u["notes"]] for u in UNITS if u["kind"] == "capture_session" and u["path"].startswith("logs/monitor [session")],
    "last_sensor_end_utc": hms(max(u["t1"] for u in UNITS if u["kind"] in ("video", "episode") and u["source"] == "ours" and u["t1"])),
}

need = {}
lerobot_unique = [u for u in UNITS if u["kind"] == "dataset" and is_unique(u) and u.get("n_eps") is not None]
need["T1_episode_outcome_label"] = {
    "sort_episodes_without_success_key": sum(1 for e in EPS if "success" not in e["labels"]),
    "lerobot_episodes_without_outcome": sum(int(u["n_eps"]) for u in lerobot_unique),
    "twin_raw_episodes_without_outcome": sum(1 for u in UNITS if u["kind"] == "episode" and str(u.get("dsrc", "")).startswith("twin-raw") and is_unique(u) and "success" not in u["labels"]),
}
need["T2_target_colour_plate_per_episode"] = {
    "sort_episodes_without_colour_or_plate": sum(1 for e in EPS if ("colour" not in e["labels"]) or ("plate" not in e["labels"])),
    "lerobot_episodes_colour_only_in_task_text": sum(int(u["n_eps"]) for u in lerobot_unique),
    "twin_raw_episodes_without_colour_label": sum(1 for u in UNITS if u["kind"] == "episode" and str(u.get("dsrc", "")).startswith("twin-raw") and is_unique(u)),
}
vu = [u for u in UNITS if u["kind"] == "video" and is_unique(u)]
need["T3_frame_boxes_or_masks_per_video"] = {"unique_video_files": len(vu), "hours": round(sum((u["dur"] or 0) for u in vu) / 3600.0, 2)}
anom = [u for u in UNITS if u["kind"] == "dataset" and u["path"] == "data/anomaly/real"]
if anom:
    mm = re.search(r"\((\d+)/(\d+) filled\)", anom[0]["labels"])
    if mm:
        need["T4_anomaly_defect_label_per_crop"] = {"rows": int(mm.group(2)), "blank": int(mm.group(2)) - int(mm.group(1))}
need["T5_depth_height_ground_truth_per_session"] = {"depth_capture_sessions": sum(1 for u in UNITS if u["kind"] == "capture_session" and u["path"].startswith("logs/depth/"))}
T["needs_labelling"] = need

qc = {}
for u in UNITS:
    for f in u["flags"]:
        key = re.split(r"[(:\d<]", f)[0]
        qc[key] = qc.get(key, 0) + 1
T["flag_counts"] = qc
T["video_pts"] = {
    "PTS_ABS": sum(1 for u in UNITS if u["kind"] == "video" and "PTS_ABS" in u["flags"]),
    "PTS_REL": sum(1 for u in UNITS if u["kind"] == "video" and "PTS_REL" in u["flags"]),
    "GAPS_units": sum(1 for u in UNITS if u["kind"] == "video" and any(f.startswith("GAPS") for f in u["flags"])),
    "CORRUPT_units": sum(1 for u in UNITS if "CORRUPT" in u["flags"]),
    "copies_flagged_videos": sum(1 for u in UNITS if u["kind"] == "video" and u.get("copy_of")),
}
T["scope_files_logs_data_run"] = {"files": sum(len(fs) for d in ("logs", "data", "run") for _, _, fs in os.walk(A(d))),
                                  "bytes": sum(walk_size(A(d))[0] for d in ("logs", "data", "run"))}

# ---- writers
COLS = ["id", "kind", "source", "path", "utc_start", "utc_end", "duration_s", "size_MB", "cameras", "frames",
        "fps", "joint_log", "labels", "linked", "flags", "notes"]

def fmt_f(x, nd=2):
    return ("%." + str(nd) + "f") % x

def row_of(u):
    fr = u["frames"]
    if isinstance(fr, float):
        fr = "%d" % fr
    fps = u["fps"]
    if isinstance(fps, (int, float)) and not isinstance(fps, bool):
        fps = fmt_f(float(fps))
    return [u["id"], u["kind"], u["source"], u["path"], hms(u["t0"]), hms(u["t1"]),
            fmt_f(u["dur"]) if u["dur"] is not None else "-",
            ("%.2f" % (u["size"] / 1e6)) if u["size"] is not None else "-",
            u["cams"] or "-", str(fr) if fr is not None else "-", fps if fps is not None else "-",
            u["joint"] or "-", u["labels"] or "-", ";".join(u["links"]) if u["links"] else "-",
            ";".join(u["flags"]) if u["flags"] else "-", clean(u["notes"] or "-")]

rows = [row_of(u) for u in UNITS]
with open(os.path.join(E1, "catalogue.csv"), "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(COLS)
    w.writerows(rows)
with open(os.path.join(E1, "catalogue.json"), "w") as f:
    json.dump({"snapshot_utc": SNAP, "day_utc": "2026-11-14", "columns": COLS,
               "units": [dict(zip(COLS, r)) for r in rows]}, f, indent=1, sort_keys=False)
with open(os.path.join(RAW, "unit_detail.jsonl"), "w") as f:
    for u in UNITS:
        f.write(json.dumps({k: v for k, v in u.items()}, default=str) + "\n")
with open(os.path.join(RAW, "census_tables.json"), "w") as f:
    json.dump(T, f, indent=1, default=str)
print("units", len(UNITS))
print("kinds", {k: sum(1 for u in UNITS if u["kind"] == k) for k in KORDER})
print("csv bytes", os.path.getsize(os.path.join(E1, "catalogue.csv")))
print("sha256 catalogue.csv", sha256(os.path.join(E1, "catalogue.csv")))
