#!/usr/bin/env python3
"""Read-only inventory of one sim version root (data only, never executed).
Usage: python3 -I inventory.py ROOT OUT.json"""
import collections, glob, hashlib, json, os, subprocess, sys
import numpy as np

ROOT, OUT = sys.argv[1].rstrip("/"), sys.argv[2]
FILES = ("metadata.json", "trajectory.npz", "trace.json", "overhead.mp4", "wrist.mp4")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def probe(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=codec_name,width,height,r_frame_rate,nb_frames,pix_fmt:format=duration,start_time",
                        "-of", "json", p], capture_output=True, text=True)
    try:
        j = json.loads(r.stdout)
        s = j["streams"][0]
        f = j.get("format", {})
        return {"codec": s.get("codec_name"), "w": s.get("width"), "h": s.get("height"), "fps": s.get("r_frame_rate"),
                "nb": s.get("nb_frames"), "pix": s.get("pix_fmt"), "dur": f.get("duration"), "start": f.get("start_time")}
    except Exception as ex:
        return {"error": str(ex)[:120]}


res = {"root": ROOT, "top_level": sorted(os.listdir(ROOT))}
tar = os.path.join(ROOT, "dataset-150-lerobot.tar.gz")
res["tar_sha256"] = sha(tar) if os.path.isfile(tar) else None
res["manifest_present"] = os.path.isfile(os.path.join(ROOT, "SHA256.json"))
raws = sorted(glob.glob(ROOT + "/dataset-*-raw"))
raw = raws[0] if raws else None
res["raw_dir"] = raw
att = sorted(glob.glob(raw + "/attempt-*")) if raw else []
res["attempts"] = len(att)
res["raw_top_entries"] = sorted(x for x in os.listdir(raw) if not x.startswith("attempt-")) if raw else []
res["attempt_file_sets"] = {str(k): v for k, v in collections.Counter(
    tuple(sorted(os.listdir(d))) for d in att).items()}

per, mdkeys, vals = [], collections.Counter(), collections.defaultdict(collections.Counter)
for d in att:
    e = {"name": os.path.basename(d), "sha": {}, "size": {}}
    for f in FILES:
        p = os.path.join(d, f)
        if os.path.isfile(p):
            e["sha"][f] = sha(p)
            e["size"][f] = os.path.getsize(p)
    md = json.load(open(os.path.join(d, "metadata.json")))
    e["md"] = md
    for k, v in md.items():
        mdkeys[k] += 1
        vals[k][json.dumps(v)[:120] if isinstance(v, (str, bool, int, float)) or v is None else "<%s>" % type(v).__name__] += 1
    e["probe"] = {"overhead": probe(os.path.join(d, "overhead.mp4")), "wrist": probe(os.path.join(d, "wrist.mp4"))}
    per.append(e)
res["attempts_detail"] = per
res["md_keys"] = dict(mdkeys)
res["md_scalar_values"] = {k: dict(c.most_common(15)) for k, c in vals.items()}
res["probe_summary"] = {cam: dict(collections.Counter(json.dumps({k: x["probe"][cam].get(k) for k in ("codec", "w", "h", "fps", "pix")}, sort_keys=True) for x in per).most_common(6)) for cam in ("overhead", "wrist")}

if att:
    z = np.load(os.path.join(att[0], "trajectory.npz"))
    res["npz_keys_attempt0"] = {k: [list(z[k].shape), str(z[k].dtype)] for k in z.files}
    tr = json.load(open(os.path.join(att[0], "trace.json")))
    res["trace_attempt0"] = {"type": type(tr).__name__, "len": len(tr),
                             "first_keys": sorted(tr[0].keys()) if isinstance(tr, list) and tr and isinstance(tr[0], dict) else None}

rc = os.path.join(raw, "recording_context.json") if raw else None
res["recording_context"] = json.load(open(rc)) if rc and os.path.isfile(rc) else None
src = os.path.join(raw, "source") if raw else None
res["source_files"] = {f: [os.path.getsize(os.path.join(src, f)), sha(os.path.join(src, f))]
                       for f in sorted(os.listdir(src))} if src and os.path.isdir(src) else None
others = []
for dp, dn, fn in os.walk(ROOT):
    if raw and (dp == raw or dp.startswith(raw + "/")):
        continue
    for f in fn:
        p = os.path.join(dp, f)
        others.append([os.path.relpath(p, ROOT), os.path.getsize(p)])
res["files_outside_raw"] = sorted(others)[:300]
with open(OUT, "w") as fh:
    json.dump(res, fh, indent=1, default=str)
print(json.dumps({"root": ROOT, "attempts": len(att), "manifest": res["manifest_present"],
                  "tar_sha256": res["tar_sha256"], "file_sets": res["attempt_file_sets"],
                  "md_keys": res["md_keys"], "probe": res["probe_summary"], "npz": res.get("npz_keys_attempt0"),
                  "trace": res.get("trace_attempt0"), "outside_raw": res["files_outside_raw"][:12]}, indent=1, default=str)[:6000])
