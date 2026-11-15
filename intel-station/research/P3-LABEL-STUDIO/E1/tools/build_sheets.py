#!/usr/bin/env python3
"""Group duplicate episodes, hash the mp4 segments, extract frames and build one contact sheet per unique episode.

Reads out/episodes_census.jsonl. Writes out/unique_episodes.json, frames/*.png, sheets/U*.png.
"""
import collections
import hashlib
import json
import os
import subprocess

from PIL import Image, ImageDraw

R = "/home/intel/intel-junction-2026"
SD = os.path.join(R, "data", "studio-datasets")
WD = os.path.join(R, "work", "P3-LABEL-STUDIO", "E1")
OUTD = os.path.join(WD, "out")
FRD = os.path.join(WD, "frames")
SHD = os.path.join(WD, "sheets")
CAMS = ["observation.images.realsense 243622061421", "observation.images.realsense 243722064316"]
FPS = 30
TW, TH = 320, 240
for p in (FRD, SHD):
    os.makedirs(p, exist_ok=True)


def sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def moments(e):
    n = e["n_frames"]
    ev = e.get("grip_events") or []
    lo = [i for i, s in ev if s == "to_lo"]
    hi = [i for i, s in ev if s == "to_hi"]
    if lo:
        g = lo[0]
        after = [i for i in hi if i > g]
        rel = after[-1] if after else n - 1
        return [("start", 0), ("pregrasp", max(0, g - 45)), ("grasp", g),
                ("carry", (g + rel) // 2), ("release", rel), ("end", n - 1)]
    return [("start", 0), ("p20", n // 5), ("p40", 2 * n // 5), ("p60", 3 * n // 5),
            ("p80", 4 * n // 5), ("end", n - 1)]


def extract(mp4, t, out):
    if not os.path.exists(out):
        subprocess.run(["ffmpeg", "-v", "error", "-ss", "%.4f" % t, "-i", mp4,
                        "-frames:v", "1", "-y", out], capture_output=True, text=True)
    return os.path.exists(out)


def main():
    eps = [json.loads(line) for line in open(os.path.join(OUTD, "episodes_census.jsonl"))]
    groups = collections.OrderedDict()
    for e in eps:
        key = e.get("sig") or "nosig_%s_%d" % (e["dataset"], e["episode_index"])
        groups.setdefault(key, []).append(e)
    sha_cache = {}

    def sha_of(path):
        if path not in sha_cache:
            sha_cache[path] = sha1_file(path) if os.path.exists(path) else None
        return sha_cache[path]

    unique = []
    for gi, (key, members) in enumerate(groups.items(), 1):
        uid = "U%02d" % gi
        rep = members[0]
        mom = moments(rep)
        cam_info = []
        for k in CAMS:
            seg = rep["video_segments"].get(k)
            if not seg:
                cam_info.append(None)
                continue
            mp4 = os.path.join(SD, rep["dataset"], seg["mp4"])
            cam_info.append({"cam": k[-6:], "mp4": mp4, "from_ts": seg["from_ts"],
                             "to_ts": seg["to_ts"], "exists": seg["exists"],
                             "sha1": sha_of(mp4)})
        mem_list = []
        for m in members:
            same = []
            for k, ci in zip(CAMS, cam_info):
                if ci is None or not m["video_segments"].get(k):
                    same.append(None)
                    continue
                mp = os.path.join(SD, m["dataset"], m["video_segments"][k]["mp4"])
                same.append(sha_of(mp) == ci["sha1"])
            mem_list.append({"ds": m["dataset"][:8], "ep": m["episode_index"], "rows": m["data_rows"],
                             "same_video_as_rep": same, "tasks": m.get("tasks"),
                             "flags": m.get("flags")})
        frames_ok = True
        for ci in cam_info:
            if ci is None:
                frames_ok = False
                continue
            for label, idx in mom:
                t = ci["from_ts"] + idx / FPS
                out = os.path.join(FRD, "%s_%s_%s_f%04d.png" % (uid, ci["cam"], label, idx))
                frames_ok = extract(ci["mp4"], t, out) and frames_ok
        W = len(mom) * (TW + 10) + 10
        H = 36 + len(CAMS) * (TH + 26) + 4
        img = Image.new("RGB", (W, H), (255, 255, 255))
        dr = ImageDraw.Draw(img)
        title = "%s n=%d | %s | task: %s" % (
            uid, rep["n_frames"], " ".join("%s:%d" % (x["ds"], x["ep"]) for x in mem_list),
            "; ".join(rep.get("tasks") or []))
        dr.text((10, 8), title, fill=(0, 0, 0))
        for ci_i, ci in enumerate(cam_info):
            if ci is None:
                continue
            for mi, (label, idx) in enumerate(mom):
                x = 10 + mi * (TW + 10)
                y = 36 + ci_i * (TH + 26)
                fp = os.path.join(FRD, "%s_%s_%s_f%04d.png" % (uid, ci["cam"], label, idx))
                if os.path.exists(fp):
                    im = Image.open(fp).convert("RGB").resize((TW, TH))
                    img.paste(im, (x, y + 18))
                dr.text((x, y + 2), "cam %s %s f%d %.2fs" % (ci["cam"], label, idx, idx / FPS),
                        fill=(0, 0, 0))
        img.save(os.path.join(SHD, uid + ".png"))
        unique.append({"uid": uid, "n_frames": rep["n_frames"], "dur_s": rep["dur_s"],
                       "tasks": rep.get("tasks"), "grip_events_s": rep.get("grip_events_s"),
                       "lag_best": rep.get("lag_best"), "mae_lag_best": rep.get("mae_lag_best"),
                       "members": mem_list, "cams": cam_info, "moments": mom,
                       "frames_ok": frames_ok, "sheet": "sheets/%s.png" % uid})
    with open(os.path.join(OUTD, "unique_episodes.json"), "w") as fh:
        json.dump(unique, fh, indent=1)
    print("episodes=%d unique=%d" % (len(eps), len(unique)))
    for u in unique:
        print("%s n=%d dur=%s tasks=%s grip=%s lag=%s/%s frames_ok=%s" % (
            u["uid"], u["n_frames"], u["dur_s"], json.dumps(u["tasks"]), json.dumps(u["grip_events_s"]),
            u["lag_best"], u["mae_lag_best"], u["frames_ok"]))
        print("   members:", json.dumps([[m["ds"], m["ep"], m["rows"], m["same_video_as_rep"]] for m in u["members"]]))
        print("   cams:", json.dumps([None if c is None else [c["cam"], c["from_ts"], c["to_ts"], c["sha1"]]
                                      for c in u["cams"]]))


if __name__ == "__main__":
    main()
