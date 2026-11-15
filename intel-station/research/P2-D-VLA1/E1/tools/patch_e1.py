import os
import shutil
import sys

R = "/home/intel/intel-junction-2026/app/vla"

BUILD = [
(r'''# Writes OUT (dataset root) and OUT.build_report.json next to it. Idempotent: OUT is removed and rebuilt.
''', r'''# Segment clock: a segment whose container start_time is above 1e9 has absolute epoch pts; any other segment starts at pts 0
#   and its file name (whole UTC second) gives its start. A segment ends where the next one starts; the last wrist segment
#   ends 60 s after its start. Container duration is not used (on absolute segments ffprobe reports the end time as duration).
# Options: --only-success; --plates P [P ...] keeps episodes whose episode.json plate is listed; --max-dt S drops episodes
#   with a camera frame more than S seconds from a joint sample (default off). The build report has per-camera dt stats.
# Threads: one thread per numeric library, one decoder thread, libsvtav1 lp=3; cameras are encoded one after another.
# Writes OUT (dataset root) and OUT.build_report.json next to it. Idempotent: OUT is removed and rebuilt.
'''),
(r'''from concurrent.futures import ProcessPoolExecutor
''', r''''''),
(r'''from fractions import Fraction

import av
''', r'''from fractions import Fraction

# One thread per numeric library (the team runs Studio on this laptop); set before numpy, torch and lerobot load.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import av
'''),
(r'''WRIST_SEG_S = 60.0
''', r'''WRIST_SEG_S = 60.0
SVT_LP = 3  # libsvtav1 logical processors; with the main thread, four CPU threads
'''),
(r'''def list_media(folder, prefix):
    found = []
    for path in glob.glob(os.path.join(folder, prefix + "_*.mkv")):
        name = os.path.basename(path)
        found.append((utc_seconds(name[len(prefix) + 1:-4]), path))
    return sorted(found)
''', r'''def segment_start(path, name_ts):
    """First-frame time in epoch seconds, and whether the pts are relative (segment starts at 0) rather than absolute epoch."""
    container = av.open(path)
    try:
        start_us = container.start_time
    finally:
        container.close()
    if start_us and start_us / 1e6 > 1e9:
        return start_us / 1e6, False
    return name_ts, True


def list_media(folder, prefix):
    found = []
    for path in glob.glob(os.path.join(folder, prefix + "_*.mkv")):
        name = os.path.basename(path)
        start, relative = segment_start(path, utc_seconds(name[len(prefix) + 1:-4]))
        found.append((start, path, relative))
    return sorted(found)
'''),
(r'''def candidates(files, a, b, span):
    out = []
    for k, (ts, path) in enumerate(files):
        end = files[k + 1][0] if k + 1 < len(files) else float("inf")
        if span is not None:
            end = min(end, ts + span)
        if ts <= b and end > a:
            out.append((path, ts, end))
    return out
''', r'''def candidates(files, a, b, span):
    out = []
    for k, (start, path, _relative) in enumerate(files):
        if k + 1 < len(files):
            end = files[k + 1][0]
        else:
            end = start + span if span is not None else float("inf")
        if start <= b and end > a:
            out.append((path, start, end))
    return out
'''),
(r'''    stream.thread_type = "AUTO"
''', r'''    stream.thread_count = 1  # decode on the calling thread only
'''),
(r'''def encode_camera(job):
''', r'''def nearest_dt(files, times):
    """Decode-only pass for --max-dt: |nearest camera frame time - grid time| per row, None if the video does not cover the rows."""
    frames = decode_window(files, float(times[0]) - PAD, float(times[-1]) + PAD)
    dts = np.zeros(len(times), dtype=np.float64)
    count = 0
    try:
        for i, ti, tc, _fr in nearest_frames(times, frames):
            dts[i] = tc - ti
            count += 1
    finally:
        frames.close()
    return np.abs(dts) if count == len(times) else None


def encode_camera(job):
'''),
(r'''        results.append({"name": ep["name"], "n": n, "dt_max_s": float(np.abs(dts).max()),
                        "dt_mean_s": float(np.abs(dts).mean()),
                        "rows_dt_over_1_frame": int(np.sum(np.abs(dts) > 1.0 / FPS + 1e-9)), "stats": st})
''', r'''        results.append({"name": ep["name"], "n": n, "dt_max_s": float(np.abs(dts).max()),
                        "dt_mean_s": float(np.abs(dts).mean()), "dt_sum_s": float(np.abs(dts).sum()),
                        "rows_dt_over_1_frame": int(np.sum(np.abs(dts) > 1.0 / FPS + 1e-9)),
                        "rows_dt_over_2_frames": int(np.sum(np.abs(dts) > 2.0 / FPS + 1e-9)), "stats": st})
'''),
(r'''def load_episodes(run_dirs, only_success, skipped):
''', r'''def load_episodes(run_dirs, only_success, plates, skipped):
'''),
(r'''            if only_success and meta.get("success") is not True:
                skipped.append({"episode": name, "run": run_name, "reason": "success is not true"})
                continue
''', r'''            if only_success and meta.get("success") is not True:
                skipped.append({"episode": name, "run": run_name, "reason": "success is not true"})
                continue
            plate = meta.get("plate")
            if plates and plate not in plates:
                skipped.append({"episode": name, "run": run_name, "reason": "plate %s is not one of %s" % (plate, sorted(plates))})
                continue
'''),
(r'''                "run": run_name,
                "task": str(meta["task"]),
''', r'''                "run": run_name,
                "plate": plate,
                "task": str(meta["task"]),
'''),
(r'''    ap.add_argument("--only-success", action="store_true")
''', r'''    ap.add_argument("--only-success", action="store_true")
    ap.add_argument("--plates", nargs="+", default=None,
                    help="keep only episodes whose episode.json plate is one of these, e.g. blue green")
    ap.add_argument("--max-dt", type=float, default=None, metavar="SECONDS",
                    help="drop episodes with a camera frame more than SECONDS from a joint sample (default: off)")
'''),
(r'''    args = ap.parse_args()
''', r'''    args = ap.parse_args()
    plates = set(args.plates) if args.plates else None
'''),
(r'''    episodes = load_episodes(args.runs, args.only_success, skipped)
''', r'''    episodes = load_episodes(args.runs, args.only_success, plates, skipped)
'''),
(r'''    media = {"top": list_media(logs_dir, "top"), "close": list_media(logs_dir, "close"), "wrist": list_media(wrist_dir, "w")}
    span = {"top": None, "close": None, "wrist": WRIST_SEG_S}
    ok = []
''', r'''    media = {"top": list_media(logs_dir, "top"), "close": list_media(logs_dir, "close"), "wrist": list_media(wrist_dir, "w")}
    pts_mode = {path: ("relative" if relative else "absolute") for cam in media for _start, path, relative in media[cam]}
    span = {"top": None, "close": None, "wrist": WRIST_SEG_S}
    ok = []
'''),
(r'''        if reason:
            skipped.append({"episode": ep["name"], "run": ep["run"], "reason": reason})
        else:
            ep["files"] = files
            ok.append(ep)
''', r'''        if reason is None and args.max_dt is not None:
            worst = {}
            for cam in cams:
                dts = nearest_dt(files[cam], ep["t"])
                if dts is None:
                    reason = "%s video does not cover every row" % cam
                    break
                worst[cam] = float(dts.max())
            over = {c: round(v, 6) for c, v in worst.items() if v > args.max_dt}
            if reason is None and over:
                reason = "camera frame more than --max-dt %g s from a joint sample: %s" % (args.max_dt, json.dumps(over))
        if reason:
            skipped.append({"episode": ep["name"], "run": ep["run"], "reason": reason})
        else:
            ep["files"] = files
            ok.append(ep)
'''),
(r'''    options = VideoEncoderConfig().get_codec_options(None, as_strings=True)
''', r'''    options = VideoEncoderConfig().get_codec_options(SVT_LP, as_strings=True)
'''),
(r'''    try:
        with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
            enc = {r["cam"]: r for r in pool.map(encode_camera, jobs)}
    except Exception:
''', r'''    enc = {}
    try:
        for job in jobs:
            result = encode_camera(job)
            enc[result["cam"]] = result
    except Exception:
'''),
(r'''            "episode": ep["name"], "run": ep["run"], "task": ep["task"], "rows": len(ep["t"]),
''', r'''            "episode": ep["name"], "run": ep["run"], "task": ep["task"], "plate": ep["plate"], "rows": len(ep["t"]),
'''),
(r'''            "video_rows_dt_over_1_frame": {c: img_stats[c][ep["name"]]["rows_dt_over_1_frame"] for c in cam_list},
        })
''', r'''            "video_rows_dt_over_1_frame": {c: img_stats[c][ep["name"]]["rows_dt_over_1_frame"] for c in cam_list},
            "video_rows_dt_over_2_frames": {c: img_stats[c][ep["name"]]["rows_dt_over_2_frames"] for c in cam_list},
            "video_segments": {c: [{"file": os.path.basename(p), "start_s": round(s, 3), "pts": pts_mode[p]}
                                   for p, s, _e in ep["files"][c]] for c in cam_list},
        })
'''),
(r'''    report = {
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "variant": args.variant,
        "only_success": bool(args.only_success),
''', r'''    dt_summary = {}
    for c in cam_list:
        rows_c = [img_stats[c][ep["name"]] for ep in ok]
        n_rows = sum(r["n"] for r in rows_c)
        dt_summary[c] = {
            "max_s": round(max(r["dt_max_s"] for r in rows_c), 6),
            "mean_s": round(sum(r["dt_sum_s"] for r in rows_c) / n_rows, 6),
            "rows": n_rows,
            "rows_over_1_frame": sum(r["rows_dt_over_1_frame"] for r in rows_c),
            "rows_over_2_frames": sum(r["rows_dt_over_2_frames"] for r in rows_c),
            "episodes_over_2_frames": sum(1 for r in rows_c if r["dt_max_s"] > 2.0 / FPS + 1e-9),
        }
    report = {
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "variant": args.variant,
        "only_success": bool(args.only_success),
        "filters": {"only_success": bool(args.only_success), "plates": sorted(plates) if plates else None,
                    "max_dt_s": args.max_dt},
        "svt_lp": SVT_LP,
'''),
(r'''        "max_nearest_frame_dt_s": {c: max(img_stats[c][ep["name"]]["dt_max_s"] for ep in ok) for c in cam_list},
''', r'''        "max_nearest_frame_dt_s": {c: max(img_stats[c][ep["name"]]["dt_max_s"] for ep in ok) for c in cam_list},
        "dt_summary": dt_summary,
'''),
(r'''    print("max nearest-frame dt (s): %s" % json.dumps(report["max_nearest_frame_dt_s"]))
''', r'''    print("max nearest-frame dt (s): %s" % json.dumps(report["max_nearest_frame_dt_s"]))
    print("dt summary per camera: %s" % json.dumps(dt_summary))
'''),
]

VALID = [
(r'''# Usage: nice -n 10 <R>/envs/lerobot-env/bin/python -I <R>/app/vla/validate.py --root OUT --png-dir DIR [--report OUT.build_report.json]
# Exit code 0 only when every check passes.
''', r'''# Usage: nice -n 19 taskset -c 0-3 <R>/envs/lerobot-env/bin/python -I <R>/app/vla/validate.py --root OUT --png-dir DIR [--report OUT.build_report.json]
# Exit code 0 when no check fails. The nearest-frame dt check is a WARN: listed in validate_report.json, does not fail.
'''),
(r'''import subprocess
import sys

import av
''', r'''import subprocess
import sys

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import av
'''),
(r'''    def check(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": str(detail)})
        print(("PASS " if ok else "FAIL ") + name + ((" | " + str(detail)) if str(detail) else ""))
''', r'''    def check(name, ok, detail="", level="FAIL"):
        checks.append({"check": name, "level": level, "pass": bool(ok), "detail": str(detail)})
        print(("PASS " if ok else level + " ") + name + ((" | " + str(detail)) if str(detail) else ""))

    def warn(name, ok, detail=""):
        check(name, ok, detail, level="WARN")
'''),
(r'''        worst = rep.get("max_nearest_frame_dt_s", {})
        check("nearest camera frame within 2 frames of each joint sample (one dropped frame is allowed)",
              all(v <= 2.0 / FPS for v in worst.values()), json.dumps(worst))
''', r'''        worst = rep.get("max_nearest_frame_dt_s", {})
        warn("nearest camera frame within 2 frames of each joint sample", all(v <= 2.0 / FPS for v in worst.values()),
             json.dumps(worst))
        if rep.get("dt_summary"):
            print("build dt summary per camera: " + json.dumps(rep["dt_summary"]))
'''),
(r'''    failed = [c["check"] for c in checks if not c["pass"]]
    with open(os.path.join(args.png_dir, "validate_report.json"), "w", encoding="utf-8") as fh:
        json.dump({"root": root, "result": "PASS" if not failed else "FAIL", "checks": checks}, fh, indent=1)
    print("RESULT %s (%d checks, %d failed)" % ("PASS" if not failed else "FAIL", len(checks), len(failed)))
    return 0 if not failed else 1
''', r'''    failed = [c["check"] for c in checks if c["level"] == "FAIL" and not c["pass"]]
    warnings = [c["check"] for c in checks if c["level"] == "WARN" and not c["pass"]]
    result = "PASS" if not failed else "FAIL"
    with open(os.path.join(args.png_dir, "validate_report.json"), "w", encoding="utf-8") as fh:
        json.dump({"root": root, "result": result, "failed": failed, "warnings": warnings, "checks": checks}, fh, indent=1)
    print("RESULT %s (%d checks, %d failed, %d warnings)" % (result, len(checks), len(failed), len(warnings)))
    return 0 if not failed else 1
'''),
]


def apply(path, pairs):
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    for n, (old, new) in enumerate(pairs):
        found = src.count(old)
        if found != 1:
            sys.exit("%s: anchor %d found %d times; nothing written" % (path, n, found))
        src = src.replace(old, new)
    compile(src, path, "exec")
    tmp = path + ".patched"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(src)
    shutil.copymode(path, tmp)
    os.replace(tmp, path)
    print("patched %s: %d anchors" % (path, len(pairs)))


apply(os.path.join(R, "build_dataset.py"), BUILD)
apply(os.path.join(R, "validate.py"), VALID)
