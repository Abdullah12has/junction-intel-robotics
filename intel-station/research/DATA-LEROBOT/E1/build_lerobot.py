#!/usr/bin/env python3
"""DATA-LEROBOT E1: LeRobot v3.0 dataset (Studio layout) from visually verified successful sort episodes.

Base: R/app/vla/build_dataset.py (P2-D-VLA1). Kept as written there: segment clock rules (absolute pts when the
container start is above 1e9, else relative with the segment start taken from the UTC file name), decode window,
nearest camera frame per 30 fps grid row, AV1 encoding (libsvtav1 lp=3 through VideoEncoderConfig, yuv420p, crf 30,
preset 12, GOP 2), table and stats layout, RealSense camera keys, template metadata from the team's Studio dataset
(read only).
Changed here:
  1. camera clock lag (P3-VLA-CLOCK): the camera frame for grid time t is the frame nearest to t + lag, where
     lag = camera time minus joint time (top +21.5 ms, wrist +7.5 ms, close 0 ms: not determined);
  2. one AV1 file per episode and camera, with stable file and episode numbers in state.json, so a re-run encodes
     only new or changed episodes and appends new ones;
  3. inclusion rule: a visual verdict of "placed on the plate the loop chose, decision correct" (VISUAL table,
     transcribed from P3-LABEL-OURS V1). Loop-only successes are excluded unless --include-loop-only is given;
  4. a status and a reason for every candidate episode (episodes.tsv, excluded.tsv);
  5. validation.json, data_card.md, STATUS.md and 5 random visual samples.

Usage on the laptop (R = /home/intel/intel-junction-2026):
  nice -n 19 env OMP_NUM_THREADS=4 /home/intel/intel-junction-2026/envs/lerobot-env/bin/python -I \
      /home/intel/intel-junction-2026/work/DATA-LEROBOT/E1/build_lerobot.py [--no-encode] [--include-loop-only]
"""

import argparse
import copy
import csv
import glob
import hashlib
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from fractions import Fraction

import numpy as np

SCRIPT_VERSION = "build_lerobot.py e1-2 2026-11-14"
ROOT = "/home/intel/intel-junction-2026"
TEMPLATE_REL = "data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af"
OUT_REL = "data/ours-lerobot/v1"
WORK_REL = "work/DATA-LEROBOT/E1"
REPO_ID = "local/ours_lerobot_v1"
FPS = 30
PAD = 0.5
WRIST_SEG_S = 60.0
SVT_LP = 3
MAX_DT_S = 0.1
WARN_DT_S = 2.0 / FPS
MAX_JOINT_GAP_S = 1.0
CAMS = ["top", "close", "wrist"]
LAG_S = {"top": 0.0215, "wrist": 0.0075, "close": 0.0}
LAG_TEXT = {
    "top": "+21.5 ms (P3-VLA-CLOCK median; reliable in 11 of 15 episodes)",
    "wrist": "+7.5 ms (P3-VLA-CLOCK median; reliable in 15 of 15 episodes)",
    "close": "0 ms applied: not determined (P3-VLA-CLOCK reliable in 3 of 15 episodes)",
}
CAM_KEY = {
    "close": "observation.images.realsense 243722064316",
    "top": "observation.images.realsense 243622061421",
    "wrist": "observation.images.wrist",
}
CAM_ROLE = {"top": "overhead, serial 243622061421", "close": "close view, serial 243722064316",
            "wrist": "wrist camera (MJPEG segments, 60 s)"}
ACTION_COLS = ["goal%d" % j for j in range(6)]
STATE_COLS = ["pos%d" % j for j in range(6)]
STAT_ORDER = ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]
FEATURE_ORDER = ["action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index"]
OURS_COLS = ["ours_ep_no", "ours_run", "ours_episode", "ours_colour", "ours_plate", "ours_od10_label", "ours_tier",
             "ours_verified_by", "ours_decision_correct", "ours_decision_confidence", "ours_needs_review",
             "ours_loop_task", "ours_note"]
EP_FIELDS = ["run", "episode", "ep_no", "status", "tier", "verified_by", "visual_outcome", "colour", "plate", "od10",
             "task", "needs_review", "decision_correct", "decision_confidence", "rows", "frames", "duration_s",
             "joint_gap_max_s", "loop_success", "loop_label", "loop_task", "loop_failure", "reason", "note"]
EX_FIELDS = ["run", "episode", "reason", "visual_outcome", "loop_success", "loop_failure", "loop_task"]

# Visual verdicts for the 22 sort episodes of any_1731 and any_1740 (sandbox P3-LABEL-OURS V1: labels_episodes_v2.jsonl
# and verdicts.tsv). outcome: placed | estop | carry_over | stopped | approach_fail | grasp_fail | not_placed |
# unverified | stub. decision: whether the loop plate was correct (yes | no | n/a | unverified | unknown).
# plate: the plate the brick was released on, or the correct OD-10 plate where the visual check gives one.
# colour: the loop colour name where the visual check agrees (cream/yellow bricks are named yellow by the loop).
# unverifiable: items the visual check could not confirm at 640x480 (sets needs_review).
VISUAL = {
    "any_1731/ep_0001": {"outcome": "estop", "colour": "orange", "plate": None, "decision": "n/a", "conf": "high",
                         "note": "estop with the orange 2x2 (no tape) held; brick not placed"},
    "any_1731/ep_0002": {"outcome": "estop", "colour": "orange", "plate": "blue", "decision": "no", "conf": "high",
                         "note": "orange 2x2 with no tape on any face (closed_wrist, lifted_wrist); loop plate green is wrong; estop, brick not placed"},
    "any_1731/ep_0003": {"outcome": "placed", "colour": "orange", "plate": "blue", "decision": "yes", "conf": "med",
                         "note": "released_wrist: orange 2x2 in the blue plate (brick carried over from ep_0002); the loop view sat on the cream brick, so the loop evidence is wrong"},
    "any_1731/ep_0004": {"outcome": "stub", "colour": None, "plate": None, "decision": "unknown", "conf": "low",
                         "note": "episode.json holds only t_end; no image files"},
    "any_1740/ep_0001": {"outcome": "placed", "colour": "orange", "plate": "blue", "decision": "yes", "conf": "high",
                         "note": "orange 2x2 without tape (pre_close, stage1_wrist, closed_wrist); released_wrist: in the blue plate; loop colour yellow is wrong (scan hue 14)"},
    "any_1740/ep_0002": {"outcome": "approach_fail", "colour": None, "plate": None, "decision": "unknown", "conf": "med",
                         "note": "approach failed, no brick moved (pre frames)"},
    "any_1740/ep_0003": {"outcome": "approach_fail", "colour": None, "plate": None, "decision": "unknown", "conf": "med",
                         "note": "approach failed, no brick moved; same scene as ep_0002"},
    "any_1740/ep_0004": {"outcome": "carry_over", "colour": "red", "plate": None, "decision": "n/a", "conf": "high",
                         "note": "estop with the red 2x2 held (closed_wrist, pre_close); carried into ep_0005; not placed"},
    "any_1740/ep_0005": {"outcome": "not_placed", "colour": None, "plate": None, "decision": "unverified", "conf": "low",
                         "note": "loop success not supported: red 2x2 on the mat in released_top and released_close; no brick added to the blue plate"},
    "any_1740/ep_0006": {"outcome": "not_placed", "colour": "yellow", "plate": "green", "decision": "yes", "conf": "high",
                         "note": "cream/yellow 2x2 with tape (pre_close); released on the mat beside the green plate (released_top, released_close); loop success rejected, green decision correct"},
    "any_1740/ep_0007": {"outcome": "carry_over", "colour": "red", "plate": None, "decision": "n/a", "conf": "med",
                         "note": "red 2x2 held through the lift (grasp_open_close, closed_close, lifted_wrist); estop; no release frames on disk"},
    "any_1740/ep_0008": {"outcome": "stopped", "colour": "red", "plate": None, "decision": "n/a", "conf": "high",
                         "note": "red 2x2 put down at the station on the mat (station_top, station_close); no plate; loop success not supported"},
    "any_1740/ep_0009": {"outcome": "unverified", "colour": None, "plate": None, "decision": "unverified", "conf": "low",
                         "note": "pre frames only; no grasp or release frames on disk"},
    "any_1740/ep_0010": {"outcome": "unverified", "colour": None, "plate": None, "decision": "unverified", "conf": "low",
                         "note": "pre frames only; same scene as ep_0009"},
    "any_1740/ep_0011": {"outcome": "unverified", "colour": None, "plate": None, "decision": "unverified", "conf": "low",
                         "note": "pre frames only (contact sheet); no release frames"},
    "any_1740/ep_0012": {"outcome": "unverified", "colour": None, "plate": None, "decision": "unverified", "conf": "low",
                         "note": "pre frames only (contact sheet); no release frames"},
    "any_1740/ep_0013": {"outcome": "grasp_fail", "colour": "red", "plate": None, "decision": "n/a", "conf": "med",
                         "note": "grasp failed, brick not lifted (lifted_wrist); the closed_wrist reading conflicts and stays unresolved at sheet resolution"},
    "any_1740/ep_0014": {"outcome": "unverified", "colour": None, "plate": None, "decision": "unverified", "conf": "low",
                         "note": "white placement not seen at release (released_top, released_close); the cream taped brick on the mat is a resident brick"},
    "any_1740/ep_0015": {"outcome": "placed", "colour": "red", "plate": "blue", "decision": "yes", "conf": "med",
                         "unverifiable": ["tape", "orientation"],
                         "note": "released_close: red 2x2 in the blue plate; tape and orientation not resolvable at 640x480"},
    "any_1740/ep_0016": {"outcome": "unverified", "colour": None, "plate": None, "decision": "unverified", "conf": "low",
                         "note": "pre frames only; loop reports approach_fail"},
    "any_1740/ep_0017": {"outcome": "placed", "colour": "yellow", "plate": "green", "decision": "yes", "conf": "med",
                         "unverifiable": ["tape"],
                         "note": "released_top, released_close: cream/yellow 2x2 on the green plate; dark tape not confirmable at 640x480, so the green decision holds only if the tape is present"},
    "any_1740/ep_0018": {"outcome": "stub", "colour": None, "plate": None, "decision": "unknown", "conf": "low",
                         "note": "no frames on disk"},
}


class GapError(Exception):
    pass


LOG_FH = None


def log(msg):
    line = "%s %s" % (datetime.now(timezone.utc).strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    if LOG_FH is not None:
        LOG_FH.write(line + "\n")
        LOG_FH.flush()


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def date_u():
    return time.strftime("%a %b %d %H:%M:%S UTC %Y", time.gmtime())


def utc_seconds(stamp):
    return datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()


def write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def video_path(out, cam, fi):
    return os.path.join(out, "videos", CAM_KEY[cam], "chunk-000", "file-%03d.mp4" % fi)


# ---------------------------------------------------------------- media (P2 segment clock rules)

def segment_start(path, name_ts):
    """Segment start in epoch seconds, and whether its pts are relative (starting at 0) rather than absolute."""
    import av
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


def load_media(root):
    return {
        "top": list_media(os.path.join(root, "logs/monitor"), "top"),
        "close": list_media(os.path.join(root, "logs/monitor"), "close"),
        "wrist": list_media(os.path.join(root, "data/wrist/raw"), "w"),
    }


def candidates(files, a, b, span):
    """Segments that overlap [a, b]; a segment ends where the next one starts (last one: span, or open-ended)."""
    out = []
    for k, (start, path, _relative) in enumerate(files):
        if k + 1 < len(files):
            end = files[k + 1][0]
        else:
            end = start + span if span is not None else float("inf")
        if start <= b and end > a:
            out.append((path, start, end))
    return out


def open_video(path):
    import av
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_count = 1
    relative = (container.start_time or 0) / 1e6 < 1e9
    return container, stream, relative


def has_packet_at_or_after(path, ts, t):
    try:
        container, stream, relative = open_video(path)
    except Exception:
        return False
    try:
        base = ts if relative else 0.0
        tb = float(stream.time_base)
        target = int(round((t - base) / tb))
        if target > 0:
            try:
                container.seek(target, stream=stream, backward=True)
            except Exception:
                return False
        for pkt in container.demux(stream):
            if pkt.pts is not None and pkt.pts * tb + base >= t:
                return True
        return False
    finally:
        container.close()


def decode_window(files, a, b):
    """Decoded frames (time, frame) with a <= time <= b, in the order of the segments, using decoded-frame pts
    (packet order is not monotonic because of B-frames)."""
    import av
    for path, ts, end in files:
        container, stream, relative = open_video(path)
        base = ts if relative else 0.0
        tb = float(stream.time_base)
        stop = past = False
        try:
            target = int(round((a - base) / tb))
            if target > 0:
                try:
                    container.seek(target, stream=stream, backward=True)
                except av.error.FFmpegError:
                    pass
            for pkt in container.demux(stream):
                if pkt.pts is None:
                    continue
                try:
                    decoded = pkt.decode()
                except av.error.FFmpegError:
                    continue
                for fr in decoded:
                    if fr.pts is None:
                        continue
                    t = fr.pts * tb + base
                    if t > b:
                        past = stop = True
                        break
                    if t >= end:
                        stop = True
                        break
                    if t >= a:
                        yield t, fr
                if stop:
                    break
        finally:
            container.close()
        if past:
            return


def nearest_frames(times, frames):
    it = iter(frames)
    cur = next(it, None)
    if cur is None:
        return
    nxt = next(it, None)
    for i, ti in enumerate(times):
        while nxt is not None and abs(nxt[0] - ti) <= abs(cur[0] - ti):
            cur, nxt = nxt, next(it, None)
        yield i, ti, cur[0], cur[1]


def coverage(rec, media):
    """Segments per camera for the lag-shifted episode window, or (None, reason)."""
    t0, t1 = float(rec["t"][0]), float(rec["t"][-1])
    files = {}
    for cam in CAMS:
        lag = LAG_S[cam]
        span = WRIST_SEG_S if cam == "wrist" else None
        cand = candidates(media[cam], t0 + lag - PAD, t1 + lag + PAD, span)
        if not cand:
            return None, "no %s video covers the episode window" % cam
        if cand[0][1] > t0 + lag:
            return None, "%s video starts after the episode start" % cam
        relative = {p: rel for _s, p, rel in media[cam]}
        if any(relative.get(c[0], False) for c in cand):
            return None, "%s segment has relative pts (no absolute clock)" % cam
        last_path, last_start, _end = cand[-1]
        if not has_packet_at_or_after(last_path, last_start, t1 + lag - 1.0 / FPS):
            return None, "%s video does not reach the episode end (still recording, or a gap)" % cam
        files[cam] = cand
    return files, ""


# ---------------------------------------------------------------- episodes and labels

def blank_record(run, name):
    return {"run": run, "episode": name, "key": "%s/%s" % (run, name), "status": "excluded", "reason": "",
            "tier": "", "verified_by": "", "visual_outcome": "", "note": "", "colour": "", "plate": "", "od10": "",
            "task": "", "decision_correct": "", "decision_confidence": "", "needs_review": "",
            "loop_success": "", "loop_failure": "", "loop_label": "", "loop_task": "", "rows": 0, "bad_rows": 0,
            "joint_gap_max_s": None, "fingerprint": "", "t": None, "action": None, "state": None, "cover": None,
            "file_index": None, "ep_no": None, "index": None, "row0": None, "cam_stats": None, "stats": None}


def read_joints(path, t_start, t_end):
    """Joint rows inside [t_start, t_end]. Returns (times, values[goal0..5, pos0..5], bad_rows, error)."""
    need = ["t"] + ACTION_COLS + STATE_COLS
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        missing = [c for c in need if c not in header]
        if missing:
            return None, None, 0, "missing columns %s" % missing
        ts, rows, bad = [], [], 0
        for row in reader:
            try:
                t = float(row["t"])
                vals = [float(row[c]) for c in ACTION_COLS + STATE_COLS]
            except (TypeError, ValueError):
                bad += 1
                continue
            if t_start <= t <= t_end:
                ts.append(t)
                rows.append(vals)
    if not ts:
        return np.zeros(0), np.zeros((0, 12)), bad, ""
    return np.array(ts, dtype=np.float64), np.array(rows, dtype=np.float64), bad, ""


def classify(run, name, ep_dir, include_loop_only):
    """Status and reason for one episode: visual verdict first, then the files, then the loop record."""
    rec = blank_record(run, name)
    key = rec["key"]
    v = VISUAL.get(key)
    lead = ""
    if v is not None:
        rec["verified_by"] = "visual"
        rec["visual_outcome"] = v["outcome"]
        rec["note"] = v["note"]
        if v["outcome"] != "placed" or v["decision"] != "yes":
            rec["reason"] = "visual check: %s; %s" % (v["outcome"], v["note"])
            return rec
        lead = "visual check placed it, but "
    else:
        rec["verified_by"] = "loop"

    def fail(msg):
        rec["reason"] = lead + msg
        return rec

    meta_path = os.path.join(ep_dir, "episode.json")
    joints_path = os.path.join(ep_dir, "joints.csv")
    missing = [f for f, p in (("episode.json", meta_path), ("joints.csv", joints_path)) if not os.path.isfile(p)]
    if missing:
        return fail("not finished: %s missing" % " and ".join(missing))
    raw_meta = open(meta_path, "rb").read()
    raw_joints = open(joints_path, "rb").read()
    rec["fingerprint"] = hashlib.sha256(b"\0".join([
        raw_meta, raw_joints, json.dumps(LAG_S, sort_keys=True).encode(), SCRIPT_VERSION.encode(),
        json.dumps(v, sort_keys=True).encode(), str(include_loop_only).encode()])).hexdigest()
    try:
        meta = json.loads(raw_meta.decode("utf-8"))
    except Exception as exc:
        return fail("episode.json unreadable (interrupted write?): %s" % str(exc)[:120])
    if not isinstance(meta, dict):
        return fail("episode.json is not a JSON object")
    rec["loop_success"] = repr(meta.get("success"))
    rec["loop_failure"] = str(meta.get("failure"))
    rec["loop_label"] = str(meta.get("label") or "")
    rec["loop_task"] = str(meta.get("task") or "")
    if v is None and meta.get("success") is not True:
        return fail("not a success: loop success flag %r (%s)" % (
            meta.get("success"), meta.get("failure") or "no failure text"))
    if meta.get("t_start") is None or meta.get("t_end") is None:
        return fail("episode.json has no t_start or t_end (no joint window)")
    t_start, t_end = float(meta["t_start"]), float(meta["t_end"])
    t_arr, src, bad, why = read_joints(joints_path, t_start, t_end)
    if why:
        return fail("joints.csv: " + why)
    rec["rows"], rec["bad_rows"] = len(t_arr), bad
    if len(t_arr) < 2:
        return fail("fewer than 2 joint rows in the episode window (%d)" % len(t_arr))
    if np.any(np.diff(t_arr) <= 0):
        return fail("joint times are not strictly increasing")
    gap = float(np.diff(t_arr).max())
    rec["joint_gap_max_s"] = round(gap, 4)
    if gap > MAX_JOINT_GAP_S:
        return fail("joint sample gap of %.2f s inside the episode window" % gap)
    grid = t_arr[0] + np.arange(int(np.floor((t_arr[-1] - t_arr[0]) * FPS + 1e-9)) + 1) / FPS
    if len(grid) < 2:
        return fail("fewer than 2 grid rows at 30 fps")
    res = np.stack([np.interp(grid, t_arr, src[:, j]) for j in range(12)], axis=1).astype(np.float32)
    rec["t"], rec["action"], rec["state"] = grid, res[:, :6], res[:, 6:]

    loop_colour = str(meta.get("colour") or "").strip().lower()
    loop_plate = str(meta.get("plate") or "").strip().lower()
    loop_label = str(meta.get("label") or "").strip().upper()
    notes = []
    if v is not None:
        colour, plate, conf = v["colour"], v["plate"], v["conf"]
        review = conf != "high" or bool(v.get("unverifiable"))
        decision_correct, tier = "yes", "V"
        if loop_colour and loop_colour != colour:
            notes.append("loop colour %s, visual %s" % (loop_colour, colour))
        if loop_plate and loop_plate != plate:
            notes.append("loop plate %s, visual %s" % (loop_plate, plate))
        if meta.get("success") is not True:
            notes.append("loop success flag %r overridden by the visual check" % (meta.get("success"),))
    else:
        if not include_loop_only:
            return fail("loop success not visually verified (no VISUAL entry)")
        colour, plate = loop_colour, loop_plate
        target = str(meta.get("target_colour") or "").strip().lower()
        if colour in ("", "unknown", "none"):
            return fail("colour unknown in the loop record")
        if target and target != colour:
            return fail("loop target colour %s differs from the placed colour %s" % (target, colour))
        if str(meta.get("decision_conf")) == "no_evidence":
            return fail("loop decision has no evidence (default label)")
        if plate not in ("blue", "green"):
            return fail("loop plate %r is not blue or green" % plate)
        if loop_label != ("GOOD" if plate == "blue" else "DEFECT"):
            return fail("loop label %s does not match plate %s" % (loop_label, plate))
        conf = str(meta.get("decision_conf"))
        review, decision_correct, tier = True, "unverified", "L"
    rec.update({"colour": colour, "plate": plate, "od10": "GOOD" if plate == "blue" else "DEFECT",
                "task": "move the %s brick to the %s plate" % (colour, plate), "needs_review": bool(review),
                "decision_correct": decision_correct, "decision_confidence": conf, "tier": tier,
                "note": "; ".join(x for x in [rec["note"]] + notes if x), "status": "included", "reason": ""})
    return rec


def scan_episodes(root, include_loop_only):
    sort_root = os.path.join(root, "logs/sort")
    runs = sorted(d for d in os.listdir(sort_root) if d.startswith("any_") and
                  os.path.isdir(os.path.join(sort_root, d)))
    recs = []
    for run in runs:
        for name in sorted(d for d in os.listdir(os.path.join(sort_root, run)) if d.startswith("ep_")):
            recs.append(classify(run, name, os.path.join(sort_root, run, name), include_loop_only))
    return runs, recs


def category(reason):
    rules = [("camera coverage", "camera coverage"), ("visual check placed it", "visual placed, data check failed"),
             ("visual check", "visual check"), ("not finished", "not finished"),
             ("not a success", "loop not a success"), ("not visually verified", "loop success not visually verified")]
    for needle, label in rules:
        if needle in reason:
            return label
    return "joint or label data"


# ---------------------------------------------------------------- state (incremental re-runs)

def read_state(work):
    path = os.path.join(work, "state.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {"file_index": {}, "next_file": 0, "ep_no": {}, "next_ep": 0, "episodes": {}}


def write_state(work, state):
    write_text(os.path.join(work, "state.json"), json.dumps(state, indent=1, sort_keys=True))


# ---------------------------------------------------------------- encode (P2 encode loop, one file per episode)

def encode_episode(cam, path, width, height, options, times, files):
    import av
    from lerobot.datasets.compute_stats import get_feature_stats, sample_indices
    n = len(times)
    keep = set(int(k) for k in sample_indices(n))
    small = []
    dts = np.zeros(n, dtype=np.float64)
    container = av.open(path, mode="w")
    try:
        stream = container.add_stream("libsvtav1", rate=FPS, options=options)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        frames = decode_window(files, float(times[0]) - PAD, float(times[-1]) + PAD)
        count = 0
        try:
            for i, ti, tc, fr in nearest_frames(times, frames):
                out_frame = fr.reformat(format="yuv420p")
                out_frame.pts = i
                out_frame.time_base = Fraction(1, FPS)
                for pkt in stream.encode(out_frame):
                    container.mux(pkt)
                if i in keep:
                    small.append(fr.reformat(width=160, height=120, format="rgb24",
                                             interpolation="AREA").to_ndarray())
                dts[i] = tc - ti
                count += 1
        finally:
            frames.close()
        if count != n:
            raise GapError("%s: %d of %d grid rows have a camera frame" % (cam, count, n))
        if not small:
            raise GapError("%s: no sampled frames for the image statistics" % cam)
        for pkt in stream.encode():
            container.mux(pkt)
    finally:
        container.close()
    pixels = np.ascontiguousarray(np.stack(small).transpose(0, 3, 1, 2).astype(np.float32) / 255.0)
    st = get_feature_stats(pixels, axis=(0, 2, 3), keepdims=True)
    st = {k: (np.asarray(v) if k == "count" else np.asarray(v)[0]) for k, v in st.items()}
    adt = np.abs(dts)
    return {"n": n, "dt_max_s": float(adt.max()), "dt_mean_s": float(adt.mean()), "dt_sum_s": float(adt.sum()),
            "rows_over_1": int(np.sum(adt > 1.0 / FPS + 1e-9)), "rows_over_2": int(np.sum(adt > 2.0 / FPS + 1e-9)),
            "stats": {k: np.asarray(v).tolist() for k, v in st.items()}}


def encode_record(r, out, state, options, width, height):
    fi = state["file_index"].get(r["key"])
    if fi is None:
        fi = state["next_file"]
        state["next_file"] = fi + 1
        state["file_index"][r["key"]] = fi
    finals = {c: video_path(out, c, fi) for c in CAMS}
    tmps = {c: finals[c][:-4] + ".tmp.mp4" for c in CAMS}
    results = {}
    try:
        for c in CAMS:
            os.makedirs(os.path.dirname(finals[c]), exist_ok=True)
            results[c] = encode_episode(c, tmps[c], width, height, options, r["t"] + LAG_S[c], r["cover"][c])
            if results[c]["dt_max_s"] > MAX_DT_S:
                raise GapError("%s: a grid row is %.3f s from its nearest camera frame (limit %.3f s)" % (
                    c, results[c]["dt_max_s"], MAX_DT_S))
    except Exception as exc:
        for c in CAMS:
            if os.path.exists(tmps[c]):
                os.remove(tmps[c])
        if isinstance(exc, GapError):
            return False, "camera frames: " + str(exc), None, None
        log(traceback.format_exc()[-1500:])
        return False, "camera frames: encode error %s" % repr(exc)[:200], None, None
    for c in CAMS:
        os.replace(tmps[c], finals[c])
    return True, "", results, fi


def np_stats(d):
    return {k: (np.asarray(v, dtype=np.int64) if k == "count" else np.asarray(v, dtype=np.float64))
            for k, v in d.items()}


def write_fsl(values, width):
    import pyarrow as pa
    arr = np.ascontiguousarray(values, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(arr.reshape(-1), type=pa.float32()), width)


def to_json_stats(stats):
    return {k: {kk: np.asarray(vv).tolist() for kk, vv in v.items()} for k, v in stats.items()}


# ---------------------------------------------------------------- dataset writer (v3.0 layout, as the template)

def write_dataset(out, inc, tpl_info, tpl_root):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats

    hf_meta = (pq.read_schema(os.path.join(tpl_root, "data/chunk-000/file-000.parquet")).metadata or {}).get(
        b"huggingface")
    row0 = 0
    for k, r in enumerate(inc):
        r["index"] = k
        r["row0"] = row0
        row0 += len(r["t"])
    total = row0
    tasks = []
    for r in inc:
        if r["task"] not in tasks:
            tasks.append(r["task"])
    n_eps = len(inc)
    for sub in ("data/chunk-000", "meta/episodes/chunk-000"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    frame_index = np.concatenate([np.arange(len(r["t"]), dtype=np.int64) for r in inc])
    table = pa.table({
        "action": write_fsl(np.concatenate([r["action"] for r in inc]), 6),
        "observation.state": write_fsl(np.concatenate([r["state"] for r in inc]), 6),
        "timestamp": pa.array((frame_index.astype(np.float64) / FPS).astype(np.float32), type=pa.float32()),
        "frame_index": pa.array(frame_index),
        "episode_index": pa.array(np.concatenate([np.full(len(r["t"]), r["index"], dtype=np.int64) for r in inc])),
        "index": pa.array(np.arange(total, dtype=np.int64)),
        "task_index": pa.array(np.concatenate([np.full(len(r["t"]), tasks.index(r["task"]), dtype=np.int64)
                                               for r in inc])),
    })
    if hf_meta:
        table = table.replace_schema_metadata({b"huggingface": hf_meta})
    pq.write_table(table, os.path.join(out, "data/chunk-000/file-000.parquet"))
    pd.DataFrame({"task_index": np.arange(len(tasks), dtype=np.int64)},
                 index=pd.Index(tasks, name="task")).to_parquet(os.path.join(out, "meta/tasks.parquet"))

    ep_stats = []
    for r in inc:
        n = len(r["t"])
        feats = {
            "action": r["action"],
            "observation.state": r["state"],
            "timestamp": (np.arange(n, dtype=np.float64) / FPS).astype(np.float32),
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, r["index"], dtype=np.int64),
            "index": np.arange(r["row0"], r["row0"] + n, dtype=np.int64),
            "task_index": np.full(n, tasks.index(r["task"]), dtype=np.int64),
        }
        stats = {k: get_feature_stats(v, axis=0, keepdims=(v.ndim == 1)) for k, v in feats.items()}
        for c in CAMS:
            stats[CAM_KEY[c]] = np_stats(r["cam_stats"][c]["stats"])
        r["stats"] = stats
        ep_stats.append(stats)
    dataset_stats = aggregate_stats(ep_stats)
    write_text(os.path.join(out, "meta/stats.json"),
               json.dumps(to_json_stats(dataset_stats), indent=4, sort_keys=True))

    names, types, cols = [], [], []

    def add(name, typ, values):
        names.append(name)
        types.append(typ)
        cols.append(values)

    add("episode_index", pa.int64(), [r["index"] for r in inc])
    add("tasks", pa.list_(pa.string()), [[r["task"]] for r in inc])
    add("length", pa.int64(), [len(r["t"]) for r in inc])
    add("data/chunk_index", pa.int64(), [0] * n_eps)
    add("data/file_index", pa.int64(), [0] * n_eps)
    add("dataset_from_index", pa.int64(), [r["row0"] for r in inc])
    add("dataset_to_index", pa.int64(), [r["row0"] + len(r["t"]) for r in inc])
    for c in CAMS:
        key = CAM_KEY[c]
        add("videos/%s/chunk_index" % key, pa.int64(), [0] * n_eps)
        add("videos/%s/file_index" % key, pa.int64(), [r["file_index"] for r in inc])
        add("videos/%s/from_timestamp" % key, pa.float64(), [0.0] * n_eps)
        add("videos/%s/to_timestamp" % key, pa.float64(), [len(r["t"]) / FPS for r in inc])
    video_keys = {CAM_KEY[c] for c in CAMS}
    for fkey in FEATURE_ORDER + [CAM_KEY[c] for c in CAMS]:
        for stat in STAT_ORDER:
            if stat == "count":
                typ = pa.list_(pa.int64())
            elif fkey in video_keys:
                typ = pa.list_(pa.list_(pa.list_(pa.float64())))
            else:
                typ = pa.list_(pa.float64())
            add("stats/%s/%s" % (fkey, stat), typ, [np.asarray(r["stats"][fkey][stat]).tolist() for r in inc])
    add("meta/episodes/chunk_index", pa.int64(), [0] * n_eps)
    add("meta/episodes/file_index", pa.int64(), [0] * n_eps)
    add("ours_ep_no", pa.int64(), [r["ep_no"] for r in inc])
    add("ours_run", pa.string(), [r["run"] for r in inc])
    add("ours_episode", pa.string(), [r["episode"] for r in inc])
    add("ours_colour", pa.string(), [r["colour"] for r in inc])
    add("ours_plate", pa.string(), [r["plate"] for r in inc])
    add("ours_od10_label", pa.string(), [r["od10"] for r in inc])
    add("ours_tier", pa.string(), [r["tier"] for r in inc])
    add("ours_verified_by", pa.string(), [r["verified_by"] for r in inc])
    add("ours_decision_correct", pa.string(), [r["decision_correct"] for r in inc])
    add("ours_decision_confidence", pa.string(), [str(r["decision_confidence"]) for r in inc])
    add("ours_needs_review", pa.bool_(), [bool(r["needs_review"]) for r in inc])
    add("ours_loop_task", pa.string(), [r["loop_task"] for r in inc])
    add("ours_note", pa.string(), [r["note"] for r in inc])
    ep_table = pa.Table.from_arrays([pa.array(c, type=t) for c, t in zip(cols, types)], names=names)
    pq.write_table(ep_table, os.path.join(out, "meta/episodes/chunk-000/file-000.parquet"))

    cam_feat = copy.deepcopy(tpl_info["features"][CAM_KEY["close"]])
    features = {}
    for key, val in tpl_info["features"].items():
        if key.startswith("observation.images."):
            continue
        features[key] = copy.deepcopy(val)
        if key == "observation.state":
            for c in CAMS:
                features[CAM_KEY[c]] = copy.deepcopy(cam_feat)
    info = copy.deepcopy(tpl_info)
    info.update({"features": features, "total_episodes": n_eps, "total_frames": total,
                 "total_tasks": len(tasks), "splits": {"train": "0:%d" % n_eps}})
    write_text(os.path.join(out, "meta/info.json"), json.dumps(info, indent=4))

    allowed = {os.path.abspath(video_path(out, c, r["file_index"])) for r in inc for c in CAMS}
    for p in glob.glob(os.path.join(out, "videos", "*", "chunk-000", "*.mp4")):
        if os.path.abspath(p) not in allowed:
            os.remove(p)
            log("removed stray video %s" % p)
    return tasks, total


# ---------------------------------------------------------------- validation

def validate(out, inc, recs, total, tasks):
    import av
    import re
    import pyarrow.parquet as pq
    checks = []

    def chk(name, ok, detail, warn_only=False):
        status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
        checks.append({"check": name, "status": status, "detail": detail})
        log("CHECK %-32s %s %s" % (name, status, detail[:220]))

    with open(os.path.join(out, "meta/info.json"), encoding="utf-8") as fh:
        info = json.load(fh)
    data = pq.read_table(os.path.join(out, "data/chunk-000/file-000.parquet"))
    meta = pq.read_table(os.path.join(out, "meta/episodes/chunk-000/file-000.parquet"))
    ep_idx = data.column("episode_index").to_numpy()
    fidx = data.column("frame_index").to_numpy()
    tsec = data.column("timestamp").to_numpy()
    index = data.column("index").to_numpy()
    task_idx = data.column("task_index").to_numpy()
    state = np.array(data.column("observation.state").to_pylist(), dtype=np.float64)
    action = np.array(data.column("action").to_pylist(), dtype=np.float64)
    n_eps = len(inc)
    keys = {r["key"] for r in recs}

    chk("rows_match_total", data.num_rows == total, "parquet rows %d, expected %d" % (data.num_rows, total))
    chk("episodes_match", meta.num_rows == n_eps, "meta rows %d, included %d" % (meta.num_rows, n_eps))
    chk("info_totals", info["total_episodes"] == n_eps and info["total_frames"] == total
        and info["total_tasks"] == len(tasks), "info.json total_episodes/frames/tasks")
    chk("index_contiguous", bool(np.array_equal(index, np.arange(total))), "global index 0..%d" % (total - 1))
    chk("values_finite", bool(np.isfinite(state).all() and np.isfinite(action).all()), "state and action finite")
    lengths = meta.column("length").to_pylist()
    from_idx = meta.column("dataset_from_index").to_pylist()
    to_idx = meta.column("dataset_to_index").to_pylist()
    tasks_meta = meta.column("tasks").to_pylist()
    per_ep_ok = frame_ok = dur_ok = layout_ok = task_meta_ok = True
    ts_err = 0.0
    for r in inc:
        k = r["index"]
        rows_k = ep_idx == k
        n = int(rows_k.sum())
        per_ep_ok &= (n == lengths[k] == len(r["t"]))
        frame_ok &= bool(np.array_equal(fidx[rows_k], np.arange(n)))
        dur_ok &= abs((n - 1) / FPS - (float(r["t"][-1]) - float(r["t"][0]))) <= 1.0 / FPS + 1e-9
        layout_ok &= (from_idx[k] == r["row0"] and to_idx[k] == r["row0"] + len(r["t"]))
        task_meta_ok &= (tasks_meta[k] == [r["task"]])
        ts_err = max(ts_err, float(np.abs(tsec[rows_k] - np.arange(n) / FPS).max()))
    chk("length_matches_rows", per_ep_ok, "meta length equals the rows of each episode and its joint grid")
    chk("frame_index_0_to_n", frame_ok, "frame_index runs 0..n-1 in each episode")
    chk("duration_matches_joint_window", dur_ok, "(n-1)/30 within one frame of the joint window")
    chk("timestamp_is_frame_over_fps", ts_err < 1e-5, "max |timestamp - frame_index/30| = %.2e s" % ts_err)
    chk("dataset_index_ranges", layout_ok, "dataset_from/to_index match the row layout")
    chk("meta_task_matches_rows", task_meta_ok, "meta tasks equal the task text of each episode")

    lo, hi = state.min(axis=0), state.max(axis=0)
    alo, ahi = action.min(axis=0), action.max(axis=0)
    chk("joint_range_state_body", bool(np.all(lo[:5] >= -100.0 - 1e-3) and np.all(hi[:5] <= 100.0 + 1e-3)),
        "pos0..4 within -100..100: min %s max %s" % (np.round(lo[:5], 2).tolist(), np.round(hi[:5], 2).tolist()))
    chk("joint_range_action_body", bool(np.all(alo[:5] >= -100.0 - 1e-3) and np.all(ahi[:5] <= 100.0 + 1e-3)),
        "goal0..4 within -100..100: min %s max %s" % (np.round(alo[:5], 2).tolist(), np.round(ahi[:5], 2).tolist()))
    chk("joint_range_gripper", bool(lo[5] >= -1e-3 and hi[5] <= 100.0 + 1e-3 and alo[5] >= -1e-3
                                    and ahi[5] <= 100.0 + 1e-3),
        "pos5 and goal5 within 0..100: state %.2f..%.2f, action %.2f..%.2f" % (lo[5], hi[5], alo[5], ahi[5]),
        warn_only=True)

    pattern = re.compile(r"^move the ([a-z]+) brick to the (blue|green) plate$")
    task_ok = label_ok = True
    for r in inc:
        m = pattern.match(tasks[int(task_idx[ep_idx == r["index"]][0])])
        task_ok &= bool(m and m.group(1) == r["colour"] and m.group(2) == r["plate"])
        label_ok &= (r["od10"] == ("GOOD" if r["plate"] == "blue" else "DEFECT"))
    chk("tasks_match_plates_and_colours", task_ok, "task text parsed and equal to colour and plate for each episode")
    chk("od10_label_matches_plate", label_ok, "GOOD with a blue plate, DEFECT with a green plate")
    chk("labels_present", all(r["colour"] and r["plate"] and r["od10"] for r in inc),
        "colour, plate and OD-10 label set for each episode")

    vid_ok, vid_detail = True, []
    for r in inc:
        for c in CAMS:
            cont = av.open(video_path(out, c, r["file_index"]))
            try:
                nf = sum(1 for _ in cont.decode(video=0))
            finally:
                cont.close()
            n = len(r["t"])
            vid_ok &= (nf == n)
            vid_detail.append("%s %s frames %d rows %d" % (r["key"], c, nf, n))
    chk("video_frames_match_rows", vid_ok, "; ".join(vid_detail))

    cam_dt = {}
    for c in CAMS:
        ds = [r["cam_stats"][c] for r in inc]
        mx = max(d["dt_max_s"] for d in ds)
        mean = sum(d["dt_sum_s"] for d in ds) / sum(d["n"] for d in ds)
        over1 = sum(d["rows_over_1"] for d in ds)
        over2 = sum(d["rows_over_2"] for d in ds)
        cam_dt[c] = {"max_s": round(mx, 6), "mean_s": round(mean, 6), "rows_over_1_frame": over1,
                     "rows_over_2_frames": over2}
        chk("camera_dt_%s" % c, mx <= WARN_DT_S + 1e-9,
            "grid row to nearest camera frame: max %.4f s, mean %.4f s, rows over 1 frame %d, over 2 frames %d"
            % (mx, mean, over1, over2), warn_only=True)

    chk("info_fps_30", info["fps"] == FPS, "info.json fps %s" % info["fps"])
    chk("video_keys_in_info", all(CAM_KEY[c] in info["features"] for c in CAMS), "three camera keys in info.json")
    chk("ours_label_columns", all(name in meta.column_names for name in OURS_COLS), "per-episode label columns present")
    chk("excluded_have_reasons", all(r["reason"] for r in recs if r["status"] != "included"),
        "every excluded candidate has a reason")
    chk("visual_entries_have_folders", all(k in keys for k in VISUAL), "every VISUAL entry matches an episode folder")

    lr = {}
    try:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            from lerobot.datasets import LeRobotDataset
        try:
            ds = LeRobotDataset(REPO_ID, root=out, video_backend="pyav")
        except TypeError:
            ds = LeRobotDataset(REPO_ID, root=out)
        item = ds[0]
        img_keys = sorted(k for k in item.keys() if k.startswith("observation.images."))
        shapes = {k: list(item[k].shape) for k in img_keys}
        lr = {"ok": True, "frames": int(len(ds)), "episodes": int(ds.meta.total_episodes), "image_keys": img_keys,
              "image_shapes": shapes, "task_example": str(item.get("task"))}
        chk("lerobot_loader", len(ds) == total and int(ds.meta.total_episodes) == n_eps and len(img_keys) == 3,
            "LeRobotDataset frames %d episodes %d images %s" % (len(ds), int(ds.meta.total_episodes), shapes))
    except Exception as exc:
        lr = {"ok": False, "error": repr(exc)[:400], "traceback_tail": traceback.format_exc()[-1200:]}
        chk("lerobot_loader", False, "LeRobotDataset failed: %s" % repr(exc)[:300])
    details = {"joint_ranges": {"state_min": np.round(lo, 3).tolist(), "state_max": np.round(hi, 3).tolist(),
                                "action_min": np.round(alo, 3).tolist(), "action_max": np.round(ahi, 3).tolist()},
               "camera_dt": cam_dt, "lerobot_loader": lr}
    return {"checks": checks, "details": details}


# ---------------------------------------------------------------- visual samples (5 random frames, seed 1)

def visual_samples(out, inc, work, seed=1, count=5):
    import av
    from PIL import Image, ImageDraw
    rng = random.Random(seed)
    picks = []
    for _ in range(count):
        r = rng.choice(inc)
        picks.append((r, rng.randrange(len(r["t"]))))
    need = {}
    for r, i in picks:
        for c in CAMS:
            need.setdefault((r["index"], c), set()).add(i)
    got = {}
    for (k, c), idxs in need.items():
        r = next(x for x in inc if x["index"] == k)
        cont = av.open(video_path(out, c, r["file_index"]))
        try:
            mx, j = max(idxs), 0
            for fr in cont.decode(video=0):
                if j in idxs:
                    got[(k, c, j)] = fr.to_ndarray(format="rgb24")
                j += 1
                if j > mx:
                    break
        finally:
            cont.close()
    lines = []
    for s, (r, i) in enumerate(picks, 1):
        canvas = Image.new("RGB", (960, 290), "white")
        draw = ImageDraw.Draw(canvas)
        for x, c in enumerate(CAMS):
            tile = Image.fromarray(got[(r["index"], c, i)]).resize((320, 240))
            canvas.paste(tile, (x * 320, 50))
            draw.text((x * 320 + 4, 34), "%s (lag %+.1f ms)" % (c, LAG_S[c] * 1000), fill="black")
        draw.text((4, 6), "sample %d: %s row %d t=%.3f s  colour %s  plate %s  %s" % (
            s, r["key"], i, i / FPS, r["colour"], r["plate"], r["od10"]), fill="black")
        draw.text((4, 274), r["task"], fill="black")
        path = os.path.join(work, "visual", "sample_%02d.png" % s)
        canvas.save(path)
        lines.append({"sample": s, "episode": r["key"], "row": i, "t_s": round(i / FPS, 3), "png": path})
    write_text(os.path.join(work, "visual", "samples.json"), json.dumps(lines, indent=1))
    log("visual samples: %s" % [x["png"] for x in lines])


# ---------------------------------------------------------------- tables, validation.json, data card, STATUS

def write_tsvs(work, recs, ep_no_map):
    def dump(path, fields, rows):
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, delimiter="\t", lineterminator="\n")
            w.writerow(fields)
            for row in rows:
                w.writerow([str(row.get(f, "")).replace("\t", " ").replace("\n", " ") for f in fields])

    rows = []
    for r in recs:
        frames = len(r["t"]) if r["t"] is not None else 0
        rows.append({"run": r["run"], "episode": r["episode"], "ep_no": ep_no_map.get(r["key"], ""),
                     "status": r["status"], "tier": r["tier"], "verified_by": r["verified_by"],
                     "visual_outcome": r["visual_outcome"], "colour": r["colour"], "plate": r["plate"],
                     "od10": r["od10"], "task": r["task"], "needs_review": r["needs_review"],
                     "decision_correct": r["decision_correct"], "decision_confidence": r["decision_confidence"],
                     "rows": r["rows"], "frames": frames,
                     "duration_s": round(frames / FPS, 3) if frames else "",
                     "joint_gap_max_s": "" if r["joint_gap_max_s"] is None else r["joint_gap_max_s"],
                     "loop_success": r["loop_success"], "loop_label": r["loop_label"], "loop_task": r["loop_task"],
                     "loop_failure": r["loop_failure"], "reason": r["reason"], "note": r["note"]})
    dump(os.path.join(work, "episodes.tsv"), EP_FIELDS, rows)
    dump(os.path.join(work, "excluded.tsv"), EX_FIELDS,
         [row for row, r in zip(rows, recs) if r["status"] != "included"])


def write_docs(root, out, work, recs, inc, tasks, total, validation, reused, encoded, t_begin, missing_visual):
    checks = validation["checks"]
    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    n_warn = sum(1 for c in checks if c["status"] == "WARN")
    n_pass = sum(1 for c in checks if c["status"] == "PASS")
    excluded = [r for r in recs if r["status"] != "included"]
    by_cat = {}
    for r in excluded:
        by_cat[category(r["reason"])] = by_cat.get(category(r["reason"]), 0) + 1
    placed_but_excluded = [r["key"] for r in excluded if r["visual_outcome"] == "placed"]
    loop_only_success = [r["key"] for r in excluded if r["verified_by"] == "loop" and r["loop_success"] == "True"]
    review = [r for r in inc if r["needs_review"]]
    dur = (total or 0) / FPS
    status = "DONE"
    if n_fail or placed_but_excluded or missing_visual or not inc:
        status = "PARTIAL"
    stamp = utc_now()
    summary = ("%d visually verified successful episodes (%d frames, %.1f s; 3 cameras AV1 30 fps); %d candidate "
               "episodes excluded with reasons; checks %d PASS %d WARN %d FAIL" % (
                   len(inc), total or 0, dur, len(excluded), n_pass, n_warn, n_fail))
    cam_files = sorted({os.path.basename(p) for r in inc for c in CAMS for (p, _s, _e) in r["cover"][c]})
    per_ep = []
    for r in inc:
        per_ep.append({"key": r["key"], "ep_no": r["ep_no"], "index": r["index"], "task": r["task"],
                       "tier": r["tier"], "colour": r["colour"], "plate": r["plate"], "od10": r["od10"],
                       "frames": len(r["t"]), "file_index": r["file_index"], "needs_review": bool(r["needs_review"]),
                       "decision_correct": r["decision_correct"], "decision_confidence": r["decision_confidence"],
                       "note": r["note"], "cam_dt_max_s": {c: round(r["cam_stats"][c]["dt_max_s"], 6) for c in CAMS}})
    write_text(os.path.join(work, "validation.json"), json.dumps({
        "stage": "e1-lerobot-v1", "utc": stamp, "status": status, "dataset": out, "script": SCRIPT_VERSION,
        "episodes_included": len(inc), "frames_total": total or 0, "episodes_excluded": len(excluded),
        "encoded_this_run": encoded, "reused_this_run": reused, "checks": checks, "details": validation["details"],
        "per_episode": per_ep, "excluded_by_category": by_cat, "visual_placed_but_excluded": placed_but_excluded,
        "loop_success_not_verified": loop_only_success, "visual_entries_without_folder": missing_visual,
        "camera_files_used": cam_files, "lags_s": LAG_S, "seconds": round(time.time() - t_begin, 1)},
        indent=1, default=str))

    lines = ["# Data card: ours-lerobot v1 (DATA-LEROBOT E1)", "",
             "Dataset: `%s` (LeRobot v3.0, Studio layout, built %s by %s)." % (out, stamp, SCRIPT_VERSION), "",
             "## What is in it",
             "- %d episodes, %d frames at 30 fps (%.1f s), %d tasks, three AV1 cameras per episode (top, close, wrist)." % (
                 len(inc), total or 0, dur, len(tasks or [])),
             "- Included only when a visual check confirms the brick was placed on the plate the loop chose and the "
             "OD-10 decision is correct (VISUAL table in the script, from P3-LABEL-OURS V1). Loop-only successes are "
             "not included (%d of them so far)." % len(loop_only_success),
             "- Task text: %s." % ("; ".join('"%s"' % t for t in (tasks or [])) or "none"),
             "- Colour and plate: %s." % (", ".join("%s on %s (OD-10 %s)" % (r["colour"], r["plate"], r["od10"])
                                                    for r in inc) or "none"),
             "- Camera files used: %s." % (", ".join(cam_files) or "none"), "",
             "## Per-episode labels (meta/episodes columns ours_*)"]
    if inc:
        lines += ["| episode | ep no | task | tier | needs_review | decision confidence | frames | note |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in inc:
            lines.append("| %s | %d | %s | %s | %s | %s | %d | %s |" % (
                r["key"], r["ep_no"], r["task"], r["tier"], "yes" if r["needs_review"] else "no",
                r["decision_confidence"], len(r["t"]), r["note"].replace("|", "/")))
    lines += ["", "Columns: ours_ep_no, ours_run, ours_episode, ours_colour, ours_plate, ours_od10_label, "
              "ours_tier (V visual, L loop-only), ours_verified_by, ours_decision_correct, "
              "ours_decision_confidence (high, med, low), ours_needs_review, ours_loop_task, ours_note.",
              "- needs_review is true when the visual confidence is below high or when a tape or orientation item "
              "could not be confirmed at 640x480.", "",
              "## Clock and alignment"]
    for c in CAMS:
        lines.append("- %s (%s): lag applied %s. The camera frame for grid time t is the frame nearest t + lag." % (
            c, CAM_ROLE[c], LAG_TEXT[c]))
    lines += ["- Joints: linear interpolation of pos (observation.state) and goal (action) onto a 30 fps grid from the "
              "rows between t_start and t_end of episode.json.",
              "- Largest grid-row to camera-frame distance per camera (camera clock): %s." % json.dumps(
                  {c: round(max(r["cam_stats"][c]["dt_max_s"] for r in inc), 4) for c in CAMS}) if inc else
              "- No included episodes, so no camera distances.", "",
              "## Excluded candidates",
              "- %d of %d candidates excluded; by category: %s. Full list with reasons: excluded.tsv." % (
                  len(excluded), len(recs), ", ".join("%s %d" % (k, v) for k, v in sorted(by_cat.items())) or "none"),
              "- Visually placed but excluded for a data reason: %s." % (", ".join(placed_but_excluded) or "none"),
              "- Visual entries without an episode folder: %s." % (", ".join(missing_visual) or "none"), "",
              "## Known issues"]
    issues = [
        "Close camera clock is not determined; 0 ms is applied (flagged in every row's camera metadata).",
        "Wrist offset conflict: P2-D-VLA1 open issue R4 (wrist alignment unresolved) reports wrist offsets of -103 and "
        "-161 ms; P3-VLA-CLOCK measured +7.5 ms (median, 15 of 15 episodes reliable). This build uses +7.5 ms. The "
        "conflict is open and needs a decision before training.",
        "Top offset +21.5 ms is the P3-VLA-CLOCK median (11 of 15 reliable; any_1731 ep_0001 and ep_0002 and any_1740 "
        "ep_0001 and ep_0004 are not reliable there).",
        "%d of %d included episodes have needs_review set (see the table above)." % (len(review), len(inc)),
        "Loop colour names disagree with the visual check for: %s. The task text uses the visual colour name." % (
            ", ".join(r["key"] for r in inc if "loop colour" in r["note"]) or "none"),
        "Monitor files recorded at 15 fps (relative clock, before 17:03:51Z) and wrist segments on a relative clock "
        "are not used: the coverage check excludes any episode that touches them.",
        "The live monitor files (from 20:53:47Z) are still recording; the coverage check excludes an episode whose "
        "window the file does not reach.",
        "Not imported into Physical AI Studio (not requested). The LeRobot loader check is in validation.json.",
    ]
    lines += ["%d. %s" % (i, t) for i, t in enumerate(issues, 1)]
    lines += ["", "## Re-run",
              "`nice -n 19 env OMP_NUM_THREADS=4 <R>/envs/lerobot-env/bin/python -I <R>/work/DATA-LEROBOT/E1/build_lerobot.py`. "
              "Episodes whose episode.json, joints.csv and VISUAL entry are unchanged reuse their videos; new visual "
              "verdicts are encoded and appended with the next stable episode number. Dataset tables are rebuilt "
              "each run."]
    write_text(os.path.join(work, "data_card.md"), "\n".join(lines) + "\n")

    st = ["%s %s %s" % (status, date_u(), summary), "",
          "Dataset: %s. Working files: %s." % (out, work),
          "Included: %s." % ("; ".join("%s (%s on %s)" % (r["key"], r["colour"], r["plate"]) for r in inc) or "none"),
          "Excluded: %d (%s)." % (len(excluded), ", ".join("%s %d" % (k, v) for k, v in sorted(by_cat.items())) or "none"),
          "Visually placed but excluded for a data reason: %s." % (", ".join(placed_but_excluded) or "none"),
          "Loop successes not visually verified: %s." % (", ".join(loop_only_success) or "none"),
          "Checks: %d PASS, %d WARN, %d FAIL (validation.json)." % (n_pass, n_warn, n_fail),
          "Open: close clock not determined (0 ms); wrist offset conflict (P2 R4 vs P3-VLA-CLOCK); needs_review on %d of %d."
          % (len(review), len(inc)),
          "Requests to Opus: decide the wrist offset for training; review the visual entries for the loop-only "
          "successes listed above before they can be included."]
    write_text(os.path.join(work, "STATUS.md"), "\n".join(st) + "\n")
    log("STATUS %s" % status)
    log(summary)


# ---------------------------------------------------------------- build

def build(args):
    global LOG_FH
    root = os.path.abspath(args.root)
    out = os.path.abspath(args.out or os.path.join(root, OUT_REL))
    work = os.path.abspath(args.work or os.path.join(root, WORK_REL))
    if not out.startswith(os.path.join(root, "data", "ours-lerobot") + os.sep) or "studio" in out or "curated" in out:
        sys.exit("refusing --out %s" % out)
    os.makedirs(os.path.join(work, "visual"), exist_ok=True)
    LOG_FH = open(os.path.join(work, "build.log"), "a", encoding="utf-8")
    t_begin = time.time()
    log("start %s root=%s out=%s work=%s encode=%s include_loop_only=%s" % (
        SCRIPT_VERSION, root, out, work, not args.no_encode, args.include_loop_only))

    runs, recs = scan_episodes(root, args.include_loop_only)
    keys = {r["key"] for r in recs}
    missing_visual = sorted(k for k in VISUAL if k not in keys)
    log("candidates %d in runs %s; VISUAL entries without a folder: %s" % (len(recs), runs, missing_visual or "none"))

    inc = [r for r in recs if r["status"] == "included"]
    if inc:
        media = load_media(root)
        for r in inc:
            files, why = coverage(r, media)
            if files is None:
                r["status"], r["reason"] = "excluded", "camera coverage: " + why
            else:
                r["cover"] = files
        inc = [r for r in recs if r["status"] == "included"]
    log("included after camera coverage: %d" % len(inc))

    state = read_state(work)
    for r in inc:
        if r["key"] not in state["ep_no"]:
            state["ep_no"][r["key"]] = state["next_ep"]
            state["next_ep"] += 1
        r["ep_no"] = state["ep_no"][r["key"]]
    ep_no_map = {r["key"]: state["ep_no"][r["key"]] for r in inc}

    if args.no_encode:
        write_tsvs(work, recs, ep_no_map)
        log("no-encode: episodes.tsv and excluded.tsv written; dataset, validation and data card not rebuilt")
        return

    reused = encoded = 0
    tpl_root = os.path.join(root, TEMPLATE_REL)
    tpl_info = None
    if inc:
        with open(os.path.join(tpl_root, "meta/info.json"), encoding="utf-8") as fh:
            tpl_info = json.load(fh)
        width = int(tpl_info["features"][CAM_KEY["close"]]["shape"][1])
        height = int(tpl_info["features"][CAM_KEY["close"]]["shape"][0])
        from lerobot.configs.video import VideoEncoderConfig
        options = VideoEncoderConfig().get_codec_options(SVT_LP, as_strings=True)
        for r in inc:
            cached = state["episodes"].get(r["key"])
            fi_known = state["file_index"].get(r["key"])
            if (cached and cached.get("fingerprint") == r["fingerprint"] and cached.get("lags") == LAG_S
                    and fi_known is not None
                    and all(os.path.isfile(os.path.join(out, cached["cams"][c]["mp4"])) for c in CAMS)):
                r["file_index"], r["cam_stats"] = fi_known, cached["cams"]
                reused += 1
                log("reuse %s (file %03d)" % (r["key"], fi_known))
                continue
            ok, why, results, fi = encode_record(r, out, state, options, width, height)
            if not ok:
                r["status"], r["reason"] = "excluded", why
                log("EXCLUDE %s: %s" % (r["key"], why))
                write_state(work, state)
                continue
            state["episodes"][r["key"]] = {
                "fingerprint": r["fingerprint"], "lags": LAG_S, "file_index": fi, "encoded_utc": utc_now(),
                "cams": {c: dict({k: v for k, v in results[c].items() if k != "stats"}, stats=results[c]["stats"],
                                 mp4=os.path.relpath(video_path(out, c, fi), out)) for c in CAMS}}
            write_state(work, state)
            r["file_index"], r["cam_stats"] = fi, state["episodes"][r["key"]]["cams"]
            encoded += 1
            log("encoded %s (file %03d), max grid-to-frame distance %s" % (
                r["key"], fi, {c: round(results[c]["dt_max_s"], 4) for c in CAMS}))
        inc = [r for r in recs if r["status"] == "included"]
        ep_no_map = {r["key"]: state["ep_no"][r["key"]] for r in inc}
        keep = {r["key"] for r in inc}
        for key in sorted(set(state["episodes"]) - keep):
            for c in CAMS:
                p = os.path.join(out, state["episodes"][key]["cams"][c]["mp4"])
                if os.path.isfile(p):
                    os.remove(p)
            del state["episodes"][key]
            log("prune %s (no longer included)" % key)
        write_state(work, state)

    write_tsvs(work, recs, ep_no_map)
    tasks, total = None, None
    validation = {"checks": [], "details": {}}
    if inc:
        tasks, total = write_dataset(out, inc, tpl_info, tpl_root)
        validation = validate(out, inc, recs, total, tasks)
        visual_samples(out, inc, work)
    else:
        log("no included episodes: dataset not written")
    log("encoded %d, reused %d" % (encoded, reused))
    write_docs(root, out, work, recs, inc, tasks, total, validation, reused, encoded, t_begin, missing_visual)


def main():
    ap = argparse.ArgumentParser(description="Build the ours-lerobot v1 dataset from visually verified sort episodes.")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--work", default=None)
    ap.add_argument("--no-encode", action="store_true", help="classify only: write episodes.tsv and excluded.tsv")
    ap.add_argument("--include-loop-only", action="store_true",
                    help="also include loop-success episodes without a VISUAL entry (not recommended)")
    args = ap.parse_args()
    try:
        build(args)
    except SystemExit:
        raise
    except Exception:
        tb = traceback.format_exc()
        work = os.path.abspath(args.work or os.path.join(os.path.abspath(args.root), WORK_REL))
        os.makedirs(work, exist_ok=True)
        write_text(os.path.join(work, "STATUS.md"), "PARTIAL %s build crashed; see build.log\n\n%s\n" % (
            date_u(), tb[-3000:]))
        log("CRASH\n" + tb)
        raise


if __name__ == "__main__":
    main()
