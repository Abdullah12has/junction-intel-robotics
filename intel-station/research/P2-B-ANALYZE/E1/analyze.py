#!/usr/bin/env python3
"""P2-B-ANALYZE analyst E1, analyze.py v2: evidence on the pick-anywhere runs any_1731 and any_1740.

Re-run from the project root (reads only; writes episodes.csv, estops.csv and report.md beside this file):
  OMP_NUM_THREADS=4 nice -n 19 envs/data-env/bin/python -I work/P2-B-ANALYZE/E1/analyze.py
Inputs under R (read only): logs/sort_any_<run>.out (loop log), logs/sort/<run>/cycles.jsonl (one line per save_ep
call), logs/sort/<run>/ep_*/ (episode.json, joints.csv), logs/monitor/hand_watch.log, the names of logs/monitor/estop_*.jpg,
app/tools/scan_calib.json, and kin.fk (URDF forward kinematics).
Estop classes come only from MANUAL_LABELS, which were set by viewing each image. Every other number is computed here.
"""
import ast
import collections
import csv
import datetime
import glob
import hashlib
import json
import math
import os
import re
import statistics
import sys

R = "/home/intel/intel-junction-2026"
HERE = os.path.dirname(os.path.abspath(__file__))
EPOCH0 = 1794614400  # 2026-11-14T00:00:00Z. Check: estop_top_1794677552.jpg = 17:32:32Z = loop STOP of 1731 ep 1.
RUNS = ["any_1731", "any_1740"]
MON = R + "/logs/monitor"
HAND_LOG = MON + "/hand_watch.log"
PERSIST = {"top": 1, "close": 2, "wrist": 2}  # PERSIST table in hand_watch.py as read
SCAN_CALIB = R + "/app/tools/scan_calib.json"
SAVE_GAP_S = 5.0     # a loop attempt line and its cycles.jsonl t_start agree within this many seconds
EVENT_GAP_S = 20.0   # loop estop lines closer together than this belong to one estop event
HELD_MIN = 1.1       # grip reading at or above this = brick held (docs/PICK-ANYWHERE.md)
sys.path.insert(0, R + "/app")  # pick_v2 package lives here (imported by kin.py)
sys.path.insert(0, R + "/app/tools")
import kin  # noqa: E402  (kin.fk and kin.MAT_Z only; the import reads files only: the URDF and app/motion/project_robots.json; no device, port, camera or write)

# estop image file -> (class, phrase). Set by viewing every image (laptop-log.md entries 36, 38, 40, 41, 42).
# REAL_HAND: a person's hand is in the frame. FALSE_POSITIVE: no hand in the frame. UNKNOWN: not decidable from it.
MANUAL_LABELS = {
    "estop_top_1794673145.jpg": ("REAL_HAND", "hand at top-left edge by laptop, outside mat"),
    "estop_close_1794673200.jpg": ("FALSE_POSITIVE", "no hand; beige/white bricks and red jaw tips"),
    "estop_top_1794673200.jpg": ("FALSE_POSITIVE", "no hand in view; arm and bricks only"),
    "estop_top_1794675022.jpg": ("REAL_HAND", "hand and forearm over blue plate"),
    "estop_wrist_1794675172.jpg": ("FALSE_POSITIVE", "no hand; beige brick beside red jaws over green plate"),
    "estop_top_1794677184.jpg": ("REAL_HAND", "finger pointing at blue plate from top edge"),
    "estop_top_1794677552.jpg": ("REAL_HAND", "arm and hand over blue plate, left side"),
    "estop_top_1794677636.jpg": ("REAL_HAND", "hand reaching to orange brick by blue plate"),
    "estop_close_1794677636.jpg": ("REAL_HAND", "blurred hand at left edge toward orange brick"),
    "estop_top_1794677654.jpg": ("REAL_HAND", "hand at top edge above arm"),
    "estop_top_1794677779.jpg": ("REAL_HAND", "hand at top edge by arm"),
    "estop_top_1794677889.jpg": ("REAL_HAND", "hand at top right by orange gripper piece"),
    "estop_top_1794677902.jpg": ("REAL_HAND", "hand at top centre by arm"),
    "estop_top_1794677923.jpg": ("REAL_HAND", "motion-blurred hand at top edge by arm"),
    "estop_close_1794677939.jpg": ("REAL_HAND", "fingers at top-left edge by blue plate"),
    "estop_top_1794677939.jpg": ("REAL_HAND", "hand at top edge over mat near arm"),
    "estop_top_1794677947.jpg": ("REAL_HAND", "hand at top right by arm"),
    "estop_top_1794677973.jpg": ("REAL_HAND", "hand at top edge reaching toward arm"),
    "estop_top_1794678371.jpg": ("REAL_HAND", "arm and pointing hand over blue plate bricks"),
    "estop_top_1794678583.jpg": ("REAL_HAND", "hand at top-left over blue plate"),
    "estop_top_1794679239.jpg": ("REAL_HAND", "hand touching orange brick in blue plate"),
    "estop_close_1794679304.jpg": ("FALSE_POSITIVE", "no hand; white brick held in red jaws"),
    "estop_close_1794679373.jpg": ("FALSE_POSITIVE", "no hand; yellow and red bricks"),
    "estop_close_1794679818.jpg": ("FALSE_POSITIVE", "no hand; yellow and red bricks"),
    "estop_close_1794679859.jpg": ("FALSE_POSITIVE", "no hand; white brick in red jaws entering view"),
    "estop_close_1794679874.jpg": ("FALSE_POSITIVE", "no hand; white brick in red jaws above mat"),
    "estop_close_1794679987.jpg": ("FALSE_POSITIVE", "no hand; white brick in red jaws above yellow brick"),
    "estop_wrist_1794679388.jpg": ("FALSE_POSITIVE", "no hand; white brick in red jaws over blue plate"),
    "estop_wrist_1794679389.jpg": ("FALSE_POSITIVE", "no hand; white brick in red jaws over blue plate"),
    "estop_wrist_1794679390.jpg": ("FALSE_POSITIVE", "no hand; white brick in red jaws over blue plate"),
    "estop_wrist_1794679391.jpg": ("FALSE_POSITIVE", "no hand; white brick in red jaws over blue plate"),
    "estop_wrist_1794679392.jpg": ("FALSE_POSITIVE", "no hand; white brick in red jaws over blue plate"),
    "estop_wrist_1794679393.jpg": ("FALSE_POSITIVE", "no hand; white brick in red jaws over blue plate"),
    "estop_wrist_1794679394.jpg": ("FALSE_POSITIVE", "no hand; white brick in red jaws over blue plate"),
    "estop_wrist_1794679395.jpg": ("FALSE_POSITIVE", "no hand; white brick in red jaws over blue plate"),
    "estop_wrist_1794679396.jpg": ("FALSE_POSITIVE", "no hand; white brick in red jaws over blue plate"),
    "estop_wrist_1794679397.jpg": ("FALSE_POSITIVE", "no hand; white brick in red jaws over blue plate"),
    "estop_wrist_1794679398.jpg": ("FALSE_POSITIVE", "no hand; white brick in red jaws over blue plate"),
    "estop_wrist_1794679399.jpg": ("FALSE_POSITIVE", "no hand; white brick in red jaws over blue plate"),
    "estop_wrist_1794679401.jpg": ("UNKNOWN", "white brick in jaws; faint skin-like shape at bottom edge, inconclusive"),
    "estop_wrist_1794679402.jpg": ("UNKNOWN", "white brick in jaws; faint skin-like shape at bottom edge, inconclusive"),
}

STAMP_RE = re.compile(r"^(\d\d:\d\d:\d\d)Z (.*)$")
GOING_RE = re.compile(r"^ep (\d+): (\d+) brick\(s\) in view; going for \[(.*?)\] est \[(.*?)\] mm \(d \[(.*?)\]\)$")
CARRY_RE = re.compile(r"^holding a brick at start \(gripper ([\d.]+)\): inspect, decide, place$")
EP_RE = re.compile(r"^ep (\d+): (.*)$")
SERVO_RE = re.compile(r"^servo it(\d+): (?:(\w+) )?brick \[(.*?)\] err \[(.*?)\] short ([\d.]+)$")
YAW_RE = re.compile(r"^yaw (-?[\d.]+) deg -> roll (-?[\d.]+)$")
INTERRUPT_RE = re.compile(r"^(recovery|carry-over) interrupted \((.*)\)(?:; retry (\d+))?$")
STOP_RE = re.compile(r"^STOP (.*?)(?:; .*)?$")
STATION_RE = re.compile(r"^(\w+) (\w+) (\[.*\]) tape (\S+)$")
DECISION_RE = re.compile(r"^(\w+) (2x2|not_2x2) -> (GOOD|DEFECT) -> (blue|green) \((\w+); (.*)\)$")
DONE_RE = re.compile(r"^(GOOD|DEFECT) -> (blue|green) in ([\d.]+) s (\{.*\})$")
HELD_RE = re.compile(r"^held brick -> (GOOD|DEFECT) (blue|green) \((\w+); (.*)\)$")
HAND_RE = re.compile(r"^(\d\d:\d\d:\d\d)Z HAND\? view=(top|close|wrist) blob=(\d+)px active=(True|False)(?: hits=(\d+))?$")
IMG_RE = re.compile(r"^estop_(top|close|wrist)_(\d+)\.jpg$")
PLACED_RE = re.compile(r"placed on (\w+)")

COLS = ["run", "ep", "attempt_idx", "session", "kind", "source", "t_start_utc", "t_end_utc", "t_log_utc",
        "json_state", "outcome", "outcome_detail", "failure_category", "failure_text",
        "estop_in_attempt", "estop_phase", "session_continued",
        "scan_px_x", "scan_px_y", "scan_dist_px", "scan_dist_mm", "n_candidates",
        "est_x_mm", "est_y_mm", "hover_x_mm", "hover_y_mm", "est_hover_err_mm",
        "servo_pre_yaw_n", "servo_post_yaw_n", "servo_retry_n", "servo_first_err_px", "servo_final_err_px",
        "approach_to_first_servo_s", "yaw_deg", "yaw_roll",
        "hover_tip_size_mm", "fk_tip_mm", "tip_diff_mm",
        "grip_closed", "grip_lifted", "grip_class", "held_size_ratio", "open_short", "shape", "shape_why",
        "pre_close_verdict", "pre_close_reasons", "station_used", "station_verdict",
        "label_loop", "label_det2", "label", "label_source", "label_conflict", "colour", "plate", "final_plate",
        "decision_conf", "decision_reasons", "views_used",
        "t_wait_s", "t_pick_s", "t_inspect_s", "t_place_s", "t_return_s", "t_cycle_s",
        "joints_window", "joints_rows", "joints_span_s", "t_close_rel_s", "moving_pre_close_s",
        "n_bursts_pre_close", "last_pre_burst_s", "gap_move_to_close_s", "close_s", "yaw_burst_s",
        "lift_start_after_close_s", "lift_burst_s", "grip_plateau_joints", "cams_top", "dir_files"]
JOINT_KEYS = ["t_close_rel_s", "moving_pre_close_s", "n_bursts_pre_close", "last_pre_burst_s", "gap_move_to_close_s",
              "close_s", "yaw_burst_s", "lift_start_after_close_s", "lift_burst_s", "grip_plateau_joints",
              "joints_span_s"]
ESTOP_COLS = ["run", "event_id", "row_source", "session", "loop_line", "t_utc", "kind", "ep", "phase", "text",
              "hand_view", "hand_blob", "hand_active", "hand_dt_s", "trigger_view", "trigger_hits", "trigger_dt_s",
              "image_file", "image_dt_s", "image_class", "image_note"]


def epoch_of(hhmmss):
    h, m, s = (int(x) for x in hhmmss.split(":"))
    return EPOCH0 + 3600 * h + 60 * m + s


def iso(ep):
    if ep is None or ep == "":
        return ""
    return datetime.datetime.fromtimestamp(float(ep), datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hhmm(ep):
    return "" if ep in (None, "") else iso(ep)[11:19]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def nums(s):
    # drop the 'float64' type name first, otherwise its digits are read as numbers
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", s.replace("float64", ""))]


def rnd(x, nd=1):
    return "" if x is None else round(float(x), nd)


def summ(xs):
    xs = [float(x) for x in xs if x not in (None, "")]
    if not xs:
        return {"n": 0, "med": None, "min": None, "max": None}
    return {"n": len(xs), "med": round(statistics.median(xs), 2), "min": round(min(xs), 2), "max": round(max(xs), 2)}


def vals(rows, key):
    return [r[key] for r in rows if r.get(key) not in ("", None)]


def pearson(xs, ys):
    pts = [(float(x), float(y)) for x, y in zip(xs, ys) if x not in ("", None) and y not in ("", None)]
    n = len(pts)
    if n < 3:
        return None
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    syy = sum((p[1] - my) ** 2 for p in pts)
    if sxx == 0 or syy == 0:
        return None
    return round(sxy / math.sqrt(sxx * syy), 3)


def phase_hint(text):
    s = text or ""
    m = re.search(r"move_rel \[(.*?)\] -> estop", s)
    if m:
        parts = m.group(1).split(", ")
        zz = nums(parts[-1]) if parts else []
        if not zz:
            return "move_rel"
        if abs(zz[0]) < 1e-9:
            return "xy move (approach or servo step)"
        return "ascent or lift (dz>0)" if zz[0] > 0 else "descent (dz<0)"
    if "tmove" in s:
        return "transit move (tmove)"
    if "move -> estop" in s or "move ->" in s:
        return "transit or plate move (move)"
    if "home_to" in s or "blue_wp" in s or "carry" in s:
        return "carry transit"
    return "other" if s else ""


def servo_stats(a):
    main = [s for s in a["servo"] if s["phase"] != "retry"]
    out = {"servo_pre_yaw_n": sum(1 for s in a["servo"] if s["phase"] == "pre_yaw"),
           "servo_post_yaw_n": sum(1 for s in a["servo"] if s["phase"] == "post_yaw"),
           "servo_retry_n": sum(1 for s in a["servo"] if s["phase"] == "retry")}
    if main:
        out["servo_first_err_px"] = rnd(math.hypot(main[0]["err"][0], main[0]["err"][1]))
        out["servo_final_err_px"] = rnd(math.hypot(main[-1]["err"][0], main[-1]["err"][1]))
    return out


# ----------------------------------------------------------------- hand watchdog log and estop images
def load_hand():
    rows, unparsed = [], []
    with open(HAND_LOG, encoding="utf-8", errors="replace") as f:
        for n, raw in enumerate(f, 1):
            m = HAND_RE.match(raw.rstrip("\r\n"))
            if not m:
                unparsed.append(n)
                continue
            hh, view, blob, act, hits = m.groups()
            hits = int(hits) if hits is not None else None
            active = act == "True"
            rows.append({"n": n, "t": epoch_of(hh), "view": view, "blob": int(blob), "active": active,
                         "hits": hits, "trigger": bool(active and hits is not None and hits >= PERSIST[view])})
    return rows, unparsed


def load_images():
    out = []
    for p in sorted(glob.glob(MON + "/estop_*.jpg")):
        m = IMG_RE.match(os.path.basename(p))
        if m:
            out.append({"file": os.path.basename(p), "view": m.group(1), "t": int(m.group(2)), "hand": None})
    return out


def link_images(images, hand):
    # an image is written 0..2 s after its trigger; prefer rows that met the PERSIST rule
    for im in images:
        best = None
        for h in hand:
            if h["view"] != im["view"]:
                continue
            dt = im["t"] - h["t"]
            if -1 <= dt <= 2:
                key = (not h["trigger"], abs(dt))
                if best is None or key < best[0]:
                    best = (key, h)
        if best:
            im["hand"] = best[1]


def nearest_row(rows, ref, lo, hi):
    # the row whose time t satisfies lo <= ref - t <= hi, nearest in time; returns (row, ref - t)
    best = None
    for x in rows:
        dt = ref - x["t"]
        if lo <= dt <= hi and (best is None or abs(dt) < abs(best[1])):
            best = (x, dt)
    return best


def hand_bins(hand, t0, t1, step=300):
    counts = collections.Counter()
    for h in hand:
        if t0 <= h["t"] < t1:
            counts[int((h["t"] - t0) // step)] += 1
    n = int((t1 - t0) // step)
    return ["%s %d" % (hhmm(t0 + i * step)[:5], counts.get(i, 0)) for i in range(n)]


# ----------------------------------------------------------------- loop logs
def read_lines(run):
    out, last = [], None
    with open(R + "/logs/sort_any_" + run + ".out", encoding="utf-8", errors="replace") as f:
        for n, raw in enumerate(f, 1):
            ln = raw.rstrip("\r\n")
            m = STAMP_RE.match(ln)
            if m:
                last = m.group(1)
                out.append({"n": n, "t": epoch_of(last), "text": m.group(2), "stamped": True})
            else:
                out.append({"n": n, "t": epoch_of(last) if last else None, "text": ln, "stamped": False})
    return out


def new_att(run, sess, kind, t, n, **kw):
    return {"run": run, "session": sess, "kind": kind, "k": kw.get("k"), "t_log": t, "line": n,
            "n_cand": int(kw["n_cand"]) if kw.get("n_cand") else None,
            "log_px": nums(kw["px"]) if kw.get("px") else None,
            "log_est": nums(kw["est"]) if kw.get("est") else None,
            "grip_log": kw.get("grip"), "servo": [], "yaws": [], "station": None, "decision": None, "held": None,
            "done": None, "approach_fail": None, "grasp_fail": None, "grasp_missed": None, "station_fail": False,
            "stop": None, "stop_t": None, "stop_estop": False, "stop_request": False, "crash": None, "waits": 0,
            "cleared": 0, "raised": 0, "missed_seen": False, "yaw_seen": False, "events": [],
            "recovered": False, "attempt_idx": 0, "rec": None, "fjson": {}, "folder": None, "source": "",
            "json_state": ""}


def make_estop(run, sess, L, kind, cur, text):
    return {"run": run, "session": sess, "line": L["n"], "t": L["t"], "stamped": L["stamped"], "kind": kind,
            "ep": cur["k"] if cur else None, "text": text, "phase": phase_hint(text)}


def parse_run(run):
    atts, estops, sessions = [], [], []
    notes = {"jpeg_warnings": 0, "scan_map_lines": 0, "jaw_check_lines": 0, "stop_requested_lines": 0,
             "unparsed": [], "old_stop_wording": 0, "recovery_failed_lines": 0, "tape_held_views": 0,
             "crash_lines": 0, "estop_cleared_auto": 0, "waits_outside_attempt": 0}
    cur = None
    sess = 0
    tb = None

    def close_cur():
        nonlocal cur
        if cur is not None:
            atts.append(cur)
        cur = None

    for L in read_lines(run):
        text, t, n = L["text"], L["t"], L["n"]
        if not L["stamped"]:
            if text.startswith("Traceback"):
                tb = {"last": None}
                notes["crash_lines"] += 1
            elif tb is not None and text.strip():
                tb["last"] = text.strip()
            elif text.startswith(("Premature end of JPEG", "Corrupt JPEG")):
                notes["jpeg_warnings"] += 1
            elif text.strip() == "stop requested":
                notes["stop_requested_lines"] += 1
                if cur is not None and cur["done"] is None and cur["approach_fail"] is None and cur["grasp_fail"] is None:
                    cur["stop_request"] = True
            elif text.strip():
                notes["unparsed"].append(str(n) + ": " + text)
            if "-> estop" in text:
                estops.append(make_estop(run, sess, L, "traceback" if tb is not None else "unstamped", cur, text))
                if cur is not None:
                    cur["stop_estop"] = True
                    cur["stop"] = cur["stop"] or text
            continue
        if tb is not None:
            if cur is not None:
                cur["crash"] = tb["last"]
            tb = None
        if text.startswith("start "):
            close_cur()
            sess += 1
            sessions.append({"run": run, "session": sess, "t": t, "text": text})
            continue
        m = GOING_RE.match(text)
        if m:
            close_cur()
            cur = new_att(run, sess, "pick", t, n, k=int(m.group(1)), n_cand=m.group(2), px=m.group(3),
                          est=m.group(4))
            continue
        m = CARRY_RE.match(text)
        if m:
            close_cur()
            cur = new_att(run, sess, "carry_over", t, n, k=None, grip=float(m.group(1)))
            continue
        if text == "loop end" or text.startswith("stopping:"):
            if text.startswith("stopping:") and cur is not None and cur["done"] is None \
                    and cur["approach_fail"] is None and cur["grasp_fail"] is None:
                cur["stop_request"] = True
            close_cur()
            continue
        if text.startswith("scan map"):
            notes["scan_map_lines"] += 1
            continue
        if text.startswith(("jaw check", "jaw gap", "JAW CHECK")):
            notes["jaw_check_lines"] += 1
            continue
        if cur is None:
            if "-> estop" in text:
                estops.append(make_estop(run, sess, L, "outside-attempt", None, text))
            if text.startswith("waiting for a clear workspace"):
                notes["waits_outside_attempt"] += 1
            elif text.startswith("estop cleared automatically"):
                notes["estop_cleared_auto"] += 1
            else:
                notes["unparsed"].append(str(n) + ": " + text)
            continue
        m = EP_RE.match(text)
        if m:
            rest = m.group(2)
            if rest.startswith("APPROACH FAILED: "):
                cur["approach_fail"] = re.sub(r" \(brick at \[.*\] skipped for 5 min\)$", "",
                                              rest[len("APPROACH FAILED: "):])
            elif rest.startswith("grasp missed ("):
                cur["missed_seen"] = True
                cur["grasp_missed"] = rest
            elif rest.startswith("GRASP FAILED: "):
                cur["grasp_fail"] = rest[len("GRASP FAILED: "):]
            elif rest == "station re-pick failed":
                cur["grasp_fail"] = cur["grasp_fail"] or "re-pick at station failed"
            elif rest.startswith("STOP "):
                sm = STOP_RE.match(rest)
                stop_txt = sm.group(1) if sm else rest
                cur["stop"] = stop_txt
                cur["stop_t"] = t
                cur["stop_estop"] = "-> estop" in stop_txt
                if "waiting for a clear workspace, then recovering" in rest:
                    notes["old_stop_wording"] += 1
                estops.append(make_estop(run, sess, L, "STOP", cur, rest))
            elif rest.startswith("station close view "):
                sm = STATION_RE.match(rest[len("station close view "):])
                if sm:
                    cur["station"] = {"colour": sm.group(1), "verdict": sm.group(2), "reasons": sm.group(3),
                                      "tape": sm.group(4)}
                else:
                    cur["events"].append(rest)
            elif rest.startswith("DONE "):
                dm = DONE_RE.match(rest[len("DONE "):])
                if dm:
                    cur["done"] = {"label": dm.group(1), "plate": dm.group(2), "cycle_s": float(dm.group(3)),
                                   "timings": ast.literal_eval(dm.group(4))}
                else:
                    cur["events"].append(rest)
            else:
                dm = DECISION_RE.match(rest)
                if dm:
                    cur["decision"] = {"colour": dm.group(1), "shape": dm.group(2), "label": dm.group(3),
                                       "plate": dm.group(4), "conf": dm.group(5), "reasons": dm.group(6)}
                    if "tape_held_r" in dm.group(6):
                        notes["tape_held_views"] += 1
                else:
                    cur["events"].append(rest)
            continue
        m = SERVO_RE.match(text)
        if m:
            phase = "post_yaw" if cur["yaw_seen"] else ("retry" if cur["missed_seen"] else "pre_yaw")
            cur["servo"].append({"t": t, "it": int(m.group(1)), "colour": m.group(2), "centre": nums(m.group(3)),
                                 "err": nums(m.group(4)), "short": float(m.group(5)), "phase": phase})
            continue
        m = YAW_RE.match(text)
        if m:
            cur["yaw_seen"] = True
            cur["yaws"].append({"t": t, "deg": float(m.group(1)), "roll": float(m.group(2))})
            continue
        m = INTERRUPT_RE.match(text)
        if m:
            if "-> estop" in m.group(2):
                cur["stop_estop"] = True
                cur["stop"] = cur["stop"] or m.group(2)
                cur["stop_t"] = cur["stop_t"] or t
                estops.append(make_estop(run, sess, L, m.group(1) + " interrupted", cur, text))
            else:
                cur["events"].append(text)
            continue
        if text.startswith("recovery failed"):
            notes["recovery_failed_lines"] += 1
        if "-> estop" in text:
            estops.append(make_estop(run, sess, L, "other stamped", cur, text))
            cur["stop_estop"] = True
            cur["stop"] = cur["stop"] or text
            cur["events"].append(text)
            continue
        if text.startswith("raised look"):
            cur["raised"] += 1
            continue
        m = HELD_RE.match(text)
        if m:
            cur["held"] = {"label": m.group(1), "plate": m.group(2), "conf": m.group(3), "reasons": m.group(4)}
            continue
        if text.startswith("held brick put down at the station"):
            cur["station_fail"] = True
            continue
        if text.startswith("waiting for a clear workspace"):
            cur["waits"] += 1
            continue
        if text.startswith("estop cleared automatically"):
            cur["cleared"] += 1
            notes["estop_cleared_auto"] += 1
            continue
        cur["events"].append(text)
    if tb is not None and cur is not None:
        cur["crash"] = tb["last"]
    close_cur()
    for i, a in enumerate(atts):
        a["attempt_idx"] = i + 1
        a["recovered"] = bool(i + 1 < len(atts) and atts[i + 1]["session"] == a["session"])
    return atts, estops, sessions, notes


# ----------------------------------------------------------------- episode folders, save records, matching
def load_dirs(run):
    recs = {}
    for d in glob.glob(R + "/logs/sort/" + run + "/ep_*"):
        mm = re.match(r"^ep_(\d+)$", os.path.basename(d))
        if not mm or not os.path.isdir(d):
            continue
        rec = {"k": int(mm.group(1)), "dir": d, "json": None, "joints": None}
        jp = os.path.join(d, "episode.json")
        if os.path.exists(jp):
            try:
                with open(jp, encoding="utf-8") as f:
                    rec["json"] = json.load(f)
            except (OSError, ValueError):
                rec["json"] = None
        jj = os.path.join(d, "joints.csv")
        if os.path.exists(jj):
            rec["joints"] = jj
        rec["n_files"] = sum(len(fs) for _, _, fs in os.walk(d))
        recs[rec["k"]] = rec
    return recs


def load_cycles(run):
    # one JSON line per save_ep call (append-only file written by sort_any.py)
    recs = []
    with open(R + "/logs/sort/" + run + "/cycles.jsonl", encoding="utf-8") as f:
        for n, raw in enumerate(f, 1):
            if raw.strip():
                d = json.loads(raw)
                d["_line"] = n
                recs.append(d)
    return recs


def session_at(sessions, t):
    cur = 0
    for s in sessions:
        if s["t"] <= t + 1.0:
            cur = s["session"]
    return cur


def att_time(a):
    rc = a.get("rec") or {}
    if rc.get("t_start") is not None:
        return float(rc["t_start"])
    if a.get("line"):
        return float(a["t_log"])
    return float(rc.get("t_end") or 0.0)


def match_all(run, atts, recs, dirs, sessions):
    """Pair each loop attempt with its save record; add save records that have no loop attempt line."""
    used = set()
    for a in atts:
        best = None
        for i, rc in enumerate(recs):
            if i in used or rc.get("t_start") is None:
                continue
            # the two early 1731 saves carry no mode field; they are pick saves like the others
            if a["kind"] == "pick" and (rc.get("mode") not in ("pick_anywhere", None) or rc.get("episode") != a["k"]):
                continue
            if a["kind"] == "carry_over" and rc.get("mode") != "carry_over":
                continue
            if abs(float(rc["t_start"]) - float(a["t_log"])) <= SAVE_GAP_S:
                best = i
                break
        if best is not None:
            used.add(best)
            a["rec"] = recs[best]
            a["source"] = "loop log + cycles.jsonl"
            if a["kind"] == "carry_over":
                a["k"] = recs[best].get("episode")
        else:
            a["source"] = "loop log only (no save record)"
    extra = []
    for i, rc in enumerate(recs):
        if i in used:
            continue
        if rc.get("t_start") is None:
            kind = "wait_stop"
        elif rc.get("mode") == "carry_over":
            kind = "carry_over"
        else:
            kind = "pick"
        a = new_att(run, session_at(sessions, float(rc["t_end"])), kind, float(rc["t_end"]), 0,
                    k=rc.get("episode"))
        a["rec"] = rc
        a["source"] = "cycles.jsonl only (no loop-log attempt line)"
        extra.append(a)
    allatts = atts + extra
    for a in allatts:
        rc = a.get("rec")
        k = a["k"] if a["k"] is not None else (rc or {}).get("episode")
        d = dirs.get(k) if k is not None else None
        J = (d or {}).get("json") or {}
        a["folder"] = d
        a["fjson"] = {}
        if rc is not None:
            if J and abs(float(J.get("t_end") or 0) - float(rc["t_end"])) < 0.01:
                a["fjson"] = J
                a["json_state"] = "folder json = this save"
            elif J:
                a["json_state"] = "folder json = another save (overwritten)"
            else:
                a["json_state"] = "no folder json"
        else:
            a["json_state"] = "not saved (no save record)"
            if J.get("t_end") is not None and a.get("line") and float(J["t_end"]) < a["t_log"]:
                a["json_state"] += "; folder json is from an earlier save"
    allatts.sort(key=att_time)
    for i, a in enumerate(allatts):
        a["attempt_idx"] = i + 1
        a["recovered"] = bool(i + 1 < len(allatts) and allatts[i + 1]["session"] == a["session"])
    return allatts


def joint_metrics(path, t_start):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            try:
                rows.append((float(rec["t"]), [float(rec["goal%d" % j]) for j in range(6)], float(rec["pos5"])))
            except (KeyError, TypeError, ValueError):
                continue
    m = {"joints_rows": len(rows), "joints_window": ""}
    if len(rows) < 5:
        m["joints_window"] = "too few rows"
        return m
    t = [x[0] for x in rows]
    g = [x[1] for x in rows]
    p5 = [x[2] for x in rows]
    m["joints_span_s"] = round(t[-1] - t[0], 2)
    if t_start is None or not (t[0] - 1.0 <= t_start <= t[-1] + 1.0):
        m["joints_window"] = "not used: attempt start outside this joints.csv span"
        return m
    m["joints_window"] = "ok"
    i0 = next(i for i in range(len(t)) if t[i] >= t_start - 1.0)
    i_close, seen_open, last_open = None, False, None
    for i in range(i0, len(rows)):
        if g[i][5] >= 20.0:
            seen_open = True
            last_open = i
        elif seen_open and g[i][5] < 5.0:
            i_close = i
            break
    if i_close is None:
        m["joints_window"] = "ok, no close found"
        return m
    tc = t[i_close]
    m["t_close_rel_s"] = round(tc - t_start, 2)
    m["close_s"] = round(tc - t[last_open], 2) if last_open is not None else None
    bursts = []  # [first moving row, last moving row]; gaps up to 0.3 s are merged
    for i in range(i0 + 1, len(rows)):
        dj = max(abs(g[i][j] - g[i - 1][j]) for j in range(5))
        if dj <= 0.02:
            continue
        if bursts and t[i] - t[bursts[-1][1]] <= 0.3:
            bursts[-1][1] = i
        else:
            bursts.append([i, i])

    def dur(b):
        return t[b[1]] - t[b[0]]

    pre = [b for b in bursts if t[b[1]] <= tc]
    post = [b for b in bursts if t[b[0]] > tc]
    m["n_bursts_pre_close"] = len(pre)
    m["moving_pre_close_s"] = round(sum(dur(b) for b in pre), 2)
    m["last_pre_burst_s"] = round(dur(pre[-1]), 2) if pre else None
    m["gap_move_to_close_s"] = round(tc - t[pre[-1][1]], 2) if pre else None
    yaw = []
    for b in pre:
        roll = sum(abs(g[i][4] - g[i - 1][4]) for i in range(b[0], b[1] + 1))
        other = max(max(abs(g[i][j] - g[i - 1][j]) for j in range(4)) for i in range(b[0], b[1] + 1))
        if roll > 1.0 and other < 0.05:
            yaw.append(b)
    m["yaw_burst_s"] = round(sum(dur(b) for b in yaw), 2) if yaw else None
    if post:
        m["lift_start_after_close_s"] = round(t[post[0][0]] - tc, 2)
        m["lift_burst_s"] = round(dur(post[0]), 2)
    win = [p5[i] for i in range(len(rows)) if tc + 0.3 <= t[i] <= tc + 0.6]
    m["grip_plateau_joints"] = round(statistics.median(win), 2) if win else None
    return m


def outcome_of(a):
    rc = a.get("rec") or {}
    if a["kind"] == "wait_stop":
        return "stopped", "stop requested while waiting (no approach started)"
    if a["kind"] == "carry_over":
        if a.get("crash"):
            return "carry_over", "crash: " + str(a["crash"])
        fail = str(rc.get("failure") or "")
        m = PLACED_RE.search(fail)
        if m:
            return "carry_over", "placed on " + m.group(1)
        if "-> estop" in fail or a.get("stop_estop"):
            return "carry_over", "estop during carry-over"
        if rc.get("success") is True:
            return "carry_over", "placed (save record success=True, no placement text)"
        return "carry_over", "not placed: " + (fail or "no save record")
    done = a.get("done") or {}
    if rc.get("success") is True or done:
        lab = rc.get("label") or done.get("label") or ""
        plate = rc.get("plate") or done.get("plate") or ""
        return "success", "label " + lab + " -> " + plate
    text = str(rc.get("failure") or a.get("approach_fail") or a.get("grasp_fail") or a.get("stop")
               or a.get("crash") or "")
    if "-> estop" in text or a.get("stop_estop"):
        return "estop", text
    if a.get("grasp_fail"):
        return "grasp_fail", text
    if a.get("approach_fail"):
        return "approach_fail", text
    if text:
        return ("stopped" if "stop" in text.lower() else "approach_fail"), text
    if a.get("stop_request"):
        return "stopped", "stop requested mid-attempt"
    return "unknown", "no outcome recorded"


def category(a, outc, detail):
    rc = a.get("rec") or {}
    text = str(rc.get("failure") or a.get("approach_fail") or a.get("grasp_fail") or a.get("stop") or "")
    if outc == "success":
        return "success"
    if outc == "approach_fail":
        if text.startswith("ik fail"):
            return "approach: IK fail"
        if "not visible at hover" in text:
            return "approach: brick not visible at hover"
        if "could not centre" in text:
            return "approach: could not centre at hover"
        if "lost" in text:
            return "approach: brick lost during servo"
        return "approach: other"
    if outc == "grasp_fail":
        if "nothing held" in text:
            return "grasp: nothing held after lift"
        if "re-pick" in text:
            return "grasp: station re-pick failed"
        return "grasp: other"
    if outc == "estop":
        return "estop: " + phase_hint(text)
    if outc == "carry_over":
        if a.get("crash"):
            return "carry-over: crash"
        if detail.startswith("placed"):
            return "carry-over: placed"
        return "carry-over: estop" if "estop" in detail else "carry-over: not placed"
    if outc == "stopped":
        return "stopped: while waiting" if a["kind"] == "wait_stop" else "stopped: other"
    return outc


def build_row(run, a, geo):
    rc = a.get("rec") or {}
    J = a.get("fjson") or {}
    cur = bool(J)
    d = a.get("folder")
    k = a["k"] if a["k"] is not None else rc.get("episode")
    t_start = rc.get("t_start")
    if t_start is None and a["kind"] in ("pick", "carry_over") and a.get("line"):
        t_start = a["t_log"]
    r = {c: "" for c in COLS}
    r["run"] = run
    r["ep"] = "" if k is None else k
    r["attempt_idx"] = a["attempt_idx"]
    r["session"] = a["session"]
    r["kind"] = a["kind"]
    r["source"] = a["source"]
    r["t_start_utc"] = iso(t_start)
    r["t_end_utc"] = iso(rc.get("t_end"))
    r["t_log_utc"] = iso(a["t_log"]) if a.get("line") else ""
    r["json_state"] = a["json_state"]
    outc, detail = outcome_of(a)
    r["outcome"], r["outcome_detail"] = outc, detail
    r["failure_category"] = category(a, outc, detail)
    r["failure_text"] = str(rc.get("failure") or a.get("approach_fail") or a.get("grasp_fail")
                            or a.get("stop") or a.get("crash") or "").strip()
    if a.get("stop_estop"):
        r["estop_in_attempt"] = "y"
        r["estop_phase"] = phase_hint(a.get("stop") or "")
        r["session_continued"] = "y" if a.get("recovered") else "n"
    px = None
    if cur and (J.get("target_scan_blob") or {}).get("centre"):
        px = J["target_scan_blob"]["centre"]
    elif a.get("log_px"):
        px = a["log_px"]
    if px:
        r["scan_px_x"], r["scan_px_y"] = rnd(px[0]), rnd(px[1])
        r["scan_dist_px"] = rnd(math.hypot(px[0] - geo["TS"][0], px[1] - geo["TS"][1]))
    est = None
    if cur and J.get("brick_xy_est"):
        est = J["brick_xy_est"]
    elif rc.get("brick_xy_est"):
        est = rc["brick_xy_est"]
    elif a.get("log_est"):
        est = a["log_est"]
    if est:
        r["est_x_mm"], r["est_y_mm"] = rnd(est[0]), rnd(est[1])
        r["scan_dist_mm"] = rnd(math.hypot(est[0] - geo["SCAN_XY"][0], est[1] - geo["SCAN_XY"][1]))
    if a.get("n_cand") is not None:
        r["n_candidates"] = a["n_cand"]
    fk_tip = None
    hq = J.get("hover_q") if cur else None
    if hq:
        p, _ = kin.fk(hq)
        hx, hy, hz = float(p[0]), float(p[1]), float(p[2])
        r["hover_x_mm"], r["hover_y_mm"] = rnd(hx), rnd(hy)
        fk_tip = hz - float(kin.MAT_Z)
        r["fk_tip_mm"] = rnd(fk_tip)
        if est:
            r["est_hover_err_mm"] = rnd(math.hypot(est[0] - hx, est[1] - hy))
    st = J.get("hover_tip_size_mm") if cur else None
    if st is not None:
        r["hover_tip_size_mm"] = rnd(st)
        if fk_tip is not None:
            r["tip_diff_mm"] = rnd(st - fk_tip)
    if a.get("line"):
        r.update(servo_stats(a))
        if a["servo"]:
            r["approach_to_first_servo_s"] = round(a["servo"][0]["t"] - a["t_log"], 1)
        if a["yaws"]:
            r["yaw_deg"], r["yaw_roll"] = a["yaws"][0]["deg"], a["yaws"][0]["roll"]
    gc = J.get("grip_closed") if cur and J.get("grip_closed") is not None else rc.get("grip_closed")
    if gc is not None:
        r["grip_closed"] = gc
        r["grip_class"] = "held" if float(gc) >= HELD_MIN else "empty"
    if cur and J.get("grip_lifted") is not None:
        r["grip_lifted"] = J["grip_lifted"]
    for key in ("held_size_ratio", "open_short", "shape_why"):
        if cur and J.get(key) is not None:
            r[key] = J[key]
    shape = J.get("shape") if cur and J.get("shape") is not None else rc.get("shape")
    if shape is not None:
        r["shape"] = shape
    pc = J.get("pre_close") if cur else None
    if isinstance(pc, dict):
        r["pre_close_verdict"] = pc.get("verdict", "")
        r["pre_close_reasons"] = ";".join(str(x) for x in pc.get("reasons", []))
    stn = (J.get("station") if cur else None) or a.get("station")
    r["station_used"] = "y" if stn else "n"
    if stn:
        r["station_verdict"] = stn.get("verdict", "")
    loop_label = rc.get("label") or (a["decision"] or {}).get("label") or (a["done"] or {}).get("label") or ""
    r["label_loop"] = loop_label
    r["label_det2"] = J.get("label_det2", "") if cur else ""
    if cur and J.get("label") is not None:
        r["label"] = J["label"]
        r["label_source"] = J.get("label_source", "")
    else:
        r["label"] = loop_label
        r["label_source"] = "loop (cycles.jsonl)" if loop_label else ""
    jl, rl = (J.get("label") if cur else None), rc.get("label")
    r["label_conflict"] = "y" if (jl is not None and rl is not None and jl != rl) else ""
    colour = (J.get("colour") if cur and J.get("colour") else None) or rc.get("colour") \
        or (a["decision"] or {}).get("colour") or ""
    r["colour"] = colour
    plate = (J.get("plate") if cur and J.get("plate") else None) or rc.get("plate") \
        or (a["done"] or {}).get("plate") or ""
    r["plate"] = plate
    fin = ""
    pm = PLACED_RE.search(str(rc.get("failure") or ""))
    if pm:
        fin = pm.group(1)
    elif rc.get("success") is True:
        fin = plate
    r["final_plate"] = fin
    dc = J.get("decision_conf") if cur and J.get("decision_conf") is not None else rc.get("decision_conf")
    dr = J.get("decision_reasons") if cur and J.get("decision_reasons") is not None else rc.get("decision_reasons")
    r["decision_conf"] = dc or ""
    if isinstance(dr, list):
        r["decision_reasons"] = ";".join(str(x) for x in dr)
    elif dr:
        r["decision_reasons"] = str(dr)
    if cur and isinstance(J.get("views"), list):
        r["views_used"] = len(J["views"])
    T = rc.get("timings") or (a["done"] or {}).get("timings") or (J.get("timings") if cur else None) or {}
    for key in ("wait_s", "pick_s", "inspect_s", "place_s", "return_s", "cycle_s"):
        if T.get(key) is not None:
            r["t_" + key] = T[key]
    if d and d.get("joints") and a["kind"] == "pick" and t_start is not None:
        jm = joint_metrics(d["joints"], t_start)
        for key in JOINT_KEYS:
            if jm.get(key) is not None:
                r[key] = jm[key]
        r["joints_window"] = jm.get("joints_window", "")
        r["joints_rows"] = jm.get("joints_rows", "")
    elif d and d.get("joints"):
        r["joints_window"] = "not used (" + a["kind"] + ")"
    r["cams_top"] = (J.get("cams") or {}).get("top", "") if cur else ""
    r["dir_files"] = d["n_files"] if d else ""
    return r


def process_run(run, geo):
    atts, estops, sessions, notes = parse_run(run)
    dirs = load_dirs(run)
    recs = load_cycles(run)
    allatts = match_all(run, atts, recs, dirs, sessions)
    rows = [build_row(run, a, geo) for a in allatts]
    intervals = []
    for a in atts:
        ts = [s["t"] for s in a["servo"]]
        intervals += [b - x for x, b in zip(ts, ts[1:]) if 0 < b - x <= 10]
    return {"run": run, "atts": allatts, "estops": estops, "sessions": sessions, "notes": notes, "rows": rows,
            "recs": recs, "dirs": dirs, "intervals": intervals, "n_loop_attempts": len(atts),
            "n_recs": len(recs), "n_cycles_only": sum(1 for a in allatts if a["source"].startswith("cycles"))}


# ----------------------------------------------------------------- estop lines, images and triggers
def image_class(im):
    if im is None:
        return "UNKNOWN", "no image saved"
    lab = MANUAL_LABELS.get(im["file"])
    if lab is None:
        return "UNREVIEWED", ""
    return lab[0], lab[1]


def trigger_events(triggers):
    # hand_watch rows that met the PERSIST rule, one event per view and per second (a second can hold several rows)
    groups = collections.OrderedDict()
    for h in triggers:
        groups.setdefault((h["view"], int(h["t"])), []).append(h)
    out = []
    for (view, sec), hs in groups.items():
        out.append({"id": len(out), "view": view, "t": float(sec), "rows": [h["n"] for h in hs],
                    "blob": max(h["blob"] for h in hs), "active": any(h["active"] for h in hs),
                    "hits": max(h["hits"] for h in hs)})
    return out


def image_near(images, ref, window, view=None):
    # the estop image nearest in time to ref (|ref - t| <= window s); an image of the same view wins over a closer one
    best = None
    for im in images:
        dt = ref - im["t"]
        if abs(dt) > window:
            continue
        key = (0 if (view is None or im["view"] == view) else 1, abs(dt))
        if best is None or key < best[0]:
            best = (key, im, dt)
    return (best[1], best[2]) if best else (None, None)


def estop_rows(all_estops, hand, images, triggers):
    evs = sorted(all_estops, key=lambda e: e["t"])
    ev_no, last_t = 0, None
    for e in evs:
        if last_t is None or e["t"] - last_t > EVENT_GAP_S:
            ev_no += 1
        last_t = e["t"]
        e["event"] = "E%02d" % ev_no
    tev = trigger_events(triggers)
    out, linked_trig, linked_img = [], set(), set()
    base = {c: "" for c in ESTOP_COLS}
    for e in evs:
        nh = nearest_row(hand, e["t"], -10, 10)
        tr = nearest_row(tev, e["t"], -10, 10)
        im, im_dt = image_near(images, e["t"], 10, tr[0]["view"] if tr else None)
        if tr:
            linked_trig.add(tr[0]["id"])
        if im is not None:
            linked_img.add(im["file"])
            cls, note = image_class(im)
        elif "-> estop" not in e["text"]:
            cls, note = "NOT_ESTOP", "STOP line without estop text"
        else:
            cls, note = "UNKNOWN", "no estop image within +-10 s"
        row = dict(base)
        row.update({"run": e["run"], "event_id": e["event"], "row_source": "loop estop line",
                    "session": e["session"], "loop_line": e["line"], "t_utc": iso(e["t"]), "kind": e["kind"],
                    "ep": "" if e["ep"] is None else e["ep"], "phase": e["phase"], "text": e["text"],
                    "hand_view": nh[0]["view"] if nh else "", "hand_blob": nh[0]["blob"] if nh else "",
                    "hand_active": nh[0]["active"] if nh else "", "hand_dt_s": rnd(nh[1]) if nh else "",
                    "trigger_view": tr[0]["view"] if tr else "", "trigger_hits": tr[0]["hits"] if tr else "",
                    "trigger_dt_s": rnd(tr[1]) if tr else "", "image_file": im["file"] if im else "",
                    "image_dt_s": rnd(im_dt) if im else "", "image_class": cls, "image_note": note})
        out.append(row)
    for ev in tev:
        if ev["id"] in linked_trig:
            continue
        im, im_dt = image_near(images, ev["t"], 2, ev["view"])
        if im is not None:
            linked_img.add(im["file"])
            cls, note = image_class(im)
        else:
            cls, note = "UNKNOWN", "trigger second with no image within +-2 s"
        row = dict(base)
        row.update({"row_source": "hand_watch trigger (no loop line)", "t_utc": iso(ev["t"]), "kind": "hand trigger",
                    "hand_view": ev["view"], "hand_blob": ev["blob"], "hand_active": ev["active"],
                    "trigger_view": ev["view"], "trigger_hits": ev["hits"], "image_file": im["file"] if im else "",
                    "image_dt_s": rnd(im_dt) if im else "", "image_class": cls,
                    "image_note": note + "; %d hand_watch trigger rows in this second" % len(ev["rows"])})
        out.append(row)
    for im in images:
        if im["file"] in linked_img:
            continue
        cls, note = image_class(im)
        near = sorted(set(e["event"] for e in evs if abs(e["t"] - im["t"]) <= 10))
        if near:
            note = note + "; same time as loop estop event " + ", ".join(near)
        else:
            note = note + "; no loop estop line within +-10 s"
        if im["hand"] is None:
            note = (note + "; no hand_watch row within -1..+2 s").lstrip("; ")
        row = dict(base)
        row.update({"row_source": "estop image (no linked loop line)", "t_utc": iso(im["t"]), "kind": "image",
                    "hand_view": im["view"], "trigger_view": im["view"], "image_file": im["file"],
                    "image_class": cls, "image_note": note})
        out.append(row)
    return out


def write_csv(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="raise", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def pct(a, b):
    return "" if not b else "%.1f%%" % (100.0 * a / b)


def epoch_utc(s):
    # ISO UTC text as written by iso() -> epoch seconds
    return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).timestamp()


def ref_of(r):
    # row reference used in the report: run number and attempt index (unique), with the episode number in brackets;
    # episode numbers repeat across attempts (report section 7, rec 3)
    if r["ep"] != "":
        return "%s a%s (ep %s)" % (r["run"][4:], r["attempt_idx"], r["ep"])
    return "%s a%s" % (r["run"][4:], r["attempt_idx"])


def attempt_at(rows, run, t):
    # the attempt row of run whose start is the latest one at or before epoch t (wait-stop rows have no start time)
    best = None
    for r in rows:
        if r["run"] == run and r["t_start_utc"] and epoch_utc(r["t_start_utc"]) <= t:
            if best is None or epoch_utc(r["t_start_utc"]) > epoch_utc(best["t_start_utc"]):
                best = r
    return best


def recover_cap():
    # the retry bound in sort_any.py: the first "for x in range(N)" after the def recover( line
    with open(R + "/app/tools/sort_any.py", encoding="utf-8", errors="replace") as f:
        src = f.read().splitlines()
    start = next((n for n, s in enumerate(src) if s.startswith("def recover(")), None)
    if start is None:
        return None, None
    for n in range(start, min(start + 40, len(src))):
        m = re.search(r"for \w+ in range\((\d+)\)", src[n])
        if m:
            return int(m.group(1)), n + 1
    return None, None


def line_of(path, needle):
    # first 1-based line number of path that contains needle, or None
    with open(path, encoding="utf-8", errors="replace") as f:
        for n, s in enumerate(f, 1):
            if needle in s:
                return n
    return None


def recs_and_checklist(results, rows, erows, images, hand, hand_unparsed, triggers):
    rec, chk = [], []
    loop_est = [e for e in erows if e["row_source"] == "loop estop line"]
    # 1. recovery retries: the loop event with the most "; retry N" lines
    rl = [e for e in loop_est if "; retry " in e["text"]]
    by_ev = collections.Counter((e["run"], e["event_id"]) for e in rl)
    cap, cap_line = recover_cap()
    if by_ev:
        (srun, sev), _ = max(by_ev.items(), key=lambda kv: kv[1])
        st = [e for e in rl if (e["run"], e["event_id"]) == (srun, sev)]
        t_st = [epoch_utc(e["t_utc"]) for e in st]
        last_n = max(int(re.search(r"; retry (\d+)", e["text"]).group(1)) for e in st)
        cap_note = "cap reached" if cap and last_n >= cap else "below cap"
        att = attempt_at(rows, srun, t_st[0])
        where = ref_of(att) if att else srun[4:]
        rec.append("1. Stop retrying sooner. %s logged %d retry lines in %d s (%s, last retry %d); recover() loops "
                   "range(%s) (sort_any.py line %s). Hand over to a person after a few retries."
                   % (where, len(st), int(max(t_st) - min(t_st)), st[0]["t_utc"][11:19] + " to " + st[-1]["t_utc"][11:19],
                      last_n, cap, cap_line))
    # 2. attempts with no save record
    attempts = [r for r in rows if r["kind"] in ("pick", "carry_over")]
    ns = [r for r in attempts if str(r["json_state"]).startswith("not saved")]
    rec.append("2. Save every attempt. %d of %d pick and carry-over attempts have no save record (rows: %s); their loop-log "
               "lines are the only record. Call save_ep() on every exit path."
               % (len(ns), len(attempts), ", ".join(ref_of(r) for r in ns) or "none"))
    # 3. episode numbers repeat, so a later save of the same number overwrites episode.json
    ow = [r for r in rows if r["json_state"] == "folder json = another save (overwritten)"]
    cnt = collections.Counter((r["run"], r["ep"]) for r in rows if r["ep"] != "")
    shared_txt = []
    for run in RUNS:
        n_ep = sum(1 for r in rows if r["run"] == run and r["ep"] != "")
        n_sh = sum(1 for r in rows if r["run"] == run and r["ep"] != "" and cnt[(run, r["ep"])] > 1)
        top = max(((k, v) for k, v in cnt.items() if k[0] == run), key=lambda kv: kv[1], default=(("", ""), 0))
        share = ("; most shared: ep %s (%d rows)" % (top[0][1], top[1])) if top[1] > 1 else ""
        shared_txt.append("%s: %d of %d rows share an episode number with another row%s" % (run[4:], n_sh, n_ep, share))
    rec.append("3. Key saves by attempt, not episode number. %s. %d rows (%s) read another save's episode.json, which "
               "save_ep() overwrites on every call."
               % ("; ".join(shared_txt), len(ow), ", ".join(ref_of(r) for r in ow) or "none"))
    # 4. carry-over crash and the gap to the next session
    for r in [r for r in rows if r["failure_category"] == "carry-over: crash"]:
        t_cr = epoch_utc(r["t_start_utc"])
        nxt = [s["t"] for res in results if res["run"] == r["run"] for s in res["sessions"] if s["t"] > t_cr]
        rec.append("4. Handle Stop on the carry-over start path. %s raised an uncaught grasp_auto.Stop from inspect_pose() at %s; "
                   "the process exited and the next session began %s s later at %s."
                   % (ref_of(r), r["t_start_utc"][11:19], int(min(nxt) - t_cr) if nxt else "n/a",
                      iso(min(nxt))[11:19] if nxt else "n/a"))
    # 5. label conflict between episode.json and cycles.jsonl
    lc = [r for r in rows if r["label_conflict"] == "y"]
    rec.append("5. One label source. %s: episode.json says %s, cycles.jsonl says %s; write one label field."
               % (", ".join(ref_of(r) for r in lc) or "none", ", ".join(str(r["label"]) for r in lc) or "-",
                  ", ".join(str(r["label_loop"]) for r in lc) or "-"))
    # 6. false positives per view (the unit is the saved image; UNKNOWN is left out of the rate)
    def fp_of(view):
        cl = collections.Counter(image_class(im)[0] for im in images if im["view"] == view)
        return cl.get("FALSE_POSITIVE", 0), cl.get("REAL_HAND", 0)
    (wf, wr), (cf, cr_), (tf, tr_) = fp_of("wrist"), fp_of("close"), fp_of("top")
    rec.append("6. Gate the wrist and close views. False positives: wrist %d of %d (%s), close %d of %d (%s), top %d of %d (%s). "
               "Require a top-view trigger or a second frame before a wrist or close trigger stops the loop."
               % (wf, wf + wr, pct(wf, wf + wr), cf, cf + cr_, pct(cf, cf + cr_), tf, tf + tr_, pct(tf, tf + tr_)))
    # 7. watchdog log silence against the loop's last stamped line
    last_hand = hand[-1]["t"]
    last_loop = max(max((e["t"] for e in read_lines(res["run"]) if e["stamped"]), default=0) for res in results)
    n_after = sum(1 for e in loop_est if epoch_utc(e["t_utc"]) > last_hand)
    hw = line_of(R + "/app/tools/hand_watch.py", "HAND? view=")
    rec.append("7. Add a watchdog heartbeat. hand_watch.py writes a HAND? line only when is_hand() is true (line %s), so silence "
               "after %s cannot show the watchdog was running; the loop logged until %s (%d min later, %d loop estop lines after it)."
               % (hw, iso(last_hand)[11:19], iso(last_loop)[11:19], (last_loop - last_hand) // 60, n_after))
    # 8. estimate-to-hover error against distance from the scan centre
    er = [r for r in rows if r["est_hover_err_mm"] != "" and r["scan_dist_mm"] != ""]
    near = summ([r["est_hover_err_mm"] for r in er if r["scan_dist_mm"] < 40.0])
    far = summ([r["est_hover_err_mm"] for r in er if r["scan_dist_mm"] >= 60.0])
    rec.append("8. Limit picks to the near scan zone or recalibrate the scan map. Estimate-to-hover error: median %s mm within "
               "40 mm of the scan centre (n=%d), %s mm at 60 mm or more (n=%d); Pearson r %s (n=%d)."
               % (near["med"], near["n"], far["med"], far["n"],
                  pearson(vals(er, "scan_dist_mm"), vals(er, "est_hover_err_mm")), len(er)))
    # 9. IK failures among approach failures
    ap = [r for r in rows if r["outcome"] == "approach_fail"]
    ik = [r for r in ap if r["failure_category"] == "approach: IK fail"]
    ex = ik[0]["failure_text"] if ik else ""
    rec.append("9. Reject unreachable bricks before the approach. %d of %d approach failures are IK failures (%s); "
               "one logged error reads '%s'." % (len(ik), len(ap), ", ".join(ref_of(r) for r in ik) or "none", ex[:80]))
    # 10. approach time (pick attempts) and pick time of successful cycles
    succ = [r for r in rows if r["outcome"] == "success"]
    mv = summ(vals(rows, "moving_pre_close_s"))
    fs = summ(vals(rows, "approach_to_first_servo_s"))
    rec.append("10. Shorten the approach. Pick attempts spent a median %s s moving before close (n=%d) and a median %s s from the "
               "attempt line to the first servo line (max %s s, n=%d); successful picks took a median %s s."
               % (mv["med"], mv["n"], fs["med"], fs["max"], fs["n"], summ(vals(succ, "t_pick_s"))["med"]))
    for res in results:
        chk.append("- [x] logs/sort_any_%s.out: %d lines; %d unparsed (section 8)"
                   % (res["run"], len(read_lines(res["run"])), len(res["notes"]["unparsed"])))
        chk.append("- [x] logs/sort/%s/cycles.jsonl: %d records; each matched to a loop line or listed as its own row"
                   % (res["run"], res["n_recs"]))
        chk.append("- [x] logs/sort/%s/ep_*: %d folders; episode.json and joints.csv read where present (json_state column)"
                   % (res["run"], len(res["dirs"])))
    chk.append("- [x] logs/monitor/hand_watch.log: %d lines; %d unparsed (section 6)" % (len(hand), len(hand_unparsed)))
    chk.append("- [x] logs/monitor/estop_*.jpg: %d files, each viewed and classified; all listed in estops.csv" % len(images))
    sa = R + "/app/tools/sort_any.py"
    with open(sa, encoding="utf-8", errors="replace") as f:
        sa_lines = sum(1 for _ in f)
    chk.append("- [x] app/tools/sort_any.py: cycle(), save_ep() and main() read for the fields used; recover() located by grep "
               "(file last modified %s, %d lines)" % (iso(os.stat(sa).st_mtime), sa_lines))
    chk.append("- [x] app/tools/kin.py: lines 8-71 read; fk() and MAT_Z used here; the import reads files only (URDF, project_robots.json)")
    chk.append("- [x] app/pick_v2/kinematics.py: import chain checked; hardware-term grep run (see laptop-log.md)")
    chk.append("- [x] app/tools/scan_calib.json: q_scan through kin.fk() gives SCAN_XY; target_scan gives TS (main())")
    doc = R + "/docs/PICK-ANYWHERE.md"
    with open(doc, encoding="utf-8", errors="replace") as f:
        doc_hit = next((n for n, s in enumerate(f, 1) if "gripper reading" in s and "1.1" in s), None)
    chk.append(("- [x] docs/PICK-ANYWHERE.md line %s states the held threshold (gripper reading >= 1.1); HELD_MIN = %s here"
                % (doc_hit, HELD_MIN)) if doc_hit else "- [ ] docs/PICK-ANYWHERE.md: held threshold line not found")
    chk.append("- [ ] app/tools/grasp_auto.py, app/tools/pick_any.py, app/tools/hand_watch.py: not re-read in full in this pass")
    return rec, chk


def build_report(results, rows, erows, images, hand, hand_unparsed, triggers):
    L = []
    add = L.append
    add("# P2-B-ANALYZE report (analyst E1): pick-anywhere runs any_1731 and any_1740")
    add("")
    add("Times are UTC (epoch base 2026-11-14T00:00:00Z). Row ids are `run aN (ep K)`: aN is the attempt index, unique within "
        "the run; K is the episode number, which repeats (section 7, rec 3). A row is one attempt: a loop-log line and/or a "
        "save record in logs/sort/<run>/cycles.jsonl.")
    add("")
    outcomes = ["success", "approach_fail", "grasp_fail", "estop", "stopped", "carry_over", "unknown"]
    add("## 1. Outcome counts (one row per attempt)")
    add("| run | rows | " + " | ".join(outcomes) + " |")
    add("|" + "---|" * (len(outcomes) + 2))
    for run in RUNS + ["all"]:
        rr = [r for r in rows if run == "all" or r["run"] == run]
        c = collections.Counter(r["outcome"] for r in rr)
        add("| %s | %d | %s |" % (run, len(rr), " | ".join(str(c.get(o, 0)) for o in outcomes)))
    pick_rows = [r for r in rows if r["kind"] == "pick"]
    add("")
    add("Pick attempts (approach started): %d, of which successes: %d. Carry-over rows: %d. Wait-stop rows: %d."
        % (len(pick_rows), sum(1 for r in pick_rows if r["outcome"] == "success"),
           sum(1 for r in rows if r["kind"] == "carry_over"), sum(1 for r in rows if r["kind"] == "wait_stop")))
    add("")
    add("## 2. Failure taxonomy (rows that are not success)")
    add("| category | n | example rows |")
    add("|---|---|---|")
    cats = collections.OrderedDict()
    for r in rows:
        if r["outcome"] == "success":
            continue
        cats.setdefault(r["failure_category"], []).append(ref_of(r))
    for cat, ids in sorted(cats.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        add("| %s | %d | %s |" % (cat, len(ids), ", ".join(ids[:4])))
    add("")
    add("## 3. Timing (seconds; median and max)")
    add("| measure | n | median | max |")
    add("|---|---|---|---|")
    succ = [r for r in rows if r["outcome"] == "success"]
    for key, name in (("t_wait_s", "wait, successful cycles"), ("t_pick_s", "pick, successful cycles"),
                      ("t_inspect_s", "inspect, successful cycles"), ("t_place_s", "place, successful cycles"),
                      ("t_return_s", "return, successful cycles"), ("t_cycle_s", "cycle, successful cycles")):
        s = summ(vals(succ, key))
        add("| %s | %d | %s | %s |" % (name, s["n"], s["med"], s["max"]))
    jr = [r for r in rows if str(r["joints_window"]).startswith("ok")]
    for key, name in (("moving_pre_close_s", "inside pick: approach moves before close (sum of bursts)"),
                      ("close_s", "inside pick: close (gripper from open to closed)"),
                      ("lift_start_after_close_s", "inside pick: lift start after close")):
        s = summ(vals(jr, key))
        add("| %s | %d | %s | %s |" % (name, s["n"], s["med"], s["max"]))
    lg = [r for r in rows if r["source"].startswith("loop")]
    s = summ(vals(lg, "approach_to_first_servo_s"))
    add("| inside pick: attempt line to first servo line (log) | %d | %s | %s |" % (s["n"], s["med"], s["max"]))
    s = summ([r["servo_pre_yaw_n"] + r["servo_post_yaw_n"] for r in lg if r["servo_pre_yaw_n"] != ""])
    add("| inside pick: servo iterations per attempt (before + after yaw) | %d | %s | %s |" % (s["n"], s["med"], s["max"]))
    n_st = sum(1 for r in succ if r["station_used"] == "y")
    add("")
    add("inspect_s measures only the station re-look (sort_any.py cycle()); %d of %d successful cycles had one. "
        "place_s covers place(plate)." % (n_st, len(succ)))
    add("")
    add("## 4. Grasp readings (grip_closed after the close command; held = %.1f or more)" % HELD_MIN)
    add("| class | n | min | median | max | grip_lifted median (n) |")
    add("|---|---|---|---|---|---|")
    for cls in ("empty", "held"):
        g = [r for r in rows if r["grip_class"] == cls]
        s = summ(vals(g, "grip_closed"))
        sl = summ(vals(g, "grip_lifted"))
        add("| %s | %d | %s | %s | %s | %s (%d) |" % (cls, s["n"], s["min"], s["med"], s["max"], sl["med"], sl["n"]))
    add("")
    add("## 5. Estimate-to-hover error vs distance from the scan centre")
    add("| scan distance (mm) | n | median error (mm) | max error (mm) |")
    add("|---|---|---|---|")
    er = [r for r in rows if r["est_hover_err_mm"] != "" and r["scan_dist_mm"] != ""]
    for lo, hi in ((0.0, 20.0), (20.0, 40.0), (40.0, 60.0), (60.0, 1e9)):
        sel = [r for r in er if lo <= float(r["scan_dist_mm"]) < hi]
        s = summ(vals(sel, "est_hover_err_mm"))
        label = ("%g to %g" % (lo, hi)) if hi < 1e8 else ("%g and more" % lo)
        add("| %s | %d | %s | %s |" % (label, s["n"], s["med"], s["max"]))
    add("")
    td = summ([float(r["tip_diff_mm"]) for r in rows if r["tip_diff_mm"] != ""])
    add("Pearson r of error against distance: %s (n=%d). Hover tip size minus FK tip height (mm): n=%d, median %s, min %s, max %s."
        % (pearson(vals(er, "scan_dist_mm"), vals(er, "est_hover_err_mm")), len(er), td["n"], td["med"], td["min"], td["max"]))
    add("")
    add("## 6. Hand watchdog: the 41 estop images, reviewed by eye, per view")
    add("| view | images | REAL_HAND | FALSE_POSITIVE | UNKNOWN | false-positive rate (FP / (FP+REAL)) |")
    add("|---|---|---|---|---|---|")
    for view in ("top", "close", "wrist", "all"):
        ims = [im for im in images if view == "all" or im["view"] == view]
        cl = collections.Counter(image_class(im)[0] for im in ims)
        fp, real = cl.get("FALSE_POSITIVE", 0), cl.get("REAL_HAND", 0)
        add("| %s | %d | %d | %d | %d | %s |" % (view, len(ims), real, fp, cl.get("UNKNOWN", 0), pct(fp, fp + real)))
    hours = collections.Counter(iso(h["t"])[11:13] for h in hand)
    add("")
    add("hand_watch.log: %d HAND? lines from %s to %s (lines per UTC hour: %s), %d unparsed, %d trigger rows. "
        "Last estop image %s."
        % (len(hand), iso(hand[0]["t"])[11:19], iso(hand[-1]["t"])[11:19],
           ", ".join("%sh %d" % (h, n) for h, n in sorted(hours.items())), len(hand_unparsed), len(triggers),
           hhmm(max(im["t"] for im in images))))
    lines_by_run = collections.Counter(e["run"] for e in erows if e["row_source"] == "loop estop line")
    ev_by_run = {res["run"]: len(set(e["event_id"] for e in erows if e["run"] == res["run"] and e["event_id"]))
                 for res in results}
    img_loop = collections.Counter(e["image_class"] for e in erows if e["row_source"] == "loop estop line")
    add("Loop estop lines: %s. Estop events (lines within %g s grouped): %s. Image classes on loop lines: %s."
        % (dict(lines_by_run), EVENT_GAP_S, ev_by_run, dict(img_loop)))
    add("")
    recs, checks = recs_and_checklist(results, rows, erows, images, hand, hand_unparsed, triggers)
    add("## 7. Recommendations")
    for line in recs:
        add(line)
    add("")
    add("## 8. Data integrity")
    for res in results:
        rr = [r for r in rows if r["run"] == res["run"]]
        last_loop = [e for e in read_lines(res["run"]) if e["stamped"]][-1]
        add("- %s: %d loop attempt lines; %d save records (%d without a loop line); %d attempts with no save record; "
            "%d rows with another save's episode.json; %d label conflicts; %d traceback lines; %d unparsed lines; "
            "last stamped loop line %s."
            % (res["run"], res["n_loop_attempts"], res["n_recs"], res["n_cycles_only"],
               sum(1 for r in rr if str(r["json_state"]).startswith("not saved")),
               sum(1 for r in rr if r["json_state"] == "folder json = another save (overwritten)"),
               sum(1 for r in rr if r["label_conflict"] == "y"), res["notes"]["crash_lines"],
               len(res["notes"]["unparsed"]), hhmm(last_loop["t"])))
    add("")
    add("## 9. Coverage checklist")
    for line in checks:
        add(line)
    return "\n".join(L) + "\n"


def main():
    scan = json.load(open(SCAN_CALIB, encoding="utf-8"))
    p_scan, _ = kin.fk(scan["q_scan"])
    geo = {"TS": [float(v) for v in scan["target_scan"]], "SCAN_XY": (float(p_scan[0]), float(p_scan[1]))}
    hand, hand_unparsed = load_hand()
    triggers = [h for h in hand if h["trigger"]]
    images = load_images()
    link_images(images, hand)
    results = [process_run(run, geo) for run in RUNS]
    rows = [r for res in results for r in res["rows"]]
    estops = [e for res in results for e in res["estops"]]
    erows = estop_rows(estops, hand, images, triggers)
    write_csv(HERE + "/episodes.csv", COLS, rows)
    write_csv(HERE + "/estops.csv", ESTOP_COLS, erows)
    with open(HERE + "/report.md", "w", encoding="utf-8") as f:
        f.write(build_report(results, rows, erows, images, hand, hand_unparsed, triggers))
    summary = {"rows": len(rows), "estop_rows": len(erows), "images": len(images), "triggers": len(triggers),
               "hand_lines": len(hand), "hand_unparsed": len(hand_unparsed),
               "image_classes": dict(collections.Counter(image_class(im)[0] for im in images))}
    for res in results:
        summary[res["run"]] = {"loop_attempts": res["n_loop_attempts"], "save_records": res["n_recs"],
                               "cycles_only": res["n_cycles_only"], "dirs": len(res["dirs"]),
                               "outcomes": dict(collections.Counter(r["outcome"] for r in res["rows"])),
                               "notes": {k: v for k, v in res["notes"].items() if k != "unparsed"},
                               "unparsed_samples": res["notes"]["unparsed"][:6]}
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
