#!/usr/bin/env python3
"""Build the P3-LABEL-STUDIO E1 deliverables from the census, the exact-action grouping, the curated-set check,
the V1 report (hint only) and the visual labels typed in below from the frames (zoom/*.jpg; working_notes.txt).

Writes into the E1 folder: datasets.json, labels_episodes.jsonl, mismatches.tsv, metrics.json.
Writes out/build_checks.json: frame-level comparison of every copy with its representative (same camera, aligned
frame index), grouping coverage, and the growth check of 7e99a9e4.
"""
import csv
import datetime
import glob
import json
import os
import subprocess

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

R = "/home/intel/intel-junction-2026"
WD = os.path.join(R, "work", "P3-LABEL-STUDIO", "E1")
OUTD = os.path.join(WD, "out")
FRD = os.path.join(WD, "frames")
SD = os.path.join(R, "data", "studio-datasets")
CURD = os.path.join(R, "data", "curated")
V1D = os.path.join(R, "data", "studio-datasets-report", "V1")
FPS = 30
SERIALS = {"top": "243622061421", "close": "243722064316"}
NOW = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
MAE_SAME = 4.0      # mean absolute pixel difference (0-255) below which two decoded frames count as the same view

# ----------------------------------------------------------------------------------------------------------------
# Visual labels, one record per physical episode (P#). Typed in from the frames; see working_notes.txt for evidence.
# decision: plate the brick ended on (BLUE | GREEN | SPOT = mat put-down | NONE = no placement).
# od10: correct plate by OD-10 (clean 2x2 -> BLUE; tape or not 2x2 -> GREEN; n/a = put-down or no decision).
# ----------------------------------------------------------------------------------------------------------------
PHYS = {
    "P1": dict(colour="orange", shape="2x2", tape="no", tape_loc="", orient="not recorded",
               bricks=["orange 2x2 clean (target)"], decision="BLUE", od10="BLUE", outcome="success",
               failure="none", ends="BLUE plate (top view left)", conf="high",
               evidence="Z1 right; census grip to_hi 40.4 s in U18",
               notes="Released into BLUE plate at 40.4 s. Arm not at rest after release (V1 ep 7: 89 units)."),
    "P2": dict(colour="red", shape="2x2", tape="no", tape_loc="", orient="not recorded",
               bricks=["red 2x2 marked '2' clean (target)"], decision="SPOT", od10="n/a", outcome="success",
               failure="none (two regrasps at 7.1 s and 9.0 s)", ends="mat, left-centre of table", conf="high",
               evidence="Z12 left; census to_hi 21.3 s in U15",
               notes="Put-down on mat. V1 ep 4 WARN: arm ends 37.5 units from home (no return to rest)."),
    "P3": dict(colour="red", shape="2x2", tape="no", tape_loc="", orient="not recorded",
               bricks=["red 2x2 marked '2' clean (target)"], decision="BLUE", od10="BLUE", outcome="success",
               failure="none", ends="BLUE plate", conf="high", evidence="Z12 right; census to_hi 24.1 s in U23",
               notes="U07 copies (18468a09:6, e2eae6c8:2) end at 23.7 s, before this release: truncated."),
    "P4": dict(colour="red", shape="2x2", tape="no", tape_loc="", orient="not recorded",
               bricks=["red 2x2 clean (target)"], decision="BLUE", od10="BLUE", outcome="success",
               failure="none", ends="BLUE plate", conf="high", evidence="Z13 left; census to_hi 20.0 s in U22",
               notes="U27 = U22[57:850]; U06 = U22[57:630] (both contain the release)."),
    "P5": dict(colour="red", shape="2x2", tape="no", tape_loc="", orient="not recorded",
               bricks=["red 2x2 clean (target)"], decision="SPOT", od10="n/a", outcome="success",
               failure="none", ends="mat, left-centre near close end of table", conf="high",
               evidence="Z1 left; census to_hi 14.2 s in U17",
               notes="Put-down on mat; V1 ep 6 end arm at rest (2.0 units)."),
    "P6": dict(colour="red", shape="2x2", tape="no", tape_loc="", orient="not recorded",
               bricks=["red 2x2 clean (target)"], decision="GREEN", od10="BLUE", outcome="success",
               failure="OD-10 wrong plate: clean 2x2 sent to GREEN", ends="GREEN plate", conf="high",
               evidence="Z7 left; Z39 U25 last frame (red on GREEN); census to_hi 12.1 s in U25",
               notes="Text 'to the green plate' matches the plate; OD-10 says a clean 2x2 goes to BLUE."),
    "P7": dict(colour="yellow", shape="2x2", tape="no", tape_loc="", orient="upright",
               bricks=["yellow 2x2 clean upright (target)", "red 2x2 on mat near cluster"], decision="SPOT",
               od10="n/a", outcome="success", failure="none", ends="mat, top-centre near robot base", conf="high",
               evidence="Z3 left; Z39 U14 last frame; census to_hi 20.4 s in U14",
               notes="Human hand enters the top view at the end of U14 (23.4 s); no stacking seen."),
    "P8": dict(colour="red", shape="2x2", tape="no", tape_loc="", orient="not recorded",
               bricks=["red 2x2 clean (target)"], decision="GREEN", od10="BLUE", outcome="success",
               failure="OD-10 wrong plate: clean 2x2 sent to GREEN", ends="GREEN plate, right edge", conf="high",
               evidence="Z7 right; Z6 left; Z39 U10 last frame (red on GREEN); census to_hi 16.4 s in U10",
               notes="Text 'to the green plate' matches the plate; OD-10 says a clean 2x2 goes to BLUE."),
    "P9": dict(colour="yellow", shape="2x2", tape="no", tape_loc="", orient="not recorded",
               bricks=["yellow/cream 2x2 clean (target)"], decision="SPOT", od10="n/a", outcome="success",
               failure="none", ends="mat, lower-right of cluster, alone (not stacked)", conf="high",
               evidence="Z14; Z39 U16 last frame; census to_hi 18.3 s in U16", notes="Put-down on mat."),
    "P10": dict(colour="blue", shape="2x2", tape="no", tape_loc="", orient="not recorded",
                bricks=["blue 2x2 clean (target, held)"], decision="NONE", od10="BLUE", outcome="stopped",
                failure="held_not_placed (grasp 7.9 s, no release, held to 22.4 s)",
                ends="still in gripper at end", conf="high", evidence="Z2 left; census grip events",
                notes="V1 ep 0: picked and held, never put down."),
    "P11": dict(colour="red", shape="2x2", tape="no", tape_loc="", orient="not recorded",
                bricks=["red 2x2 marked '2' clean (target, held)"], decision="NONE", od10="BLUE",
                outcome="stopped", failure="held_not_placed (grasp 2.5 s, held to 19.3 s)",
                ends="still in gripper at end", conf="high", evidence="Z11 left; census grip events",
                notes="V1 ep 1: picked and held, never put down."),
    "P12": dict(colour="red", shape="2x2", tape="no", tape_loc="", orient="not recorded",
                bricks=["red 2x2 marked '2' clean (target, held)"], decision="NONE", od10="BLUE",
                outcome="stopped", failure="held_not_placed (grasp 3.6 s, held to 18.1 s)",
                ends="still in gripper at end", conf="high", evidence="Z11 right; census grip events",
                notes="V1 ep 2: picked and held, never put down."),
    "P13": dict(colour="white", shape="2x2", tape="no", tape_loc="", orient="not recorded",
                bricks=["white 2x2 clean, held from frame 0 (target)", "red 2x2 pre-placed on GREEN",
                        "white 2x4 and blue 2x2 on mat"],
                decision="GREEN", od10="BLUE", outcome="success",
                failure="wrong plate vs text (left plate = BLUE) and OD-10 (clean 2x2 -> BLUE)",
                ends="GREEN plate beside the red 2x2", conf="high",
                evidence="Z19 (0-5.3 s); Z30; Z32 left; Z35; Z38 (5x, no tape)",
                notes="Gripper closed at frame 0 and opens once at 5.0 s. Brick is hidden between the jaws "
                      "at frame 0 (behind the wrist mount): held identity inferred from the single release."),
    "P14": dict(colour="white", shape="2x4", tape="no", tape_loc="", orient="not recorded",
                bricks=["white 2x4 held from frame 0 (target, no tape seen)",
                        "white 2x2 and red 2x2 pre-placed on GREEN", "blue 2x2 on mat"],
                decision="GREEN", od10="GREEN", outcome="success",
                failure="wrong plate vs text (left plate = BLUE)",
                ends="GREEN plate, right rim (studs over plate surface), beside red 2x2", conf="medium",
                evidence="Z20 and Z21 (top, close, 6.7-9.7 s); Z25; Z36; Z38 (5x, end)",
                notes="Opens at 6.27 s and re-closes at 7.3 s on nothing (V1 ep 9). 2x4 shape inferred from the "
                      "elongated outline at 3x; rim position is at the frame edge."),
    "P15": dict(colour="blue", shape="2x2", tape="yes", tape_loc="silver/grey band across top",
                orient="not recorded",
                bricks=["blue 2x2 WITH silver/grey tape, held from frame 0 (target)",
                        "white 2x2 and red 2x2 pre-placed on GREEN"],
                decision="GREEN", od10="GREEN", outcome="success",
                failure="wrong plate vs text (left plate = BLUE)",
                ends="GREEN plate beside the white 2x2", conf="high",
                evidence="Z23 (2.0 s brick between jaws); Z28; Z33 right (3x tape); Z32 right; Z29",
                notes="Opens once at 4.67 s. Brick is first visible between the jaws at 2.0 s; its pick is not "
                      "visible (behind the wrist mount at 0-1 s). Tape seen at 3x only."),
    "P16": dict(colour="red", shape="2x2", tape="no", tape_loc="", orient="not recorded",
                bricks=["red 2x2 clean (target)", "white 2x4 and orange 2x2 on mat"], decision="BLUE",
                od10="BLUE", outcome="success", failure="none", ends="BLUE plate (red inside plate 20.0-21.9 s)",
                conf="high", evidence="Z18; Z22 (20.0-21.7 s); Z27 (21.3-21.9 s); Z34 (5x crop, clean)",
                notes="Grasp 10.8 s, release over BLUE plate at 17.8 s. Corrects an earlier 'failed placement' "
                      "reading of the 21.9 s frame: the red 2x2 stays in the BLUE plate."),
    "P17": dict(colour="none", shape="none", tape="no", tape_loc="", orient="none",
                bricks=[], decision="NONE", od10="n/a", outcome="none (no pick)",
                failure="no_pick: raw batch, gripper constant at 16.73, 4 joints frozen",
                ends="static scene", conf="high",
                evidence="contact sheet U26; census action stats",
                notes="Raw batch of 8405 frames (280.2 s); placeholder text. Not usable."),
}

REP_ENTRY = {"P1": "6d6bdce3:7", "P2": "6d6bdce3:4", "P3": "6d6bdce3:12", "P4": "6d6bdce3:11",
             "P5": "6d6bdce3:6", "P6": "6d6bdce3:14", "P7": "6d6bdce3:3", "P8": "6d6bdce3:15",
             "P9": "6d6bdce3:5", "P10": "6d6bdce3:0", "P11": "6d6bdce3:1", "P12": "6d6bdce3:2",
             "P13": "6d6bdce3:8", "P14": "6d6bdce3:9", "P15": "6d6bdce3:10", "P16": "6d6bdce3:13",
             "P17": "7e99a9e4:0"}
U_TO_P = {"U01": "P7", "U02": "P2", "U03": "P9", "U04": "P5", "U05": "P1", "U06": "P4", "U07": "P3",
          "U08": "P6", "U09": "P8", "U10": "P8", "U11": "P10", "U12": "P11", "U13": "P12", "U14": "P7",
          "U15": "P2", "U16": "P9", "U17": "P5", "U18": "P1", "U19": "P13", "U20": "P14", "U21": "P15",
          "U22": "P4", "U23": "P3", "U24": "P16", "U25": "P6", "U26": "P17", "U27": "P4"}
# unit -> (representative unit, start, end): the unit is frames [start:end] of the representative unit
WINDOW_REL = {"U01": ("U14", 113, 639), "U02": ("U15", 21, 663), "U03": ("U16", 46, 578),
              "U04": ("U17", 0, 454), "U05": ("U18", 31, 1241), "U06": ("U22", 57, 630),
              "U07": ("U23", 0, 711), "U08": ("U25", 1, 387), "U09": ("U10", 0, 520),
              "U27": ("U22", 57, 850)}
TRUNCATED = {"18468a09:6", "e2eae6c8:2"}   # U07 copies: end before the release (no to_hi after the grasp)
# truncated copies end 0.4 s before the release of the full copy; brick still held over the BLUE plate at the cut
# (out/trunc_crop_top_f708.jpg, frames 708 of 18468a09:6 and 6d6bdce3:12)
# copies that are re-encoded (not byte-identical): same actions, same scene; pixel MAE 2.3-4.1 at offset 0
# (out/offset_check.json, 3 frames x 2 cams x offsets -3..3); scene checked visually (out/reenc_montage.jpg rows 1-4)
REENCODED_VISUAL = {"a4ba6ede:0", "a4ba6ede:1", "a4ba6ede:2", "a4ba6ede:3"}


def eid(ds, ep):
    return "%s:%d" % (ds[:8], ep)


def expected_plate(text):
    t = (text or "").lower()
    if "placeholder" in t or "raw batch" in t:
        return "PLACEHOLDER"
    if "green plate" in t:
        return "GREEN"
    if "blue plate" in t:
        return "BLUE"
    if "left plate" in t:       # spec: the BLUE plate is on the LEFT of the top view
        return "BLUE"
    if "spot" in t:
        return "SPOT"
    return "UNKNOWN"


def cam_key(keys, serial):
    for k in keys:
        if k.endswith(serial):
            return k
    return None


def frame_png(e, cam, idx, tag):
    keys = list(e["video_segments"].keys())
    k = cam_key(keys, SERIALS[cam])
    s = e["video_segments"][k]
    out = os.path.join(FRD, "cmp_%s_%s_f%04d.png" % (tag, cam, idx))
    if not os.path.exists(out):
        mp4 = os.path.join(SD, e["dataset"], s["mp4"])
        subprocess.run(["ffmpeg", "-v", "error", "-ss", "%.4f" % (s["from_ts"] + idx / FPS), "-i", mp4,
                        "-frames:v", "1", "-y", out], capture_output=True, text=True)
    return out if os.path.exists(out) else None


def mae_png(p, q):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
    b = np.asarray(Image.open(q).convert("RGB"), dtype=np.float32)
    if a.shape != b.shape:
        return None
    return float(np.abs(a - b).mean())


def main():
    os.makedirs(FRD, exist_ok=True)
    census = json.load(open(os.path.join(OUTD, "datasets_census.json")))
    eps = [json.loads(line) for line in open(os.path.join(OUTD, "episodes_census.jsonl"))]
    uniq = json.load(open(os.path.join(OUTD, "unique_episodes.json")))
    cur = json.load(open(os.path.join(OUTD, "curated_check.json")))["sets"]
    v1_labels = json.load(open(os.path.join(V1D, "labels.json")))
    v1_eps = list(csv.DictReader(open(os.path.join(V1D, "episodes.csv"))))
    cur_v1 = list(csv.DictReader(open(os.path.join(CURD, "episodes_v1.csv"))))
    uuid_of = {d["uuid"][:8]: d["uuid"] for d in census}
    dset = {d["uuid"]: d for d in census}
    # pixel alignment of each copy against its representative, from tools/offset_check.py
    OFFCHK = {}
    if os.path.exists(os.path.join(OUTD, "offset_check.json")):
        for ck, v in json.load(open(os.path.join(OUTD, "offset_check.json"))).items():
            means = v["mean_mae_by_offset"]
            combined = {dd: (means["top"][dd] + means["close"][dd]) / 2.0 for dd in means["top"]}
            best = min(combined, key=lambda dd: combined[dd])
            OFFCHK[ck] = {"best_offset_frames": int(best), "mae_offset0": {c: means[c]["0"] for c in means},
                          "mean_mae_by_offset": means}

    ENT = {}
    for e in eps:
        ENT[eid(e["dataset"], e["episode_index"])] = e
    GROUPS = {}
    for u in uniq:
        GROUPS[u["uid"]] = [eid(uuid_of[m["ds"]], m["ep"]) for m in u["members"]]
    members_all = [x for v in GROUPS.values() for x in v]
    checks = {"entries": len(ENT), "groups": len(GROUPS),
              "entries_in_groups": len(members_all), "each_entry_once": sorted(members_all) == sorted(ENT),
              "U_TO_P_covers_groups": set(U_TO_P) == set(GROUPS),
              "P_count": len(PHYS), "rep_entries_exist": all(v in ENT for v in REP_ENTRY.values())}
    assert checks["each_entry_once"] and checks["U_TO_P_covers_groups"] and checks["rep_entries_exist"], checks

    ENTRY_P = {}
    ENTRY_U = {}
    for u, members in GROUPS.items():
        for m in members:
            ENTRY_P[m] = U_TO_P[u]
            ENTRY_U[m] = u
    # release time of each physical episode = last opening after the grasp in its representative entry
    release_s = {}
    for p, rep in REP_ENTRY.items():
        ev = [x for x in ENT[rep]["grip_events_s"]]
        opens = [t for t, s in ev if s == "to_hi"]
        release_s[p] = opens[-1] if (opens and p not in ("P10", "P11", "P12", "P17")) else None

    # ---- per-entry records -----------------------------------------------------------------------------------
    recs = {}
    for key, e in ENT.items():
        p = ENTRY_P[key]
        u = ENTRY_U[key]
        ph = PHYS[p]
        text = (e.get("tasks") or [""])[0]
        rep = REP_ENTRY[p]
        if key == rep:
            relation, dup_of, offset = "representative (original Studio episode)", None, 0
        elif u in WINDOW_REL:
            ru, a, b = WINDOW_REL[u]
            offset = a
            relation = "window [%d:%d] of %s" % (a, b, rep)
            dup_of = rep
        else:
            offset = 0
            relation = "identical copy of %s (same frame count)" % rep
            dup_of = rep
        if key in TRUNCATED:
            relation += "; TRUNCATED before release"
        exp = expected_plate(text)
        if exp == "PLACEHOLDER":
            text_ok = "placeholder"
        elif ph["outcome"] in ("stopped",) or ph["decision"] == "NONE":
            text_ok = "incomplete (not placed)"
        elif ph["decision"] == exp:
            text_ok = "match"
        else:
            text_ok = "mismatch"
        if ph["od10"] in ("n/a",) or ph["decision"] in ("SPOT", "NONE"):
            dc = "n/a"
        else:
            dc = "yes" if ph["decision"] == ph["od10"] else "no"
        has_release = (release_s[p] is not None) and (key not in TRUNCATED)
        if p == "P17":
            trust = "nothing"
        elif (not has_release) or ph["decision"] == "NONE" or key in TRUNCATED:
            trust = "calibration"
        elif text_ok == "mismatch" or dc == "no":
            trust = "evaluation"
        else:
            trust = "training"
        ranges = e.get("action_range") or []
        names = e.get("action_first") and census[0]["joint_names"]
        sat = []
        for j, rng in enumerate(ranges):
            if rng[0] <= -99.99 or rng[1] >= 99.99:
                sat.append("%s[%s,%s]" % (names[j], rng[0], rng[1]))
        grip = e.get("grip_range")
        in_bounds = all(-100.0 <= r[0] and r[1] <= 100.0 for r in ranges[:5]) and (
            grip is not None and 0.0 <= grip[0] and grip[1] <= 100.0)
        recs[key] = {
            "entry": key, "dataset": e["dataset"], "dataset_prefix": e["dataset"][:8],
            "episode_index": e["episode_index"], "physical_episode": p, "unit_group": u,
            "relation": relation, "dup_of": dup_of, "representative_entry": rep,
            "fps": FPS, "n_frames": e["n_frames"], "dur_s": e["dur_s"], "t0_s": e.get("t0"),
            "cameras_top_serial": SERIALS["top"], "cameras_close_serial": SERIALS["close"],
            "task_text": text, "task_text_expected_plate": exp, "task_text_ok": text_ok,
            "bricks_involved": ph["bricks"],
            "target_brick": {"colour": ph["colour"], "hue_class": ph["colour"], "shape": ph["shape"],
                             "tape": ph["tape"], "tape_location": ph["tape_loc"],
                             "orientation": ph["orient"], "size_class": "DUPLO-2x2" if ph["shape"] == "2x2"
                             else ("DUPLO-2x4" if ph["shape"] == "2x4" else "n/a")},
            "outcome": ph["outcome"], "failure_category": ph["failure"],
            "where_brick_ends": ph["ends"], "release_s_in_rep": release_s[p],
            "decision_plate": ph["decision"], "od10_correct_plate": ph["od10"], "decision_correct": dc,
            "label_source": "visual", "confidence": ph["conf"], "evidence": ph["evidence"],
            "notes": ph["notes"],
            "joint_sanity": {"lag_best_frames": e.get("lag_best"), "mae_lag_best": e.get("mae_lag_best"),
                             "action_range_in_bounds": in_bounds, "gripper_range": grip,
                             "saturated_at_normalized_limit": sat, "census_flags": e.get("flags", [])},
            "timestamp_frame_alignment": {"frame_index_contiguous": "frame_index_not_contiguous" not in e.get("flags", []),
                                          "video_span_matches_rows": not any(f.startswith("video_span") for f in e.get("flags", [])),
                                          "meta_length_matches_rows": "meta_length_ne_rows" not in e.get("flags", [])},
            "trust": trust, "usable_for": [trust] if trust != "nothing" else [],
        }
    # entry-level fields of the truncated copies: the placement is not inside the entry (cut 0.4 s before release)
    for key in sorted(TRUNCATED):
        rr = recs[key]
        rr["outcome"] = "carry_over"
        rr["failure_category"] = ("cut before release: entry ends 0.4 s before the release in %s; brick still held "
                                  "over the BLUE plate at the cut (out/trunc_crop_top_f708.jpg)"
                                  % REP_ENTRY[rr["physical_episode"]])
        rr["where_brick_ends"] = "still in gripper above the BLUE plate at the cut (23.7 s); not placed inside the entry"
        rr["task_text_ok"] = "incomplete (cut before release)"
        rr["decision_plate"] = "NONE"
        rr["decision_correct"] = "unknown"
    # frame-level comparison of every non-representative entry with its representative (same camera)
    cmp_rows = []
    for key in sorted(recs):
        r = recs[key]
        if r["relation"].startswith("representative"):
            continue
        e = ENT[key]
        rep = ENT[REP_ENTRY[r["physical_episode"]]]
        offset = WINDOW_REL[ENTRY_U[key]][1] if ENTRY_U[key] in WINDOW_REL else 0
        maes = {}
        for cam in ("top", "close"):
            vals = []
            for idx in (60, e["n_frames"] // 2):
                if idx + offset >= rep["n_frames"]:
                    continue
                a = frame_png(e, cam, idx, key.replace(":", "_"))
                b = frame_png(rep, cam, idx + offset, REP_ENTRY[r["physical_episode"]].replace(":", "_"))
                if a and b:
                    vals.append(mae_png(a, b))
            maes[cam] = [round(v, 3) for v in vals if v is not None]
        flat = [v for c in maes.values() for v in c]
        same = bool(flat) and max(flat) < MAE_SAME
        if flat and max(flat) == 0.0:
            verdict = "identical_decode"
        elif key in REENCODED_VISUAL:
            verdict = "reencoded_same_scene"
        elif same:
            verdict = "pixel_match_under_threshold"
        else:
            verdict = "unverified"
        oc = OFFCHK.get(key)
        r["frame_check_vs_rep"] = {"frame_offset": offset, "mae_by_cam": maes,
                                   "same_view": same, "frame_verdict": verdict,
                                   "pixel_offset_check": oc}
        cmp_rows.append({"entry": key, "rep": REP_ENTRY[r["physical_episode"]], "offset": offset, "mae": maes,
                         "same_view": same, "verdict": verdict})
        if verdict == "unverified":
            r["notes"] += " Frame check vs representative did not match (MAE %s)." % maes
        elif verdict == "reencoded_same_scene":
            r["notes"] += (" Re-encoded copy, not byte-identical: pixel MAE at offset 0 %s (byte-identical copies give 0.0); "
                           "same scene checked visually (out/reenc_montage.jpg)." % oc["mae_offset0"])
    # growth check of 7e99a9e4 (census vs now)
    growth = {"census_run_utc": "2026-11-14T19:28Z", "now_utc": NOW, "files": []}
    for d in census:
        if d["uuid"].startswith("7e99a9e4"):
            base = os.path.join(SD, d["uuid"])
            for cam, vids in d["videos"].items():
                for v in vids:
                    p = os.path.join(base, v["file"])
                    growth["files"].append({"file": v["file"], "size_census": v["size"],
                                            "size_now": os.path.getsize(p), "same": v["size"] == os.path.getsize(p)})
            pf = glob.glob(os.path.join(base, "data", "chunk-*", "*.parquet"))
            growth["data_rows_census"] = d["data_rows"]
            growth["data_rows_now"] = sum(pq.read_metadata(f).num_rows for f in pf)
            growth["dir_mtime_note"] = "laptop listing shows 21:24 local (UTC+2) = 19:24 UTC for dataset dirs"
    growth["unchanged"] = all(f["same"] for f in growth["files"]) and growth["data_rows_now"] == growth["data_rows_census"]

    with open(os.path.join(OUTD, "build_checks.json"), "w") as fh:
        json.dump({"checks": checks, "copy_frame_checks": cmp_rows, "growth_7e99a9e4": growth}, fh, indent=1)

    # ---- labels_episodes.jsonl ----------------------------------------------------------------------------------
    keys_sorted = sorted(recs, key=lambda k: (recs[k]["dataset_prefix"], recs[k]["episode_index"]))
    with open(os.path.join(WD, "labels_episodes.jsonl"), "w") as fh:
        for k in keys_sorted:
            fh.write(json.dumps(recs[k]) + "\n")

    # ---- curated sets --------------------------------------------------------------------------------------------
    cur_out = []
    for s in cur:
        rows = []
        for e in s["episodes"]:
            m = e["match"] or {}
            ent = m.get("entry")
            r = recs.get(ent) if ent else None
            issue = []
            if r:
                if r["task_text_ok"] == "mismatch":
                    issue.append("task text mismatch")
                if r["decision_correct"] == "no":
                    issue.append("OD-10 decision wrong (clean 2x2 on GREEN)")
                if r["trust"] == "calibration":
                    issue.append("not a placement (%s)" % ("truncated" if ent in TRUNCATED else "held"))
            rows.append({"episode_index": e["episode_index"], "task": e["task"], "entry": ent,
                         "physical": ENTRY_P.get(ent) if ent else None, "matched_mae": m.get("mae"),
                         "covers_full_curated": m.get("covers_full_curated"), "tied": m.get("tied_entries"),
                         "issues": issue})
        cur_out.append({"set": s["set"], "fps": s["fps"], "info_total_episodes": s["total_episodes_info"],
                        "meta_rows": s["meta_episode_rows"], "data_episodes": s["data_episodes"],
                        "data_rows": s["data_rows"], "tasks": s["tasks"], "action_names": s["action_names"],
                        "video_keys": s["video_keys"], "matched": s["episodes_matched"],
                        "unmatched": s["episodes_unmatched"], "issues_info": s["issues"], "episodes": rows,
                        "physical_episodes": sorted({x["physical"] for x in rows if x["physical"]}),
                        "episodes_with_issues": sum(1 for x in rows if x["issues"])})

    # ---- datasets.json -----------------------------------------------------------------------------------------
    datasets = []
    for d in census:
        ent_keys = [k for k in keys_sorted if recs[k]["dataset"] == d["uuid"]]
        lens = [recs[k]["n_frames"] for k in ent_keys]
        cams = {}
        for vk, vids in d["videos"].items():
            v0 = vids[0] if vids else {}
            cams[vk] = {"serial": vk.split()[-1], "role": "top" if vk.endswith(SERIALS["top"]) else "close",
                        "codec": v0.get("codec"), "width": v0.get("w"), "height": v0.get("h"),
                        "fps_stream": v0.get("r_frame_rate"), "mp4_files": len(vids)}
        datasets.append({
            "group": "studio", "uuid": d["uuid"], "fps": d["fps"], "robot_type": d["robot_type"],
            "codebase_version": d["codebase_version"], "cameras": cams,
            "joint_names": d["joint_names"],
            "joint_units": "not stated in info.json (names end in .pos); observed action ranges shown below",
            "joint_minmax_action": d["joint_minmax_action"], "joint_minmax_state": d["joint_minmax_state"],
            "episodes_info_json": d["total_episodes_info"], "episodes_meta_rows": d["meta_episode_rows"],
            "episodes_with_data": d["data_episode_count"], "data_rows": d["data_rows"],
            "frames_info_json": d["total_frames_info"], "tasks": [t["task"] for t in d["tasks"]],
            "episode_indices": d["episode_indices"], "episode_frames": lens,
            "duration_s_total": round(sum(lens) / FPS, 2), "duration_s_min": round(min(lens) / FPS, 2) if lens else None,
            "duration_s_max": round(max(lens) / FPS, 2) if lens else None,
            "originals_in_dataset": [k for k in ent_keys if recs[k]["relation"].startswith("representative")],
            "entries": ent_keys, "trust_counts": {t: sum(1 for k in ent_keys if recs[k]["trust"] == t)
                                                  for t in ("training", "evaluation", "calibration", "nothing")},
            "census_issues": d["issues"],
            "note": {"6d6bdce3": "Studio name 'Smoke test - inspection skill - 5 episodes' (stale). 16 episodes now; "
                                 "V1 (13:35Z) saw 11.",
                     "7e99a9e4": "Raw batch: 1 entry, 280 s, placeholder text, 4 joints frozen. Growth check in "
                                 "out/build_checks.json.",
                     "18468a09": "Contains copies of episodes that also appear in 6d6bdce3 / e2eae6c8 / 800d7dc2 / "
                                 "d2fd0a80 / ba9eafba / 51100357 (see dup_of).",
                     }.get(d["uuid"][:8], "Only copies or windows of 6d6bdce3 episodes: adds no new physical "
                                           "episode (see dup_of and originals_in_dataset).")})
    for c in cur_out:
        datasets.append({"group": "curated", "name": c["set"], "path": "data/curated/" + c["set"], "fps": c["fps"],
                         "info_total_episodes": c["info_total_episodes"], "episodes_meta_rows": c["meta_rows"],
                         "episodes_with_data": c["data_episodes"], "data_rows": c["data_rows"],
                         "tasks": c["tasks"], "video_keys": c["video_keys"], "action_names": c["action_names"],
                         "matched_to_studio_entries": c["matched"], "unmatched": c["unmatched"],
                         "physical_episodes": c["physical_episodes"], "episodes_with_issues": c["episodes_with_issues"],
                         "episode_detail": c["episodes"], "issues_info": c["issues_info"]})
    datasets.append({"group": "v1_hint", "path": "data/studio-datasets-report/V1", "used_as": "hint only",
                     "stated_episodes_6d6bdce3": 11, "stated_generated_utc": "2026-11-14T13:35:40Z",
                     "release_labels_ep3_to_ep10": {k: v for k, v in v1_labels["release"].items()},
                     "agreement_with_visual_labels": "eps 3-10 agree on release plate (SPOT,SPOT,SPOT,SPOT,BLUE,GREEN,GREEN,GREEN)"})
    with open(os.path.join(WD, "datasets.json"), "w") as fh:
        json.dump({"stage": "studio-labels-v1", "utc": NOW, "datasets": datasets}, fh, indent=1)

    # ---- mismatches.tsv ----------------------------------------------------------------------------------------
    rows = []
    for k in keys_sorted:
        r = recs[k]
        if r["task_text_ok"] == "mismatch":
            rows.append(("task_text", k, r["physical_episode"], r["task_text"],
                         "%s on %s (%s)" % (r["physical_episode"], r["decision_plate"], r["where_brick_ends"]),
                         "text expects %s; placed on %s" % (r["task_text_expected_plate"], r["decision_plate"])))
        if r["decision_correct"] == "no":
            rows.append(("od10_decision", k, r["physical_episode"], "decision %s" % r["decision_plate"],
                         "OD-10 correct %s" % r["od10_correct_plate"],
                         "clean 2x2 %s" % r["target_brick"]["colour"]))
        if r["trust"] == "calibration" and k in TRUNCATED:
            rows.append(("truncated", k, r["physical_episode"], r["task_text"],
                         "no release inside the entry (cut at %.1f s)" % r["dur_s"],
                         "full copy %s releases at %s s" % (REP_ENTRY[r["physical_episode"]],
                                                            release_s[r["physical_episode"]])))
        elif r["trust"] == "calibration":
            rows.append(("not_placed", k, r["physical_episode"], r["task_text"], r["outcome"],
                         r["failure_category"]))
        if r["trust"] == "nothing":
            rows.append(("placeholder_raw", k, r["physical_episode"], r["task_text"], "no pick",
                         r["failure_category"]))
    for c in cur_out:
        for x in c["episodes"]:
            if x["issues"]:
                rows.append(("curated_label", "%s:%s" % (c["set"], x["episode_index"]), x["physical"] or "",
                             x["task"] or "", x["entry"] or "", "; ".join(x["issues"])))
    rows.append(("text_conflict", "6d6bdce3:7", "P1", "move the brick to the inspection spot",
                 "copies 18468a09:4 and e2eae6c8:0 say 'move the orange brick to the blue plate'; physical BLUE",
                 "same episode, two task texts in different datasets; the BLUE placement is verified"))
    for k in keys_sorted:
        fcv = recs[k].get("frame_check_vs_rep") or {}
        if fcv.get("frame_verdict") == "reencoded_same_scene":
            rows.append(("video_reencoded", k, recs[k]["physical_episode"],
                         "actions identical to %s (census MAE 0.0)" % recs[k]["representative_entry"],
                         "same scene (out/reenc_montage.jpg); pixel MAE at offset 0 %s" % fcv["pixel_offset_check"]["mae_offset0"],
                         "re-encoded copy: count the physical episode once (use the representative)"))
        oc = fcv.get("pixel_offset_check")
        if oc and oc["best_offset_frames"] != 0:
            m, b = oc["mean_mae_by_offset"], str(oc["best_offset_frames"])
            rows.append(("video_offset", k, recs[k]["physical_episode"], "rows vs video frames",
                         "best pixel offset %d: top %.2f close %.2f; offset 0: top %.2f close %.2f"
                         % (oc["best_offset_frames"], m["top"][b], m["close"][b], m["top"]["0"], m["close"]["0"]),
                         "copy video about %d frame off its rows; minor, flagged" % abs(oc["best_offset_frames"])))
    for x in cur_v1:
        idx = int(x["episode"])
        if idx == 13:
            rows.append(("curated_episodes_v1", "6d6bdce3:13", "P16", "DROP: " + x["reasons"],
                         "success (grasp 10.8 s, release on BLUE 17.8 s, red inside BLUE plate at end)",
                         "curated grip check missed the grasp"))
        elif idx in (7, 8, 9, 10, 14):
            rows.append(("curated_episodes_v1", "6d6bdce3:%d" % idx, ENTRY_P.get("6d6bdce3:%d" % idx, ""),
                         "%s: %s" % (x["decision"], x["reasons"][:90]),
                         recs["6d6bdce3:%d" % idx]["where_brick_ends"] + "; " + recs["6d6bdce3:%d" % idx]["outcome"],
                         "DROP reason is motion/rest based; physical outcome recorded separately"))
    rows.append(("v1_note", "6d6bdce3:8", "P13", "V1 ep 8 note: red brick on green plate",
                 "white 2x2 released on GREEN (red 2x2 pre-placed)", "V1 note misses the white release"))
    rows.append(("v1_note", "6d6bdce3:10", "P15", "V1 ep 10 release GREEN",
                 "agrees; blue brick carries silver/grey tape (DEFECT)", "V1 did not record the tape"))
    with open(os.path.join(WD, "mismatches.tsv"), "w") as fh:
        fh.write("kind\tentry\tphysical\trecorded\tverified\tevidence_or_reason\n")
        for row in rows:
            fh.write("\t".join(str(x).replace("\t", " ") for x in row) + "\n")

    # ---- metrics.json --------------------------------------------------------------------------------------------
    cnt = {t: sum(1 for k in recs if recs[k]["trust"] == t) for t in ("training", "evaluation", "calibration", "nothing")}
    phys_usable = sorted({recs[k]["physical_episode"] for k in recs if recs[k]["trust"] in ("training", "evaluation")})
    copies_same = sum(1 for c in cmp_rows if c["same_view"])
    verdicts = [c["verdict"] for c in cmp_rows]
    phys_trust = {}
    for p in sorted({recs[k]["physical_episode"] for k in recs}):
        ts = {recs[k]["trust"] for k in recs if recs[k]["physical_episode"] == p}
        phys_trust[p] = next(t for t in ("training", "evaluation", "calibration", "nothing") if t in ts)
    metrics = {
        "episodes_total": len(recs),
        "episodes_total_physical_unique": len(REP_ENTRY),
        "datasets_studio": len(census),
        "curated_sets": len(cur_out),
        "curated_episodes": sum(len(c["episodes"]) for c in cur_out),
        "curated_episodes_matched_exactly": sum(sum(1 for x in c["episodes"] if x["matched_mae"] == 0.0)
                                                for c in cur_out),
        "verified": len(REP_ENTRY) + verdicts.count("identical_decode") + verdicts.count("reencoded_same_scene"),
        "verified_by_own_frames": len(REP_ENTRY),
        "verified_copies_pixel_identical": verdicts.count("identical_decode"),
        "verified_copies_reencoded_visual_only": verdicts.count("reencoded_same_scene"),
        "copies_unverified": verdicts.count("unverified"),
        "copies_checked_by_frames": len(cmp_rows),
        "mismatched": sum(1 for k in recs if recs[k]["task_text_ok"] == "mismatch" or recs[k]["decision_correct"] == "no"),
        "task_text_mismatch": sum(1 for k in recs if recs[k]["task_text_ok"] == "mismatch"),
        "od10_decision_wrong": sum(1 for k in recs if recs[k]["decision_correct"] == "no"),
        "usable_training": cnt["training"],
        "usable_evaluation": cnt["evaluation"],
        "usable_calibration_only": cnt["calibration"],
        "nothing": cnt["nothing"],
        "usable_training_or_evaluation": cnt["training"] + cnt["evaluation"],
        "physical_usable_training_or_evaluation": len(phys_usable),
        "physical_usable_list": phys_usable,
        "incomplete_or_cut": sum(1 for k in recs if recs[k]["outcome"] in ("stopped", "carry_over")
                                 or recs[k]["trust"] == "nothing"),
        "text_plate_conflicts_physical": sum(1 for p in phys_trust if len({recs[k]["task_text_expected_plate"]
                                             for k in recs if recs[k]["physical_episode"] == p}) > 1),
        "physical_trust_training": sum(1 for t in phys_trust.values() if t == "training"),
        "physical_trust_evaluation_only": sum(1 for t in phys_trust.values() if t == "evaluation"),
        "physical_trust_calibration_only": sum(1 for t in phys_trust.values() if t == "calibration"),
        "physical_trust_nothing": sum(1 for t in phys_trust.values() if t == "nothing"),
        "reencoded_copies": verdicts.count("reencoded_same_scene"),
        "video_offset_entries": sum(1 for k in recs
                                    if ((recs[k].get("frame_check_vs_rep") or {}).get("pixel_offset_check") or {})
                                    .get("best_offset_frames", 0) != 0),
        "duplicate_entries": sum(1 for k in recs if recs[k]["dup_of"]),
    }
    prev_path = os.path.join(OUTD, "metrics_run1_2002Z.json")
    prev = json.load(open(prev_path))["metrics"] if os.path.exists(prev_path) else None
    delta = None
    if prev is not None:
        delta = {k: metrics[k] - prev[k] for k in metrics
                 if isinstance(metrics[k], int) and isinstance(prev.get(k), int)
                 and not isinstance(metrics[k], bool) and not isinstance(prev.get(k), bool)}
    with open(os.path.join(WD, "metrics.json"), "w") as fh:
        json.dump({"stage": "studio-labels-v1", "utc": NOW, "metrics": metrics, "previous": prev, "delta": delta},
                  fh, indent=1)
    print(json.dumps(metrics, indent=1))
    print("checks", json.dumps(checks))
    print("growth unchanged", growth["unchanged"], "copies same view", copies_same, "of", len(cmp_rows))
    print("mismatch rows", len(rows))


if __name__ == "__main__":
    main()
