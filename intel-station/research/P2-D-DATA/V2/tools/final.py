import csv
import datetime
import json
import os

import numpy as np

R = "/home/intel/intel-junction-2026"
REP = R + "/data/studio-datasets-report/V2"
OUT = R + "/work/P2-D-DATA/V2/out"
os.makedirs(REP, exist_ok=True)
A = json.load(open(OUT + "/partA.json"))
C = json.load(open(OUT + "/partC.json"))
EP = {int(k): v for k, v in A["episodes"].items()}
NAMES = A["joint_names"]
DS = "6d6bdce3-b382-478d-9697-fb9e573b97af"
NOW = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
TASK = {0: "move the brick to the inspection spot", 1: "move the brick to the left plate"}
LAB = {
    0: ("S1-hold", "WARN", "grasp onset 233; closed to episode end (673); no put-down or return; OD-9 S1 not complete"),
    1: ("S1-hold", "WARN", "starts moving at frame 0; grasp onset 68; closed 85-580 to episode end; no put-down or return"),
    2: ("S1-hold", "WARN", "grasp onset 182; closed 191-544 to episode end; no put-down or return"),
    3: ("S1-place", "OK", "grasp onset 240; put-down 607 at spot A (lift -59.35); back home 649; end pose 2.63 from home"),
    4: ("S1-place", "WARN", "grasp onset 268; put-down 632 at spot A (lift -60.58); never returns home (end pose 37.48 from home)"),
    5: ("S1-place", "OK", "grasp onset 224; put-down 546 at spot B (lift 20.99); back home 619; end pose 2.03 from home"),
    6: ("S1-place", "OK", "grasp onset 164; put-down 423 at spot B (lift 12.61); back home 480; end pose 1.95 from home"),
    7: ("full-pick-place-BLUE", "WARN", "task 0 text but released on BLUE (image f1209/f1230); grasp onset 345; no return home (end pose 89.24 from home)"),
    8: ("place-only-GREEN", "WARN", "starts holding (closed 2.155 = brick band, not image-confirmed); release 148 on GREEN (image f148/f165); no return within 8 units (end pose 8.2 from home)"),
    9: ("place-only-GREEN", "WARN", "starts closed at 1.96 (between empty 0.98 and brick band; brick unconfirmed); release 187 over GREEN (moderate); back home 311; empty closure 213-340 excluded"),
    10: ("place-only-GREEN", "OK", "starts holding (closed 2.155 = brick band); release 140 on GREEN (image f155); back home 304; end pose 5.43 from home"),
}
ONSET = {0: 233, 1: 68, 2: 182, 3: 240, 4: 268, 5: 224, 6: 164, 7: 345, 8: None, 9: None, 10: None}
REL = {0: None, 1: None, 2: None, 3: 607, 4: 632, 5: 546, 6: 423, 7: 1209, 8: 148, 9: 187, 10: 140}
RET = {0: None, 1: None, 2: None, 3: 649, 4: None, 5: 619, 6: 480, 7: None, 8: None, 9: 311, 10: 304}
SEG = {0: (233, None), 1: (68, None), 2: (182, None), 3: (0, 649), 4: (0, 632), 5: (0, 619), 6: (0, 480),
       7: (345, 1209), 8: (0, 148), 9: (0, 187), 10: (0, 140)}
SEGMEAN = {0: "to episode end, no release", 1: "to episode end, no release", 2: "to episode end, no release",
           3: "back home (OD-9 S1 complete)", 4: "put-down only, no return", 5: "back home (OD-9 S1 complete)",
           6: "back home (OD-9 S1 complete)", 7: "release on BLUE", 8: "release on GREEN",
           9: "release on GREEN (tail excluded)", 10: "release on GREEN"}


def r3(x):
    return [round(float(v), 2) for v in x]


def vec_stats(vs):
    a = np.array(vs, dtype=float)
    return {"n": len(vs), "median": r3(np.median(a, axis=0)), "p25": r3(np.percentile(a, 25, axis=0)),
            "p75": r3(np.percentile(a, 75, axis=0))}


def sc(vals):
    a = np.array(vals, dtype=float)
    return {"n": len(vals), "median": round(float(np.median(a)), 3), "p5": round(float(np.percentile(a, 5)), 3),
            "p95": round(float(np.percentile(a, 95)), 3), "min": round(float(a.min()), 3), "max": round(float(a.max()), 3)}


def dstats(vals, note):
    a = np.array(vals, dtype=float)
    p50 = float(np.percentile(a, 50))
    p95 = float(np.percentile(a, 95))
    return {"n": len(vals), "values_s": [round(float(v), 2) for v in vals], "p50_s": round(p50, 2),
            "p95_s": round(p95, 2), "max_s_1p5x_p95": round(1.5 * p95, 2), "note": note}


def rel_pose(e):
    if REL[e] is None:
        return None
    for g in EP[e]["closed_runs"]:
        if g.get("release") == REL[e]:
            return g["release_pose"]
    return None


def approach(e):
    for g in EP[e]["closed_runs"]:
        if g.get("release") == REL[e]:
            return g.get("approach_pose_0p5_1s")
    return None


def closed_level(e):
    return EP[e]["closed_runs"][0]["closed_med"]


# episodes.csv
hdr = ["episode", "task_index", "task_text", "frames", "dur_s", "type", "verdict", "grasp_onset_s", "release_s",
       "return_home_s", "seg_start_frame", "seg_end_frame", "seg_end_meaning", "closed_level", "start_home_dist",
       "end_home_dist", "release_pan", "release_lift", "release_elbow", "reasons"]
rows = []
for e in range(11):
    r = EP[e]
    lab, ver, why = LAB[e]
    s0, s1 = SEG[e]
    rp = rel_pose(e)
    rows.append([e, r["task_idx_data"][0], TASK[r["task_idx_data"][0]], r["length_data"], r["dur_s"], lab, ver,
                 round(ONSET[e] / 30.0, 2) if ONSET[e] is not None else "",
                 round(REL[e] / 30.0, 2) if REL[e] is not None else "",
                 round(RET[e] / 30.0, 2) if RET[e] is not None else "",
                 s0, s1 if s1 is not None else "", SEGMEAN[e], closed_level(e), r["start_home_dist"],
                 r["end_home_dist"], rp[0] if rp else "", rp[1] if rp else "", rp[2] if rp else "", why])
with open(REP + "/episodes.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(hdr)
    w.writerows(rows)

# calib.json
gcv = C["gripper_closure_validation"]
rv = C["release_validation"]
idle = C["idle_speed_per_episode"]
brick_eps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
rest_start = [e for e in range(11) if EP[e]["start_home_dist"] <= 8.0]
rest_end = [e for e in range(11) if EP[e]["end_home_dist"] <= 8.0]
spot_poses = {e: rel_pose(e) for e in [3, 4, 5, 6]}
green_eps = [8, 9, 10]
plates = C["plate_centroids_ref_top_640x480"]
calib = {
    "version": "V2 consolidated " + NOW,
    "dataset": DS,
    "episodes": 11,
    "frames": 6693,
    "fps": 30,
    "robot_type": "SO101_Follower",
    "joint_names": NAMES,
    "gripper": {
        "joint": "gripper.pos (index 5 of observation.state)",
        "closed_threshold": 4.0,
        "min_closed_frames": 10,
        "closing_direction": "value decreases (open about 16-36 to closed about 2.2)",
        "release_direction": "value increases through 4.0",
        "open_peak_2s_before_onset": sc([x["open_peak_2s"] for x in gcv]),
        "closed_on_brick_level": dict(sc([closed_level(e) for e in brick_eps]), episodes=brick_eps),
        "closed_empty_level": {"value": 0.98, "episode": 9, "span_frames": [213, 340],
                               "note": "fully closed, nothing in jaws"},
        "ambiguous_closed_level": {"value": 1.96, "episode": 9, "span_frames": [0, 187],
                                   "note": "between empty 0.98 and brick band 2.155-2.547; brick not confirmed"},
    },
    "closure_onset_rule": {
        "definition": "from the first frame of a closed run (gripper < 4.0 for >= 10 frames) walk back while the frame-to-frame gripper drop exceeds 0.3",
        "validated_on_episodes": [x["ep"] for x in gcv],
        "close_frames_onset_to_plateau": [x["close_frames"] for x in gcv],
        "drop_per_frame_first10": [x["drop_per_frame_10f"] for x in gcv],
        "pre_onset_max_step": [x["pre_onset_max_step"] for x in gcv],
    },
    "release_rule": {
        "definition": "first frame where the gripper is back at or above 4.0 after a closed run",
        "validated_on_episodes": [x["ep"] for x in rv],
        "rise_per_frame_next10": [x["rise_per_frame_10f"] for x in rv],
        "g_at_release": [x["g_at_release"] for x in rv],
        "g_peak_1s": [x["g_peak_1s"] for x in rv],
    },
    "stationary": {
        "max_abs_step_per_frame_arm_joints": 0.15,
        "dwell_frames": 15,
        "dwell_s": 0.5,
        "idle_first30_p95_per_episode": {str(e): idle[str(e)]["first30_p95"] for e in range(11)},
        "episodes_idle_at_start": [e for e in range(11) if idle[str(e)]["first30_p95"] < 0.15],
        "episodes_moving_at_start": [e for e in range(11) if idle[str(e)]["first30_p95"] >= 0.15],
        "basis": "per-frame max abs change of joints 0-4; idle starts have p95 <= 0.086",
    },
    "home": {
        "rule": "rest = first or last 15 frames within 8 units (lift, elbow, wrist) of the all-episode start median",
        "rest_start_episodes": rest_start,
        "start_pose": vec_stats([EP[e]["start_pose"] for e in rest_start]),
        "rest_end_episodes": rest_end,
        "end_pose": vec_stats([EP[e]["end_pose"] for e in rest_end]),
    },
    "hold_grasp_pose": {
        "episodes": [x["ep"] for x in gcv if x["ep"] != 9],
        "closure_plateau_pose": vec_stats([x["grasp_pose"] for x in gcv if x["ep"] != 9]),
        "note": "pose at the closure plateau; spread is large, so no single hold position",
    },
    "spot_put_down": {
        "episodes": {str(e): spot_poses[e] for e in [3, 4, 5, 6]},
        "all4": vec_stats([spot_poses[e] for e in [3, 4, 5, 6]]),
        "cluster_A_eps_3_4": vec_stats([spot_poses[3], spot_poses[4]]),
        "cluster_B_eps_5_6": vec_stats([spot_poses[5], spot_poses[6]]),
        "note": "two arm configurations at similar pan (lift about -60 vs about 13-21); the spot is not one pose",
    },
    "placed_brick_pixels_top_view": {
        "w3_points_640x480": [[348, 123], [340, 72], [391, 216]],
        "v2_status": "NOT ESTABLISHED: marker_check_w3_spot_points.png shows the W3 points on the robot gripper or arm at put-down (P2 on the arm in all three; P1 and P3 on the arm or next to it in eps 5 and 6); not used",
    },
    "plates_top_view_640x480": {
        "blue_xy_area": plates["blue_xy_area"],
        "green_xy_area": plates["green_xy_area"],
        "source": "HSV largest blob on ep0 f0 top view; both crosses land on the plate centres in marker_check_w3_spot_points.png",
    },
    "drops": {
        "blue": {"episodes": [7], "release_pose": rel_pose(7), "approach_0p5_1s": approach(7),
                 "basis": "image-verified: f1209 gripper over blue, f1230 brick on blue"},
        "green": {"episodes": green_eps, "release_pose": vec_stats([rel_pose(e) for e in green_eps]),
                  "approach_0p5_1s": vec_stats([approach(e) for e in green_eps]),
                  "basis": "ep8 image-verified (f148, f165); ep10 image-verified (f155); ep9 moderate (arm over green at f187)"},
    },
    "approach_poses_0p5_1s_before_release": [
        {"episode": e, "release_frame": REL[e], "pose": approach(e)} for e in [3, 4, 5, 6, 7, 8, 9, 10]],
    "durations": {
        "S1_start_to_return_home": dstats([RET[e] / 30.0 for e in [3, 5, 6]], "OD-9 S1 complete takes eps 3, 5, 6; n=3, preliminary"),
        "S1_pick_to_put_down": dstats([(REL[e] - ONSET[e]) / 30.0 for e in [3, 5, 6]], "onset to release; n=3"),
        "S2_onset_to_release": dstats([(REL[7] - ONSET[7]) / 30.0], "ep 7 only; n=1, not a basis for a limit"),
        "S3_start_to_release": dstats([REL[e] / 30.0 for e in [8, 9, 10]], "eps 8-10 start holding; n=3, preliminary"),
    },
    "joint_names_as_stored": NAMES,
}
with open(REP + "/calib.json", "w") as fh:
    json.dump(calib, fh, indent=1)

# gaps.json
gaps = {
    "version": "V2 consolidated " + NOW,
    "targets": {"S1": {"target": 40, "min": 30}, "S2": {"target": 30, "min": 25}, "S3": {"target": 30, "min": 25}},
    "usable_now": {"S1": {"OK": [3, 5, 6], "WARN": [4]}, "S2": {"OK": [], "WARN": [7]}, "S3": {"OK": [10], "WARN": [8, 9]}},
    "counts": {"S1": {"OK": 3, "WARN": 1}, "S2": {"OK": 0, "WARN": 1}, "S3": {"OK": 1, "WARN": 2}},
    "shortfall_ok_only": {"S1": {"to_target": 37, "to_min": 27}, "S2": {"to_target": 30, "to_min": 25},
                          "S3": {"to_target": 29, "to_min": 24}},
    "shortfall_incl_warn": {"S1": {"to_target": 36, "to_min": 26}, "S2": {"to_target": 29, "to_min": 24},
                            "S3": {"to_target": 27, "to_min": 22}},
    "not_counted": [
        {"episodes": [0, 1, 2], "why": "closed to episode end; no put-down, release or return (OD-9 S1 needs put-down and return)"},
        {"episodes": [3, 4, 5, 6], "why": "releases are on the mat at the spot, not on a plate; not S2/S3 data"},
        {"episodes": [9], "why": "empty closure 213-340 after release; start grip 1.96 not confirmed as a brick"},
    ],
    "datasets_reused": [{"id": DS, "studio_name": "Smoke test - inspection skill - 5 episodes (stale; 11 episodes on disk)",
                         "default_task": TASK[0], "tasks_on_disk": TASK, "episodes": list(range(11)),
                         "environment_id": "72206569-df16-4a8b-a98f-247bac4de63f"}],
    "recording_requests": [
        "S1: record +37 complete takes (min +27): pick a brick from the cluster, put it on ONE fixed inspection spot, return the arm home, open the gripper. Eps 3-4 and 5-6 used two different arm configurations; choose one spot and mark it on the table.",
        "S2 (BLUE, image-left plate in the top view): record +30 takes (min +25): brick at the spot or in hand -> BLUE plate -> release -> return home. Only ep 7 exists (no return home).",
        "S3 (GREEN, image-right plate): record +29 OK takes (min +24): brick confirmed in the jaws at start, release on GREEN, return home, no empty closures after release (ep 9 tail).",
        "Labels: task 1 'move the brick to the left plate' (eps 8-10) is used for GREEN; 'left' matches only in the robot's frame. Rename the stale Studio dataset name and re-label ep 7 (task 0 text, released on BLUE).",
        "Keep hands out of the top view at take start (visible in the ep 8 and ep 9 start frames).",
        "Vary brick start positions across the cluster; keep GOOD (clean 2x2) and DEFECTIVE (tape or not 2x2) takes in each skill.",
    ],
}
with open(REP + "/gaps.json", "w") as fh:
    json.dump(gaps, fh, indent=1)

# INVENTORY.md
L = []
L.append("# INVENTORY (V2 verifier) P2-D-DATA, " + NOW)
L.append("")
L.append("Source: laptop R/data/studio-datasets/" + DS + " (synced copy, read-only). Studio GET only: /api/dataset/{id}, /api/projects.")
L.append("")
L.append("## Dataset")
L.append("- Studio name 'Smoke test - inspection skill - 5 episodes' is stale (11 episodes on disk). default_task 'move the brick to the inspection spot'. Project 'Junction 2026' (ab767473-d6fd-47ff-812e-8239c455dabb). environment_id 72206569-df16-4a8b-a98f-247bac4de63f (name not checkable with the allowed GETs).")
L.append("- LeRobot v3.0, 30 fps, SO101_Follower, 11 episodes, 6693 frames. Tasks: 0 'move the brick to the inspection spot' (eps 0-7), 1 'move the brick to the left plate' (eps 8-10).")
L.append("- Camera keys: realsense 243622061421 (top view), realsense 243722064316 (close low view); AV1 yuv420p 640x480.")
L.append("- Checks: frame_index contiguous and equal to meta length in 11/11; timestamp = frame/30 (dev 0.0 s); container durations equal episode sums (file-000 41.77 s = eps 0+1; file-001 103.77 s = eps 2-6; file-002 77.57 s = eps 7-10); first and last frame decode in 22/22 streams.")
L.append("")
L.append("## Plates and releases (top view)")
L.append("- Plate centroids (ep0 f0, 640x480): blue " + str(plates["blue_xy_area"][:2]) + " on the image left, green " + str(plates["green_xy_area"][:2]) + " on the image right. Matches W2 within 2 px.")
L.append("- ep 7 release f1209 over BLUE (f1230 brick on blue). ep 8 release f148 over GREEN (f165 red brick on green). ep 9 release f187 over GREEN (moderate). ep 10 release f140 over GREEN (f155). Orange brick on blue in the ep 8-9 frames is ep 7's.")
L.append("- Frames: data/studio-datasets-report/V2/plate_check_top_view.png, start_and_hold_check_eps8_10.png.")
L.append("- Robot frame: task text 'left plate' matches GREEN only if 'left' is the robot's frame.")
L.append("")
L.append("## Put-down spot (eps 3-6)")
L.append("- Two arm configurations: A (eps 3, 4; lift -59.35, -60.58) and B (eps 5, 6; lift 20.99, 12.61). Pan -5.5 to -13.1 in all four. Frame: data/studio-datasets-report/V2/spot_check_put_down.png.")
L.append("- Placed-brick pixel centres NOT established (W3 points land on the arm; marker_check_w3_spot_points.png).")
L.append("")
L.append("## Episodes (episodes.csv has cut points and reasons)")
L.append("| ep | task | type | verdict | release s | return s |")
L.append("|---|---|---|---|---|---|")
for rr in rows:
    L.append("| %d | %d | %s | %s | %s | %s |" % (rr[0], rr[1], rr[5], rr[6], rr[8], rr[9]))
L.append("")
L.append("## Calibration (calib.json)")
L.append("- Gripper: open peak 2 s before closure median %.2f (n=9); closed on brick median %.3f (band %.3f-%.3f, n=10); empty closed 0.98 (ep 9); ambiguous 1.96 (ep 9 start). Closing = decrease, release = rise through 4.0." % (
    calib["gripper"]["open_peak_2s_before_onset"]["median"], calib["gripper"]["closed_on_brick_level"]["median"],
    calib["gripper"]["closed_on_brick_level"]["min"], calib["gripper"]["closed_on_brick_level"]["max"]))
L.append("- Closure onset rule validated on 9 closures (drop > 0.3 per frame; 12-20 frames to plateau, ep 7 45 frames). Release rule validated on 8 releases.")
L.append("- Stationary: 0.15 per frame over arm joints with 15-frame dwell; idle starts in 8 of 11 episodes (p95 <= 0.086).")
L.append("- Home (rest starts n=%d): median %s. Drops: blue ep 7 only; green eps 8-10, median release pose %s." % (
    len(rest_start), calib["home"]["start_pose"]["median"], calib["drops"]["green"]["release_pose"]["median"]))
L.append("- Durations (preliminary): S1 start-to-home p50 %.2f s, p95 %.2f s, max_s %.2f (n=3); S3 start-to-release p50 %.2f s, p95 %.2f s (n=3); S2 onset-to-release %.2f s (n=1)." % (
    calib["durations"]["S1_start_to_return_home"]["p50_s"], calib["durations"]["S1_start_to_return_home"]["p95_s"],
    calib["durations"]["S1_start_to_return_home"]["max_s_1p5x_p95"], calib["durations"]["S3_start_to_release"]["p50_s"],
    calib["durations"]["S3_start_to_release"]["p95_s"], calib["durations"]["S2_onset_to_release"]["p50_s"]))
L.append("")
L.append("## Gaps (gaps.json)")
L.append("- S1 OK 3 of 40 (min 30): shortfall 37 (27 to minimum); S2 OK 0 of 30 (min 25): shortfall 30 (25); S3 OK 1 of 30 (min 25): shortfall 29 (24).")
L.append("- Recording requests are in gaps.json.")
L.append("")
L.append("## Corrections to workers (findings.json has evidence)")
L.append("- W3 S1 cut at lift+1.0 s is wrong under OD-9 (cut must be the return-home frame; eps 3, 5, 6 return at 649, 619, 480).")
L.append("- W2 ep 8 'release on blue' is wrong: ep 8 released on GREEN.")
L.append("- W1 'plate clusters' are put-down spots, not plates.")
L.append("- W2/W3 closed median 2.547 is the plateau of eps 3-6; the brick-grasp median is %.3f." % calib["gripper"]["closed_on_brick_level"]["median"])
L.append("")
L.append("## Caveats")
L.append("- Snapshot of a growing dataset; counts change with each sync.")
L.append("- Start-of-episode brick presence for eps 8-10 is inferred from the closed level (image frames f5 show the gripper mostly out of view).")
L.append("- n=3 for S1 and S3 duration stats: preliminary. Watch mode NOT started by V2 (V1 only).")
L.append("- Pick map (W2, W3) not re-derived and not used.")
L.append("")
L.append("## Coverage checklist")
L.append("- [x] 11 episodes (per-episode row in episodes.csv)")
L.append("- [x] 22 camera streams, first and last frame decode")
L.append("- [x] 6 video files, container durations")
L.append("- [x] 3 data parquet files, 3 episode-meta files, tasks.parquet, info.json")
L.append("- [x] Studio GET /api/dataset/" + DS + " and /api/projects")
L.append("- [x] Worker reports read: W1 report and STATUS, W2 STATUS, W3 report and STATUS")
L.append("- [ ] W1 out/ and W2 results3/ and W3 out/ JSON and CSV files not re-read line by line (claims checked against the report text and against recomputed data)")
L.append("- [ ] meta/stats.json not used")
L.append("- [ ] Pick map not re-derived (excluded)")
L.append("- [ ] Close-view placed-brick pixel centres not measured")
L.append("- [x] Contact sheets: plate_check_top_view.png, spot_check_put_down.png, start_and_hold_check_eps8_10.png, marker_check_w3_spot_points.png")
with open(REP + "/INVENTORY.md", "w") as fh:
    fh.write("\n".join(L) + "\n")
print("WROTE", REP)
for f in ["INVENTORY.md", "gaps.json", "episodes.csv", "calib.json"]:
    print("FILE", f, os.path.getsize(REP + "/" + f))
print("GREEN_SUMMARY", calib["drops"]["green"]["release_pose"])
print("DURATIONS", json.dumps(calib["durations"]))
print("HOME_START", json.dumps(calib["home"]["start_pose"]))
print("HOME_END", json.dumps(calib["home"]["end_pose"]))
print("SPOT", json.dumps(calib["spot_put_down"]["all4"]), json.dumps(calib["spot_put_down"]["cluster_A_eps_3_4"]), json.dumps(calib["spot_put_down"]["cluster_B_eps_5_6"]))
print("HOLD", json.dumps(calib["hold_grasp_pose"]["closure_plateau_pose"]))
print("OPEN", json.dumps(calib["gripper"]["open_peak_2s_before_onset"]), "CLOSED", json.dumps(calib["gripper"]["closed_on_brick_level"]))
