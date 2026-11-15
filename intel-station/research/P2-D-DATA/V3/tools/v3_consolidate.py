#!/usr/bin/env python3
# V3 consolidation: applies the V3 image-review decisions to the automated pass and writes the final set.
# usage: python3 -I v3_consolidate.py AUTO_DIR REPORT_DIR
import sys
import json
from pathlib import Path

import numpy as np

AUTO = Path(sys.argv[1])
REP = Path(sys.argv[2])
REP.mkdir(parents=True, exist_ok=True)
FPS = 30.0
recs = json.loads((AUTO / "episodes_v3.json").read_text())
calib = json.loads((AUTO / "calib.json").read_text())
DSID = recs[0]["dataset_id"]

NOTE_NUM = "numbers only: grasp and lift, arm still holding in the air at episode end"
DEC = {
    "0": dict(basis=NOTE_NUM),
    "1": dict(basis=NOTE_NUM),
    "2": dict(basis=NOTE_NUM),
    "3": dict(basis="V3 frames: put-down on the mat between the plates (39.0 s); not a plate"),
    "4": dict(basis="V3 frames: put-down on the mat (63.3 s); no return home"),
    "5": dict(basis="V3 frames: put-down on the mat (84.0 s); not a plate"),
    "6": dict(basis="V3 frames: put-down on the mat (101.4 s); not a plate"),
    "7": dict(basis="V3 frames: released over the blue plate (40.2 s); brick on the blue plate at 43.4 s; pan +32.1"),
    "8": dict(basis="V3 frames: released over the green plate (48.3 s); white 2x2 on green by 49.1 s; orange brick already on blue at 43.6 s (ep07 brick); pan -54.7"),
    "9": dict(verdict="WARN",
              basis="V3 frames: green plate gains a white brick at its right edge by 66.1 s (moderate); pan -57.2",
              add="V3: start-held inferred from gripper plateau 1.96 (empty closure 0.98) and a white part under the jaws at 55.1 s (moderate); end 6.2 units from home"),
    "10": dict(basis="V3 frames: released over the green plate (71.5 s); blue 2x2 on the green right edge by 77.2 s; pan -51.6"),
}

for r in recs:
    d = DEC.get(str(r["episode_index"]), {})
    r["plate_basis"] = d.get("basis", "automated pan threshold only")
    r["verdict_auto"] = r["verdict"]
    if "verdict" in d:
        r["verdict"] = d["verdict"]
    if "add" in d:
        r["reasons"].append(d["add"])


def who(types, verdicts):
    return [r["episode_index"] for r in recs if r["type"] in types and r["verdict"] in verdicts]


BLUE_T = ["full-pick-place-BLUE", "place-only-BLUE"]
GREEN_T = ["full-pick-place-GREEN", "place-only-GREEN"]
needs = {
    "S1_OD9_pick_put_spot_return": dict(target=40, minimum=30, ok=who(["S1-place"], ["OK"]),
                                        warn=who(["S1-place-noreturn"], ["WARN"]),
                                        other_usable_for_pick_lift=who(["S1-hold"], ["WARN"])),
    "S2_carry_BLUE_release": dict(target=30, minimum=25, ok=who(BLUE_T, ["OK"]), warn=who(BLUE_T, ["WARN"])),
    "S3_carry_GREEN_release": dict(target=30, minimum=25, ok=who(GREEN_T, ["OK"]), warn=who(GREEN_T, ["WARN"])),
}
for d in needs.values():
    d["ok_n"] = len(d["ok"])
    d["warn_n"] = len(d["warn"])
    d["shortfall_target_ok_only"] = max(0, d["target"] - d["ok_n"])
    d["shortfall_min_ok_only"] = max(0, d["minimum"] - d["ok_n"])
    d["shortfall_target_ok_plus_warn"] = max(0, d["target"] - d["ok_n"] - d["warn_n"])
    d["shortfall_min_ok_plus_warn"] = max(0, d["minimum"] - d["ok_n"] - d["warn_n"])

types = {}
for r in recs:
    types.setdefault(r["type"], []).append(r["episode_index"])
gaps = dict(
    dataset_id=DSID,
    snapshot="sync_datasets.log last entry 2026-11-14T13:23:07Z; info.json total_episodes 11, total_frames 6693",
    studio_name_stale="Smoke test - inspection skill - 5 episodes",
    episodes_on_disk=len(recs), frames_on_disk=int(sum(r["n_frames"] for r in recs)),
    type_counts={k: len(v) for k, v in sorted(types.items())}, episodes_by_type=types,
    needs=needs,
    recording_requests=[
        "S1 (OD-9): pick a brick from the cluster, put it on the marked inspection spot, return the arm home, keep still 1 s, then end the episode. Use the team text 'move the brick to the inspection spot'.",
        "Fix the spot: mark it with tape. Put-down poses over eps 3-6 differ by 81 units in shoulder_lift and 98 in elbow_flex, so the spot is not stable yet.",
        "S1 takes that only pick and lift (eps 0-2 end holding in the air) do not count for S1; put the brick down and return home.",
        "S2 (carry to BLUE): record at least 25 (30 target) takes that hold the brick at the start and release it over the blue plate, then return home before ending. Ep 7 has no return.",
        "S3 (carry to GREEN): record at least 24 more (29 target) with the brick in the jaws at the start (ep 9 start is ambiguous), release over the green plate, return home.",
        "Task text: eps 8-10 say 'move the brick to the left plate' but release on the GREEN plate (right side of the top view). Decide: relabel eps 8-10 as green-plate takes, or re-record with the correct text. Do not train a text-conditioned S2 from them.",
        "Use the colour of the plate in the text (blue plate / green plate), not left/right, unless Hassan confirms the camera convention.",
        "Good clean 2x2 and defective (tape or not 2x2) bricks in every skill; vary brick start positions across the cluster; plates stay fixed.",
    ],
)
gaps["recording_requests"] = gaps["recording_requests"]
(REP / "gaps.json").write_text(json.dumps(gaps, indent=1, default=str))


def fmt(x):
    return "" if x is None else str(x)


cols = ["episode_index", "task_index", "task_text", "n_frames", "duration_s", "start_held", "start_plateau",
        "type", "verdict_auto", "verdict", "plate_final", "plate_basis", "grasp_start_s", "grasp_close_end_s",
        "grasp_plateau", "open_level", "lift_s", "n_held_grasps", "n_empty_grasps", "n_releases", "release_onset_s",
        "release_pan", "release_plate_cand", "release_pose", "returned_home_s", "start_dist_home", "end_dist_home",
        "idle_start_s", "idle_end_s", "frac_stationary", "top_file", "top_from_s", "top_to_s", "side_file",
        "side_from_s", "side_to_s", "reasons", "qa"]
csv_lines = [",".join(["dataset_id"] + cols)]
for r in recs:
    gf = r["grasps"][0] if r["grasps"] else {}
    lr = r["releases"][-1] if r["releases"] else {}
    vals = dict(
        episode_index=r["episode_index"], task_index=r["task_index"], task_text=r["task_text"],
        n_frames=r["n_frames"], duration_s=r["duration_s"], start_held=r["start_held"],
        start_plateau=r["start_plateau"], type=r["type"], verdict_auto=r["verdict_auto"], verdict=r["verdict"],
        plate_final=lr.get("plate_final", "-") if lr else "-", plate_basis=r["plate_basis"],
        grasp_start_s=round(gf["start"] / FPS, 3) if gf else "",
        grasp_close_end_s=round(gf["t"] / FPS, 3) if gf else "",
        grasp_plateau=gf.get("plateau", ""), open_level=gf.get("open_level", ""),
        lift_s=round(gf["lift_t"] / FPS, 3) if gf and gf.get("lift_t") is not None else "",
        n_held_grasps=len(r["grasps"]), n_empty_grasps=len(r["empties"]), n_releases=len(r["releases"]),
        release_onset_s=round(lr["t"] / FPS, 3) if lr else "", release_pan=lr.get("pan", "") if lr else "",
        release_plate_cand=lr.get("plate_cand", "") if lr else "",
        release_pose=" ".join(str(x) for x in lr["pose"]) if lr else "",
        returned_home_s=round(lr["returned_t"] / FPS, 3) if lr and lr.get("returned_t") is not None else "",
        start_dist_home=r["start_dist_home"], end_dist_home=r["end_dist_home"],
        idle_start_s=fmt(r["idle_start_s"]), idle_end_s=fmt(r["idle_end_s"]), frac_stationary=r["frac_stationary"],
        top_file=r["video"]["top"]["file"], top_from_s=r["video"]["top"]["from_s"], top_to_s=r["video"]["top"]["to_s"],
        side_file=r["video"]["side"]["file"], side_from_s=r["video"]["side"]["from_s"],
        side_to_s=r["video"]["side"]["to_s"],
        reasons=" | ".join(r["reasons"]).replace(",", ";"), qa="; ".join(r["qa"]),
    )
    row = [str(r["dataset_id"])] + [str(vals[c]).replace(",", ";") for c in cols]
    csv_lines.append(",".join(row))
(REP / "episodes.csv").write_text("\n".join(csv_lines) + "\n")

hold = [r for r in recs if r["type"].startswith("place-only")]
calib["v3_hold_pose_start_place_only"] = dict(
    episodes=[r["episode_index"] for r in hold], joints=calib["home_start_pose"]["joints"],
    median=np.round(np.median([r["start_pose"] for r in hold], axis=0), 2).tolist(),
    min=np.round(np.min([r["start_pose"] for r in hold], axis=0), 2).tolist(),
    max=np.round(np.max([r["start_pose"] for r in hold], axis=0), 2).tolist(),
    note="start pose of eps 8-10 (arm away from home, holding a brick); used as the S2/S3 hold position")
calib["plate_decisions_v3"] = {str(r["episode_index"]): (r["releases"][-1]["plate_final"] if r["releases"] else None)
                               for r in recs}
calib["placed_brick_pixels"] = ("V3 did not measure pixel centres numerically. Visual check of top view at put-down "
                                "times (eps 3-6) places bricks in the upper-middle mat between the plates, consistent "
                                "in region with the W3 centres; not verified to the pixel.")
calib["text_label_check"] = dict(eps_8_10_task_text="move the brick to the left plate",
                                 eps_8_10_physical_destination="GREEN plate (right side of top view)",
                                 action="relabel or re-record before text-conditioned training")
calib["v3_notes"] = ["pan threshold BLUE>=15 / GREEN<=-35 / MAT otherwise is consistent with all image-checked releases",
                     "release pose spread on GREEN is large in lift and elbow (3 takes), pan is consistent (5.6 units)",
                     "closing direction and plateau rules hold on all 8 closures; empty closure 0.98 on ep 9"]
(REP / "calib.json").write_text(json.dumps(calib, indent=1, default=str))

L = []
L.append("# INVENTORY (V3 independent check, P2-D-DATA)")
L.append("")
L.append(f"Dataset {DSID} (LeRobot v3.0, SO101_Follower, 30 fps). Snapshot: sync log last entry 2026-11-14T13:23:07Z; 11 episodes, 6693 frames on disk.")
L.append("")
L.append("## Studio and dataset facts")
L.append("- Studio project 'Junction 2026' (ab767473-d6fd-47ff-812e-8239c455dabb) lists one dataset: 6d6bdce3, named 'Smoke test - inspection skill - 5 episodes' (stale: 11 episodes on disk), default_task 'move the brick to the inspection spot'.")
L.append("- tasks.parquet has 2 tasks: 0 'move the brick to the inspection spot' (eps 0-7), 1 'move the brick to the left plate' (eps 8-10).")
L.append("- Stored joint names: shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos.")
L.append("- Videos: 2 keys x 3 files; ffprobe count_frames = 1253 / 3113 / 2327 per file, equal to data rows; AV1 640x480 30 fps; all 6 decode.")
L.append("- Top view key 'realsense 243622061421' (/dev/video8); close low view 'realsense 243722064316'.")
L.append("")
L.append("## Per-episode (V3 final)")
L.append("| ep | frames | dur s | task idx | held at start | type | verdict | plate | release onset s | returned home s | reasons |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|")
for r in recs:
    lr = r["releases"][-1] if r["releases"] else None
    plate = lr["plate_final"] if lr else "-"
    rel = f"{lr['t'] / FPS:.1f}" if lr else "-"
    ret = f"{lr['returned_t'] / FPS:.1f}" if lr and lr["returned_t"] is not None else "-"
    held0 = f"yes ({r['start_plateau']})" if r["start_held"] else "no"
    rs = "; ".join(r["reasons"]) or "-"
    L.append(f"| {r['episode_index']} | {r['n_frames']} | {r['duration_s']:.1f} | {r['task_index']} | {held0} | {r['type']} | {r['verdict']} | {plate} | {rel} | {ret} | {rs} |")
L.append("")
L.append("## Needs (OD-9) and gaps")
L.append("| need | target | minimum | OK | WARN | shortfall to target (OK only / OK+WARN) | shortfall to minimum (OK only / OK+WARN) |")
L.append("|---|---|---|---|---|---|---|")
for k, d in needs.items():
    L.append(f"| {k} | {d['target']} | {d['minimum']} | {d['ok_n']} {d['ok']} | {d['warn_n']} {d['warn']} | "
             f"{d['shortfall_target_ok_only']} / {d['shortfall_target_ok_plus_warn']} | "
             f"{d['shortfall_min_ok_only']} / {d['shortfall_min_ok_plus_warn']} |")
L.append("")
L.append("## Calibration summary (calib.json)")
g = calib["gripper"]
L.append(f"- Gripper open before grasp: median {g['open_before_grasp']['median']} (n={g['open_before_grasp']['n']}, p05 {g['open_before_grasp']['p05']}, p95 {g['open_before_grasp']['p95']}).")
L.append(f"- Closed on brick (plateau): median {g['closed_on_brick_plateau']['median']} (n={g['closed_on_brick_plateau']['n']}, incl. 3 start-held eps). Closed on nothing: {g['closed_on_nothing_plateau']['median']} (ep 9).")
L.append(f"- Open after release: median {g['open_after_release']['median']} (n={g['open_after_release']['n']}). Closing = value falls.")
L.append(f"- Home (start median, 11 eps): {calib['home_start_pose']['median']}; spread min {calib['home_start_pose']['min']}, max {calib['home_start_pose']['max']}.")
L.append(f"- BLUE drop (ep 7 only): {calib['blue_drop']['median'] if calib['blue_drop'] else None}.")
L.append(f"- GREEN drop (eps 8-10): median {calib['green_drop']['median']}; min {calib['green_drop']['min']}; max {calib['green_drop']['max']}.")
L.append(f"- MAT put-down (eps 3-6): median {calib['spot_put_down_mat']['median']}; std {calib['spot_put_down_mat']['std']}. NOT stable.")
L.append(f"- Hold pose (start of eps 8-10): median {calib['v3_hold_pose_start_place_only']['median']}.")
L.append(f"- Stationary: arm speed < {calib['stationary']['arm_speed_units_per_s']} units/s, dwell 0.5 s; idle threshold {calib['stationary']['idle_speed_units_per_s']} units/s.")
L.append("")
L.append("## Disagreements between workers and V3 resolution")
L.append("| item | W1 | W2 | W3 | V3 resolution |")
L.append("|---|---|---|---|---|")
L.append("| ep 7 plate | unresolved | BLUE (image) | BLUE (moderate) | BLUE: release over blue at 40.2 s, brick on blue at 43.4 s |")
L.append("| ep 8 plate | unresolved | 'rests on blue' (image) | GREEN (low-moderate) | GREEN: release over green 48.3 s; white 2x2 on green by 49.1 s; blue brick predates ep 8 |")
L.append("| ep 9 | unusable | closed on nothing (0.98) | BAD, nothing in jaws | WARN place-only-GREEN: start plateau 1.96 is above empty 0.98; white part under jaws (moderate) |")
L.append("| ep 10 plate | not listed | GREEN (image) | GREEN (low) | GREEN: release over green 71.5 s; blue 2x2 on green right edge by 77.2 s |")
L.append("| eps 0-2 | hold in air | not listed | S1 OK under OD-8a | S1-hold: pick and lift only, not S1 under OD-9 |")
L.append("| ep 4 | put-down, no return | not listed | S2/S3 BAD (over cluster) | S1-place-noreturn WARN |")
L.append("| S1 counts | 4 usable | not listed | OK 5 / WARN 3 (OD-8a) | OD-9: OK 3 (eps 3,5,6), WARN 1 (ep 4) |")
L.append("| task text | not reported | not reported | '2 tasks' only | eps 8-10 'left plate' text vs GREEN destination (flagged) |")
L.append("| spot | median eps 3,5,6 | not stable | not stable (lift 81, elbow 98) | agree: not stable |")
L.append("| open / closed levels | open 22.1, closed 2.55 | open peak 22.5, closed 2.547 (n=9) | open 21.2, closed 2.547 (n=9) | agree within method; V3 open 21.45 (8 grasps), closed 2.48 (11 incl. start-held) |")
L.append("| hover pose 8 cm | not derived | not derived | not derived | still not derived (needs forward kinematics) |")
L.append("")
L.append("## Recording requests (for Opus to relay to Hassan)")
for x in gaps["recording_requests"]:
    L.append(f"- {x}")
L.append("")
L.append("## Caveats")
L.append("- Plate identity for eps 7-10 is from V3 frames (top view, 640x480 full-res for eps 8-9; 320x240 tiles for others), not from a per-frame colour mask.")
L.append("- Ep 9 start holding is inferred from the gripper plateau (1.96 vs empty 0.98) and a visual part under the jaws; moderate confidence.")
L.append("- Put-down spot pixel centres were not measured numerically.")
L.append("- The dataset is still growing in principle; this is the 13:23Z snapshot. Re-run before any gap is acted on.")
L.append("")
L.append("## Coverage checklist")
for e in recs:
    L.append(f"- [x] episode {e['episode_index']} ({e['type']}, {e['verdict']})")
L.append("- [x] meta/info.json, meta/tasks.parquet, meta/episodes (3 files), data (3 files), stats not used")
L.append("- [x] 6 videos decoded for frame count (ffprobe)")
L.append("- [x] Studio GET /api/projects and /api/dataset/6d6bdce3 (names only)")
L.append("- [x] 16 full-res or tile frames reviewed for plate identity and put-down")
L.append("- [ ] per-dataset key-frame contact sheets (both views, grasp, lift, end) not produced by V3")
L.append("- [ ] placed-brick pixel centres not measured numerically")
L.append("- [ ] watch mode not started (V1 only, by protocol)")
(REP / "INVENTORY.md").write_text("\n".join(L) + "\n")
print("V3_CONSOLIDATE_OK", REP)
print("NEEDS", json.dumps({k: [d["ok_n"], d["warn_n"], d["shortfall_target_ok_only"], d["shortfall_min_ok_only"]] for k, d in needs.items()}))
print("TYPES", json.dumps(gaps["type_counts"]))
