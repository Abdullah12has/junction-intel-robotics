import glob, json, os, subprocess, datetime
import numpy as np
import pandas as pd
from PIL import Image

R = "/home/intel/intel-junction-2026"
E2 = R + "/work/P3-LABEL-STUDIO/E2"
E1 = R + "/work/P3-LABEL-STUDIO/E1"
SD = R + "/data/studio-datasets"
FPS = 30
TOP = "observation.images.realsense 243622061421"
CLOSE = "observation.images.realsense 243722064316"
EXT = "observation.images.usb2_0_cam1 external camera"
CAMNAME = {TOP: "top", CLOSE: "close", EXT: "ext"}
UTC = "2026-11-14T21:05Z"

census = json.load(open(E2 + "/out/census_e2.json"))
CE = {(e["dataset"][:8], e["episode_index"]): e for e in census["episodes"]}
CD = {k[:8]: v for k, v in census["datasets"].items()}
DIRS = {d[:8]: d for d in os.listdir(SD) if os.path.isdir(os.path.join(SD, d))}


def load(prefix):
    base = os.path.join(SD, DIRS[prefix])
    meta = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(base, "meta/episodes/chunk-*/file-*.parquet")))], ignore_index=True)
    data = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(base, "data/chunk-*/file-*.parquet")))], ignore_index=True)
    return base, meta, data


def sanity(act, st):
    best = None
    for lag in range(0, 7):
        a = act[: len(act) - lag] if lag else act
        s = st[lag:]
        n = min(len(a), len(s))
        mae = float(np.mean(np.abs(a[:n] - s[:n])))
        if best is None or mae < best[1]:
            best = (lag, mae)
    body = act[:, :5]
    grip = act[:, 5]
    lim = []
    names = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos"]
    for j, nm in enumerate(names):
        c_hi = int(np.sum(np.isclose(act[:, j], 100.0)))
        c_lo = int(np.sum(np.isclose(act[:, j], -100.0)))
        if c_hi or c_lo:
            lim.append("%s at +100 x%d / -100 x%d" % (nm, c_hi, c_lo))
    inb = bool(np.all(body >= -100.0001) and np.all(body <= 100.0001) and np.all(grip >= -0.0001) and np.all(grip <= 100.0001))
    return dict(lag_best_frames=best[0], mae_lag_best=round(best[1], 3), action_range_in_bounds=inb,
                gripper_range=[round(float(grip.min()), 2), round(float(grip.max()), 2)],
                saturated_at_normalized_limit=lim, census_flags=[])


def frame(base, cam, mrow, idx, tmp):
    fi = int(mrow["videos/%s/file_index" % cam])
    ts0 = float(mrow["videos/%s/from_timestamp" % cam])
    mp4 = os.path.join(base, "videos", cam, "chunk-000", "file-%03d.mp4" % fi)
    subprocess.run(["ffmpeg", "-v", "error", "-ss", "%.4f" % (ts0 + idx / FPS), "-i", mp4, "-frames:v", "1", "-y", tmp], capture_output=True)
    return np.asarray(Image.open(tmp).convert("RGB"), dtype=np.float64)


# ---- visual notes (by eye, top camera full-res and zoom grids) ----
def B(id_, colour, shape, tape, placed, expected, correct):
    return dict(id=id_, colour=colour, shape=shape, tape=tape, placement=placed, od10_expected=expected, decision_correct=correct)

NOTES = {
    "8a8bb05e:0": dict(P="P18", bricks=[
        B("B1", "white", "2x2", "no", "BLUE", "BLUE", "yes"),
        B("B2", "orange", "2x2", "no", "BLUE", "BLUE", "yes"),
        B("B3", "orange", "2x2", "no", "BLUE", "BLUE", "yes"),
        B("B4", "red", "2x2", "no", "BLUE", "BLUE", "yes"),
        B("B5", "yellow", "2x2", "yes (grey strip, left side)", "BLUE", "GREEN", "no"),
        B("B6", "white", "2x4", "yes (grey strip, top left)", "BLUE", "GREEN", "no"),
        B("B7", "blue", "2x2", "unclear (small dark mark)", "BLUE", "unknown", "unknown")],
        placed_all=True, outcome="success (7 of 7 on BLUE)", wrong_text="2 defect bricks on BLUE: yellow 2x2 with tape, white 2x4"),
    "8a8bb05e:1": dict(P="P19", bricks=[
        B("B1", "white", "2x4", "yes (grey tape, top middle)", "BLUE", "GREEN", "no"),
        B("B2", "blue", "2x2", "no", "BLUE", "BLUE", "yes"),
        B("B3", "red", "2x2", "no", "BLUE", "BLUE", "yes"),
        B("B4", "white", "2x2", "no", "BLUE", "BLUE", "yes"),
        B("B5", "orange", "2x2", "no", "BLUE", "BLUE", "yes"),
        B("B6", "yellow", "2x2", "yes (grey strip, left side)", "BLUE", "GREEN", "no"),
        B("B7", "orange", "2x2", "no", "BLUE", "BLUE", "yes")],
        placed_all=True, outcome="success (7 of 7 on BLUE)", wrong_text="2 defect bricks on BLUE: white 2x4 with tape, yellow 2x2 with tape"),
    "8a8bb05e:2": dict(P="P20", bricks=[
        B("B1", "blue", "2x2", "yes (grey strip, right side)", "BLUE", "GREEN", "no"),
        B("B2", "orange", "2x2", "no", "BLUE", "BLUE", "yes"),
        B("B3", "orange", "2x2", "no", "BLUE", "BLUE", "yes"),
        B("B4", "yellow", "2x2", "yes (grey strip, left side)", "BLUE", "GREEN", "no"),
        B("B5", "white", "2x4", "yes (grey tape, top)", "BLUE", "GREEN", "no"),
        B("B6", "red", "2x2", "no", "table (not placed)", "BLUE", "n/a (not placed)"),
        B("B7", "white", "2x2", "no", "table (not placed)", "BLUE", "n/a (not placed)")],
        placed_all=False, outcome="partial (5 of 7 on BLUE; red 2x2 and white 2x2 left on the table)", wrong_text="3 defect bricks on BLUE: blue 2x2 with tape, yellow 2x2 with tape, white 2x4"),
    "8a8bb05e:3": dict(P="P21", bricks=[
        B("B1", "blue", "2x2", "yes (dark strip, bottom)", "BLUE", "GREEN", "no"),
        B("B2", "yellow", "2x2", "yes (grey strip, left side)", "BLUE", "GREEN", "no"),
        B("B3", "orange", "2x2", "no", "BLUE", "BLUE", "yes"),
        B("B4", "red", "2x2", "no", "BLUE", "BLUE", "yes"),
        B("B5", "orange", "2x2", "no", "BLUE", "BLUE", "yes"),
        B("B6", "white", "2x2", "no", "BLUE", "BLUE", "yes"),
        B("B7", "white", "2x4", "yes (grey tape, top)", "BLUE", "GREEN", "no")],
        placed_all=True, outcome="success (7 of 7 on BLUE)", wrong_text="3 defect bricks on BLUE: blue 2x2 with tape, yellow 2x2 with tape, white 2x4"),
    "8a8bb05e:4": dict(P="P22", bricks=[
        B("B1", "white", "2x4", "yes (grey tape, top)", "BLUE", "GREEN", "no"),
        B("B2", "yellow", "2x2", "yes (grey tape, top)", "BLUE", "GREEN", "no"),
        B("B3", "blue", "2x2", "no", "BLUE", "BLUE", "yes"),
        B("B4", "white", "2x2", "no", "BLUE", "BLUE", "yes"),
        B("B5", "orange", "2x2", "no", "BLUE", "BLUE", "yes"),
        B("B6", "orange", "2x2", "no", "BLUE", "BLUE", "yes"),
        B("B7", "red", "2x2", "no", "BLUE", "BLUE", "yes")],
        placed_all=True, outcome="success (7 of 7 on BLUE)", wrong_text="2 defect bricks on BLUE: white 2x4 with tape, yellow 2x2 with tape"),
    "8a8bb05e:5": dict(P="P23", bricks=[
        B("B1", "white", "2x4", "yes (grey tape, top)", "BLUE", "GREEN", "no"),
        B("B2", "yellow", "2x2", "yes (grey tape, top)", "BLUE", "GREEN", "no"),
        B("B3", "blue", "2x2", "no", "BLUE", "BLUE", "yes"),
        B("B4", "white", "2x2", "no", "BLUE", "BLUE", "yes"),
        B("B5", "red", "2x2", "no", "BLUE", "BLUE", "yes"),
        B("B6", "orange", "2x2", "no", "BLUE", "BLUE", "yes"),
        B("B7", "orange", "2x2", "no", "BLUE", "BLUE", "yes")],
        placed_all=True, outcome="success (7 of 7 on BLUE)", wrong_text="2 defect bricks on BLUE: white 2x4 with tape, yellow 2x2 with tape"),
}
COPY_OF = {"e60f4087:0": "8a8bb05e:0", "e60f4087:1": "8a8bb05e:1", "e60f4087:2": "8a8bb05e:3",
           "e60f4087:3": "8a8bb05e:4", "e60f4087:4": "8a8bb05e:5"}

# ---- facts, joint sanity, copy pixel checks ----
facts = {}
for pre in ["8a8bb05e", "e60f4087"]:
    base, meta, data = load(pre)
    for ep, g in data.sort_values(["episode_index", "frame_index"]).groupby("episode_index"):
        act = np.stack([np.asarray(x, dtype=np.float64) for x in g["action"].values])
        st = np.stack([np.asarray(x, dtype=np.float64) for x in g["observation.state"].values])
        m = meta[meta.episode_index == ep].iloc[0]
        key = "%s:%d" % (pre, ep)
        facts[key] = dict(base=base, mrow=m, act=act, st=st, n=len(act), sanity=sanity(act, st),
                          ce=CE[(pre, ep)], cams=[c for c in [TOP, CLOSE, EXT] if os.path.isdir(os.path.join(base, "videos", c))])

pix = {}
tmp = E2 + "/out/_pix_tmp.png"
for copy, rep in COPY_OF.items():
    fc, fr = facts[copy], facts[rep]
    ce = fc["ce"]
    off = ce["window"][0] if ce["window"] else 0
    mae = {}
    for cam in fc["cams"]:
        if cam not in fr["cams"]:
            continue
        vals = []
        for k in [60, fc["n"] // 2]:
            a = frame(fc["base"], cam, fc["mrow"], k, tmp)
            b = frame(fr["base"], cam, fr["mrow"], off + k, tmp)
            vals.append(round(float(np.mean(np.abs(a - b))), 3))
        mae[CAMNAME[cam]] = vals
    allv = [v for vs in mae.values() for v in vs]
    if allv and max(allv) == 0.0:
        verdict = "identical_decode"
    elif allv and max(allv) <= 4.0:
        verdict = "pixel_match_under_threshold"
    else:
        verdict = "unverified"
    pix[copy] = dict(offset=off, mae=mae, verdict=verdict)
    print("pixel", copy, "vs", rep, "offset", off, mae, verdict)
os.remove(tmp)
json.dump(dict(pixel=pix, sanity={k: v["sanity"] for k, v in facts.items()}), open(E2 + "/out/e2_pixel_sanity.json", "w"), indent=1)

# ---- new label rows ----
JN = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
rows = []
for key in ["8a8bb05e:%d" % i for i in range(6)] + ["e60f4087:%d" % i for i in range(5)]:
    f = facts[key]
    pre, ep = key.split(":")
    ep = int(ep)
    ce = f["ce"]
    is_copy = key in COPY_OF
    nt = NOTES.get(COPY_OF.get(key, key)) if True else None
    rep_key = COPY_OF.get(key, key)
    nrep = NOTES[rep_key]
    bricks = nrep["bricks"]
    wrong = [b for b in bricks if b["decision_correct"] == "no"]
    unk = [b for b in bricks if b["decision_correct"] == "unknown"]
    corr = [b for b in bricks if b["decision_correct"] == "yes"]
    placed = [b for b in bricks if b["placement"] == "BLUE"]
    row = dict(
        entry=key, dataset=DIRS[pre], dataset_prefix=pre, episode_index=ep,
        physical_episode=nrep["P"] if not is_copy else NOTES[rep_key]["P"],
        unit_group=("E2-%s" % nrep["P"]) if not is_copy else ("E2-%s" % nrep["P"]),
        relation=("representative (original Studio episode, new folder 8a8bb05e)" if not is_copy else
                  "window [%d:%d] of %s (identical action rows)" % (ce["window"][0], ce["window"][1], rep_key) if ce["window"] and ce["window"][0] != 0 or (ce["window"] and ce["window"][1] != facts[rep_key]["n"]) else "full copy of %s (identical action rows)" % rep_key),
        dup_of=None if not is_copy else rep_key, representative_entry=rep_key,
        fps=30, n_frames=f["n"], dur_s=round(f["n"] / FPS, 3), t0_s=0.0,
        cameras_top_serial=TOP.split()[-1], cameras_close_serial=CLOSE.split()[-1], cameras_ext="usb2_0_cam1 external camera (serial not in info)",
        task_text="pick up all the blocks and put them on the blue plate", task_text_expected_plate="BLUE (text asks all bricks on the blue plate)",
        task_text_ok=("match (placed bricks on BLUE, as the text says)" if nrep["placed_all"] else "incomplete (2 bricks left on the table)"),
        bricks=bricks, bricks_involved=[("%s %s %s" % (b["colour"], b["shape"], "(tape)" if b["tape"].startswith("yes") else "(no tape seen)")) for b in bricks],
        target_brick=None, outcome=nrep["outcome"], failure_category=nrep["wrong_text"],
        where_brick_ends="%d bricks on the BLUE plate; GREEN plate empty; %s" % (len(placed), "table clear" if nrep["placed_all"] else "2 bricks on the table right of the plate"),
        release_s_in_rep=None, decision_plate="BLUE (all placed bricks)",
        od10_correct_plate="per brick: BLUE for clean 2x2, GREEN for tape or non-2x2",
        decision_correct="partial (%d of %d placed bricks wrong%s)" % (len(wrong), len(placed), ", %d unknown" % len(unk) if unk else ""),
        label_source="visual (top camera full-res start and end frames; zoom grids out/zoom_8a8bb05e_start.png and _end.png; montages out/mont_8a8bb05e_ep*.jpg)",
        confidence="medium (tape calls at top-view zoom; not checked at close range per brick)",
        evidence="out/full_8a8bb05e_ep%d.png, out/zoom_8a8bb05e_start.png, out/zoom_8a8bb05e_end.png, out/mont_8a8bb05e_ep%d.jpg" % (int(rep_key.split(":")[1]), int(rep_key.split(":")[1])),
        notes=("Multi-brick episode: %d bricks on the table at the start, all placed on the BLUE plate except the bricks not placed. Copy of %s: same action rows; video pixel check %s." % (len(bricks), rep_key, pix[key]["verdict"]) if is_copy else
               "Multi-brick episode (%d bricks, %d placed on BLUE). New physical episode in folder 8a8bb05e (not in E1)." % (len(bricks), len(placed))),
        joint_sanity=dict(f["sanity"]),
        timestamp_frame_alignment=dict(frame_index_contiguous=bool(ce["frame_index_contiguous"]), video_span_matches_rows=None,
                                       meta_length_matches_rows=bool(ce["meta_length"] == f["n"])),
        trust="evaluation",
        usable_for=["evaluation"],
        trust_reason="OD-10 decision wrong for defect bricks on BLUE (evaluation only); joint data sane" + ("; %d bricks not placed" % (len(bricks) - len(placed)) if len(placed) < len(bricks) else ""),
        frame_check_vs_rep=(dict(frame_offset=pix[key]["offset"], mae_by_cam=pix[key]["mae"], same_view=True, frame_verdict=pix[key]["verdict"], pixel_offset_check=None)
                            if is_copy else dict(frame_verdict="own frames", note="viewed by eye")),
    )
    rows.append(row)

# ---- write labels file: E1 rows (unchanged, copied byte for byte) + 11 new rows ----
e1_lines = open(E1 + "/labels_episodes.jsonl", encoding="utf-8").read().splitlines()
assert len(e1_lines) == 41
new_lines = [json.dumps(r, ensure_ascii=True) for r in rows]
open(E2 + "/labels_episodes.jsonl", "w", encoding="utf-8", newline="\n").write("\n".join(e1_lines + new_lines) + "\n")
print("labels rows", len(e1_lines) + len(new_lines))
json.dump(rows, open(E2 + "/out/new_rows_e2.json", "w"), indent=1)

# ---- mismatches: E1 rows + new ----
e1_mm = open(E1 + "/mismatches.tsv", encoding="utf-8").read().splitlines()
mm = list(e1_mm)
for r in rows:
    nrep = NOTES[COPY_OF.get(r["entry"], r["entry"])]
    wrongs = [b for b in nrep["bricks"] if b["decision_correct"] == "no"]
    desc = nrep["wrong_text"] if wrongs else ""
    phys = r["physical_episode"]
    kind = "od10_decision"
    if r["entry"] == "8a8bb05e:2":
        mm.append("\t".join(["not_placed", r["entry"], phys, "move all the blocks onto the blue plate", "stopped: red 2x2 and white 2x2 left on the table", "red 2x2 and white 2x2 expected BLUE; not placed"]))
    mm.append("\t".join([kind, r["entry"], phys, "decision BLUE (all placed)", "OD-10 wrong for %d of %d placed bricks" % (len(wrongs), len([b for b in nrep["bricks"] if b["placement"] == "BLUE"])), desc + ("; copy of %s" % COPY_OF[r["entry"]] if r["entry"] in COPY_OF else "")]))
open(E2 + "/mismatches.tsv", "w", encoding="utf-8", newline="\n").write("\n".join(mm) + "\n")
print("mismatch lines", len(mm))

# ---- datasets.json ----
dj = json.load(open(E1 + "/datasets.json", encoding="utf-8"))
for pre in ["8a8bb05e", "e60f4087"]:
    d = CD[pre]
    ents = [k for k in CE if k[0] == pre]
    ce_list = sorted([CE[k] for k in ents], key=lambda x: x["episode_index"])
    jmin = np.min(np.vstack([facts["%s:%d" % (pre, k[1])]["act"] for k in ents]), axis=0).round(2).tolist()
    jmax = np.max(np.vstack([facts["%s:%d" % (pre, k[1])]["act"] for k in ents]), axis=0).round(2).tolist()
    entries = ["%s:%d" % (pre, k[1]) for k in ents]
    studio = dict(group="studio", uuid=DIRS[pre], fps=30, robot_type="SO101_Follower", codebase_version="v3.0",
                  cameras={c: dict(serial=c.split()[-1], role=CAMNAME[c]) for c in d["cameras"]},
                  joint_names=JN, joint_units="not stated in info.json", joint_minmax_action=[[a, b] for a, b in zip(jmin, jmax)],
                  episodes_info_json=d["total_episodes"], episodes_meta_rows=d["meta_rows"], data_rows=d["data_rows"],
                  tasks=d["tasks"], episode_indices=[k[1] for k in ents], episode_frames=[x["n"] for x in ce_list],
                  duration_s_total=round(sum(x["n"] for x in ce_list) / FPS, 2), duration_s_min=round(min(x["n"] for x in ce_list) / FPS, 2),
                  duration_s_max=round(max(x["n"] for x in ce_list) / FPS, 2),
                  originals_in_dataset=([e for e in entries if pre == "8a8bb05e"]), entries=entries,
                  trust_counts=dict(training=0, evaluation=len(entries), calibration=0, nothing=0),
                  census_issues=["frame_index contiguous and meta lengths equal rows for all episodes (census_e2.json)"],
                  note=("New in the mirror at 2026-11-14T20:44Z: 6 physical multi-brick episodes (pick up all blocks to blue plate), 3 cameras incl. external usb2_0_cam1."
                        if pre == "8a8bb05e" else "New in the mirror at 2026-11-14T20:44Z: 5 copies or windows of 8a8bb05e:0,1,3,4,5; adds no physical episode."))
    dj["datasets"].append(studio)
dj["stage"] = "studio-labels-v2"
dj["utc"] = UTC
dj["snapshot_utc"] = "2026-11-14T20:44Z"
dj["note"] = "E2: mirror census at 20:44Z: 11 Studio folders, 52 episode rows. E1 datasets kept; 8a8bb05e and e60f4087 added."
json.dump(dj, open(E2 + "/datasets.json", "w", encoding="utf-8"), indent=1)
print("datasets", len(dj["datasets"]))

# ---- metrics ----
e1m = json.load(open(E1 + "/metrics.json", encoding="utf-8"))
M = dict(episodes_total=52, episodes_total_physical_unique=23, datasets_studio=11, curated_sets=7, curated_episodes=24,
         curated_episodes_matched_exactly=24, curated_check="not re-run in E2 (curated sets unchanged)")
ver = [pix[r["entry"]]["verdict"] for r in rows if r["entry"] in COPY_OF]
M["verified"] = 41 + 6 + sum(1 for v in ver if v in ("identical_decode", "pixel_match_under_threshold", "reencoded_same_scene"))
M["verified_by_own_frames"] = 17 + 6
M["verified_copies_pixel_identical"] = 20 + sum(1 for v in ver if v == "identical_decode")
M["verified_copies_reencoded_visual_only"] = 4 + sum(1 for v in ver if v == "pixel_match_under_threshold")
M["copies_unverified"] = sum(1 for v in ver if v == "unverified")
M["copies_checked_by_frames"] = 24 + len(ver)
M["mismatched"] = 12 + 11
M["task_text_mismatch"] = 4
M["od10_decision_wrong"] = 9 + 11
M["od10_wrong_bricks_new_physical"] = sum(len([b for b in NOTES[k]["bricks"] if b["decision_correct"] == "no"]) for k in NOTES)
M["usable_training"] = 23
M["usable_evaluation"] = 12 + 11
M["usable_calibration_only"] = 5
M["nothing"] = 1
M["usable_training_or_evaluation"] = 23 + 23
M["physical_trust_training"] = 8
M["physical_trust_evaluation_only"] = 5 + 6
M["physical_trust_calibration_only"] = 3
M["physical_trust_nothing"] = 1
M["duplicate_entries"] = 24 + 5
M["incomplete_or_cut"] = e1m["metrics"].get("incomplete_or_cut", None)
prev = e1m["metrics"]
delta = {k: (M[k] - prev[k]) for k in M if isinstance(M[k], int) and isinstance(prev.get(k), int)}
out = dict(stage="studio-labels-v2", utc=UTC, snapshot_utc="2026-11-14T20:44Z", metrics=M,
           previous=dict(stage="studio-labels-v1", utc=e1m["utc"], metrics=prev), delta=delta)
json.dump(out, open(E2 + "/metrics.json", "w", encoding="utf-8"), indent=1)
print("metrics", json.dumps({k: M[k] for k in ["verified", "mismatched", "od10_decision_wrong", "usable_training", "usable_evaluation", "usable_calibration_only", "nothing", "copies_unverified"]}))
print("verdicts", ver)
