#!/usr/bin/env python3
# V3 independent check of one LeRobot v3.0 Studio dataset. Input is read-only; output goes to OUT only.
# usage: python3 -I v3_inventory.py DATASET_DIR OUT_DIR [overrides.json]
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

DS = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
OVR = json.loads(Path(sys.argv[3]).read_text()) if len(sys.argv) > 3 else {}

FPS = 30.0
K = 6            # gripper slope window (0.2 s)
THR = 4.0        # gripper.pos change over K frames that counts as closing or opening
HLO, HHI = 1.5, 3.5   # gripper plateau that means brick between the jaws
GAP = 8          # merge event runs closer than this many frames
MOVE_V = 1.0     # arm speed (units/s, 5-frame median) below which the arm is stationary
IDLE_V = 3.0     # arm speed above which the arm is moving (idle-time measure)
HOME_TOL = 6.0   # max abs diff over the 5 arm joints to count as at home
LIFT_TOL = 4.0   # max abs diff over joints 0-3 that counts as lift onset
PAN_BLUE, PAN_GREEN = 15.0, -35.0
JN = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
TOPK = "observation.images.realsense 243622061421"
SIDEK = "observation.images.realsense 243722064316"
DSID = DS.name


def tolist(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


info = json.loads((DS / "meta/info.json").read_text())
tk = pd.read_parquet(DS / "meta/tasks.parquet").reset_index()
TASK = {int(r.task_index): str(r.task) for r in tk.itertuples()}
epm = (pd.concat([pd.read_parquet(p) for p in sorted((DS / "meta/episodes").glob("chunk-*/file-*.parquet"))],
                 ignore_index=True).sort_values("episode_index").reset_index(drop=True))
dat = pd.concat([pd.read_parquet(p) for p in sorted((DS / "data").glob("chunk-*/file-*.parquet"))], ignore_index=True)
S_ALL = np.stack(dat["observation.state"].to_numpy()).astype(float)
EPI = dat["episode_index"].to_numpy()
FI = dat["frame_index"].to_numpy()
TS = dat["timestamp"].to_numpy().astype(float)
TI = dat["task_index"].to_numpy()


def runs(mask, gap=GAP, minlen=2):
    idx = np.flatnonzero(mask)
    out = []
    if idx.size == 0:
        return out
    s = p = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - p > gap:
            if p - s + 1 >= minlen:
                out.append((s, p))
            s = i
        p = i
    if p - s + 1 >= minlen:
        out.append((s, p))
    return out


def med(x, w=5):
    return pd.Series(x).rolling(w, center=True, min_periods=1).median().to_numpy()


def events_for(S):
    n = len(S)
    J = S[:, :5]
    g = S[:, 5]
    v = med(np.r_[0.0, np.linalg.norm(np.diff(J, axis=0), axis=1) * FPS])
    dg = np.zeros(n)
    dg[:-K] = g[K:] - g[:-K]
    evs = [("close", s, t) for s, t in runs(dg < -THR)] + [("open", s, t) for s, t in runs(dg > THR)]
    evs.sort(key=lambda x: x[1])
    start_held = bool(HLO <= g[0] <= HHI)
    state = "held" if start_held else "open"
    out = []
    for kind, s, t in evs:
        end = min(t + K, n - 1)
        if kind == "close":
            plat = float(np.median(g[end:min(end + 15, n)]))
            if state == "open":
                cls = "held" if HLO <= plat <= HHI else ("empty" if plat < HLO else "partial")
                out.append(dict(kind="grasp", start=int(s), t=int(end), plateau=round(plat, 2), cls=cls,
                                open_level=round(float(np.median(g[max(0, s - 30):max(s, 1)])), 2),
                                g_at_start=round(float(g[s]), 2)))
                if cls == "held":
                    state = "held"
            else:
                out.append(dict(kind="squeeze", start=int(s), t=int(end), plateau=round(plat, 2)))
        elif state == "held":
            out.append(dict(kind="release", start=int(s), t=int(s), end=int(end),
                            plateau_after=round(float(np.median(g[end:min(end + 15, n)])), 2)))
            state = "open"
        else:
            out.append(dict(kind="open_move", start=int(s), t=int(s)))
    return dict(n=n, J=J, g=g, v=v, start_held=start_held, events=out)


EP_LIST = [int(x) for x in epm["episode_index"]]
ROWS = {e: np.flatnonzero(EPI == e) for e in EP_LIST}
START = np.stack([S_ALL[ROWS[e][0], :5] for e in EP_LIST])
HOME = np.median(START, axis=0)

recs = []
for _, m in epm.iterrows():
    e = int(m["episode_index"])
    idx = ROWS[e]
    S = S_ALL[idx]
    fi = FI[idx]
    ts = TS[idx]
    R = events_for(S)
    n, J, g, v = R["n"], R["J"], R["g"], R["v"]
    qa = []
    if n != int(m["length"]):
        qa.append(f"meta length {int(m['length'])} != rows {n}")
    if not np.array_equal(fi, np.arange(n)):
        qa.append("frame_index not 0..n-1")
    if np.max(np.abs(ts - fi / FPS)) > 2e-3:
        qa.append("timestamp != frame/30")
    if np.isnan(S).any():
        qa.append("NaN in state")
    grasps, releases, empties, partials = [], [], [], []
    for ev in R["events"]:
        if ev["kind"] == "grasp":
            if ev["cls"] == "held":
                t0 = ev["t"]
                dd = np.max(np.abs(J[t0:, :4] - J[t0, :4]), axis=1)
                hit = np.flatnonzero(dd > LIFT_TOL)
                ev["lift_t"] = int(t0 + hit[0]) if hit.size else None
                ev["pose"] = np.round(J[t0], 2).tolist()
                ev["open_then_close_ok"] = bool(ev["g_at_start"] - ev["plateau"] >= THR)
                grasps.append(ev)
            elif ev["cls"] == "empty":
                empties.append(ev)
            else:
                partials.append(ev)
        elif ev["kind"] == "release":
            t = ev["t"]
            pan = float(J[t, 0])
            ev["pose"] = np.round(J[t], 2).tolist()
            ev["pan"] = round(pan, 2)
            ev["plate_cand"] = "BLUE" if pan >= PAN_BLUE else ("GREEN" if pan <= PAN_GREEN else "MAT")
            ev["plate_final"] = OVR.get(str(e), {}).get("plate", ev["plate_cand"])
            ev["approach_05s"] = np.round(J[max(t - 15, 0)], 2).tolist()
            ev["approach_10s"] = np.round(J[max(t - 30, 0)], 2).tolist()
            ev["dist_home_at_release"] = round(float(np.max(np.abs(J[t] - HOME))), 2)
            if t + 15 < n:
                hit = np.flatnonzero(np.max(np.abs(J[t + 15:] - HOME), axis=1) < HOME_TOL)
            else:
                hit = np.array([], dtype=int)
            ev["returned_t"] = int(t + 15 + hit[0]) if hit.size else None
            releases.append(ev)
    reasons = []
    if partials:
        reasons.append(f"{len(partials)} partial closure(s) (plateau above {HHI})")
    has_grasp = bool(grasps) or R["start_held"]
    if not has_grasp:
        typ = "bad"
        reasons.append("no held grasp: gripper closed on nothing (or never closed)")
    elif not releases:
        typ = "S1-hold"
        reasons.append("pick and lift only; no put-down or release inside the episode")
    else:
        last = releases[-1]
        plate = last["plate_final"]
        ret = last["returned_t"] is not None
        if plate == "MAT":
            if grasps:
                typ = "S1-place" if ret else "S1-place-noreturn"
                if not ret:
                    reasons.append("put down on mat spot; arm did not return home")
            else:
                typ = "place-only-MAT"
                reasons.append("starts holding a brick; put down on mat, not on a plate")
        else:
            typ = ("full-pick-place-" if grasps else "place-only-") + plate
            if not ret:
                reasons.append("no return home after release")
    if typ == "bad":
        verdict = "BAD"
    elif typ in ("S1-place-noreturn", "S1-hold", "place-only-MAT") or "no return home after release" in reasons:
        verdict = "WARN"
    else:
        verdict = "OK"
    vid = {}
    for k, name in ((TOPK, "top"), (SIDEK, "side")):
        vid[name] = dict(chunk=int(m[f"videos/{k}/chunk_index"]), file=int(m[f"videos/{k}/file_index"]),
                         from_s=round(float(m[f"videos/{k}/from_timestamp"]), 3),
                         to_s=round(float(m[f"videos/{k}/to_timestamp"]), 3))
    mv = np.flatnonzero(v > IDLE_V)
    start_plateau = round(float(np.median(g[:15])), 2) if R["start_held"] else None
    rec = dict(
        dataset_id=DSID, episode_index=e, task_index=int(TI[idx[0]]), task_text=str(m["tasks"][0]),
        n_frames=n, duration_s=round(n / FPS, 3), start_held=R["start_held"], start_plateau=start_plateau,
        type=typ, verdict=verdict, reasons=reasons + [f"QA: {q}" for q in qa], qa=qa,
        grasps=[{k: x for k, x in gr.items() if k != "kind"} for gr in grasps],
        releases=[{k: x for k, x in rl.items() if k != "kind"} for rl in releases],
        empties=[{k: x for k, x in em.items() if k != "kind"} for em in empties],
        start_pose=np.round(J[0], 2).tolist(), end_pose=np.round(J[-1], 2).tolist(),
        start_dist_home=round(float(np.max(np.abs(J[0] - HOME))), 2),
        end_dist_home=round(float(np.max(np.abs(J[-1] - HOME))), 2),
        idle_start_s=round(float(mv[0] / FPS), 3) if mv.size else None,
        idle_end_s=round(float((n - 1 - mv[-1]) / FPS), 3) if mv.size else None,
        frac_stationary=round(float(np.mean(v < MOVE_V)), 3),
        video=vid, start_gripper=round(float(g[0]), 2), end_gripper=round(float(g[-1]), 2),
    )
    recs.append(rec)


def st(a):
    a = np.asarray(a, float)
    if a.size == 0:
        return None
    return dict(n=int(a.size), median=round(float(np.median(a)), 2), p05=round(float(np.percentile(a, 5)), 2),
                p95=round(float(np.percentile(a, 95)), 2), min=round(float(a.min()), 2), max=round(float(a.max()), 2))


def pst(P, eps):
    P = np.asarray(P, float)
    if P.size == 0:
        return None
    return dict(n=int(P.shape[0]), episodes=eps, joints=JN[:5], median=np.round(np.median(P, 0), 2).tolist(),
                min=np.round(P.min(0), 2).tolist(), max=np.round(P.max(0), 2).tolist(),
                std=np.round(P.std(0), 2).tolist())


open_levels = [gr["open_level"] for r in recs for gr in r["grasps"]]
closed_brick = [gr["plateau"] for r in recs for gr in r["grasps"]] + [r["start_plateau"] for r in recs if r["start_held"]]
empty_pl = [em["plateau"] for r in recs for em in r["empties"]]
release_open = [rl["plateau_after"] for r in recs for rl in r["releases"]]
drop_ok = [gr["open_then_close_ok"] for r in recs for gr in r["grasps"]]


def rel_pose(plate, need_grasp):
    out = []
    for r in recs:
        if need_grasp and not r["grasps"]:
            continue
        for rl in r["releases"]:
            if rl["plate_final"] == plate:
                out.append((r["episode_index"], rl))
    return out


blue = rel_pose("BLUE", False)
green = rel_pose("GREEN", False)
mat = rel_pose("MAT", True)


def group_pose(lst, key):
    return pst([rl[key] for _, rl in lst], [e for e, _ in lst])


calib = dict(
    dataset_id=DSID,
    generated_by="V3 independent check, tools/v3_inventory.py",
    fps=FPS,
    joint_names_as_stored=info["features"]["observation.state"]["names"],
    gripper=dict(
        open_before_grasp=st(open_levels),
        closed_on_brick_plateau=st(closed_brick),
        closed_on_nothing_plateau=st(empty_pl),
        open_after_release=st(release_open),
        closing_direction="value falls while closing; open-to-plateau drop >= %.1f on %d of %d held grasps"
                          % (THR, sum(drop_ok), len(drop_ok)),
    ),
    closure_rule=dict(window_frames=K, drop_threshold=THR, plateau_window_frames=15,
                      held_plateau_range=[HLO, HHI], onset="first frame of a run where gripper.pos drops by more than drop_threshold within window_frames",
                      episodes_with_held_grasp=[r["episode_index"] for r in recs if r["grasps"] or r["start_held"]],
                      episodes_with_empty_grasp=[r["episode_index"] for r in recs if r["empties"]]),
    stationary=dict(arm_speed_units_per_s=MOVE_V, dwell_s=0.5, idle_speed_units_per_s=IDLE_V,
                    speed_filter="L2 norm of 5 arm joint steps x 30, 5-frame median"),
    home_start_pose=pst(START, EP_LIST),
    home_tolerance_units=HOME_TOL,
    end_pose_returned_s1=pst([r["end_pose"] for r in recs if r["type"] == "S1-place"],
                             [r["episode_index"] for r in recs if r["type"] == "S1-place"]),
    grasp_pose_held=pst([gr["pose"] for r in recs for gr in r["grasps"]],
                        [r["episode_index"] for r in recs for _ in r["grasps"]]),
    blue_drop=group_pose(blue, "pose"),
    green_drop=group_pose(green, "pose"),
    spot_put_down_mat=group_pose(mat, "pose"),
    approach_0p5s_before_release=dict(BLUE=group_pose(blue, "approach_05s"), GREEN=group_pose(green, "approach_05s"),
                                      MAT=group_pose(mat, "approach_05s")),
    approach_1p0s_before_release=dict(BLUE=group_pose(blue, "approach_10s"), GREEN=group_pose(green, "approach_10s"),
                                      MAT=group_pose(mat, "approach_10s")),
    release_dist_from_home=dict(BLUE=st([rl["dist_home_at_release"] for _, rl in blue]),
                                GREEN=st([rl["dist_home_at_release"] for _, rl in green]),
                                MAT=st([rl["dist_home_at_release"] for _, rl in mat])),
    placed_brick_pixels="NOT derived by V3 (needs image review); see INVENTORY.md",
)
s1_ret = [(r["episode_index"], rl["returned_t"] / FPS) for r in recs if r["type"] == "S1-place" for rl in r["releases"][-1:] if rl["returned_t"] is not None]
durs = {}
for typ in sorted(set(r["type"] for r in recs)):
    d = [r["duration_s"] for r in recs if r["type"] == typ]
    p95 = float(np.percentile(d, 95)) if len(d) > 1 else float(d[0])
    durs[typ] = dict(n=len(d), episodes=[r["episode_index"] for r in recs if r["type"] == typ],
                     p50=round(float(np.median(d)), 2), p95=round(p95, 2), max=round(float(max(d)), 2),
                     suggested_max_s=round(1.5 * p95, 1))
calib["duration_stats_by_type"] = durs
calib["s1_start_to_return_s"] = st([t for _, t in s1_ret])
calib["s1_start_to_return_episodes"] = [e for e, _ in s1_ret]


def who(types, verdicts):
    return [r["episode_index"] for r in recs if r["type"] in types and r["verdict"] in verdicts]


BLUE_T = ["full-pick-place-BLUE", "place-only-BLUE"]
GREEN_T = ["full-pick-place-GREEN", "place-only-GREEN"]
needs = {
    "S1_OD9_pick_put_spot_return": dict(target=40, minimum=30, ok=who(["S1-place"], ["OK"]),
                                        warn=who(["S1-place-noreturn"], ["WARN"])),
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
gaps = dict(dataset_id=DSID, studio_name_stale="Smoke test - inspection skill - 5 episodes",
            episodes_on_disk=len(recs), frames_on_disk=int(sum(r["n_frames"] for r in recs)),
            type_counts={t: sum(1 for r in recs if r["type"] == t) for t in sorted(set(r["type"] for r in recs))},
            needs=needs)

(OUT / "episodes_v3.json").write_text(json.dumps(recs, indent=1, default=tolist))
(OUT / "calib.json").write_text(json.dumps(calib, indent=1, default=tolist))
(OUT / "gaps.json").write_text(json.dumps(gaps, indent=1, default=tolist))
rows = []
for r in recs:
    gf = r["grasps"][0] if r["grasps"] else {}
    lr = r["releases"][-1] if r["releases"] else {}
    rows.append(dict(
        dataset_id=r["dataset_id"], episode_index=r["episode_index"], task_index=r["task_index"],
        task_text=r["task_text"], n_frames=r["n_frames"], duration_s=r["duration_s"],
        start_held=r["start_held"], start_plateau=r["start_plateau"], type=r["type"], verdict=r["verdict"],
        reasons=" | ".join(r["reasons"]),
        grasp_start_s=round(gf["start"] / FPS, 3) if gf else None,
        grasp_close_end_s=round(gf["t"] / FPS, 3) if gf else None,
        grasp_plateau=gf.get("plateau"), open_level=gf.get("open_level"),
        lift_s=round(gf["lift_t"] / FPS, 3) if gf and gf.get("lift_t") is not None else None,
        n_held_grasps=len(r["grasps"]), n_empty_grasps=len(r["empties"]), n_releases=len(r["releases"]),
        release_onset_s=round(lr["t"] / FPS, 3) if lr else None, release_pan=lr.get("pan") if lr else None,
        release_plate_cand=lr.get("plate_cand") if lr else None,
        release_plate_final=lr.get("plate_final") if lr else None,
        release_pose=" ".join(str(x) for x in lr["pose"]) if lr else "",
        returned_home_s=round(lr["returned_t"] / FPS, 3) if lr and lr.get("returned_t") is not None else None,
        start_dist_home=r["start_dist_home"], end_dist_home=r["end_dist_home"],
        idle_start_s=r["idle_start_s"], idle_end_s=r["idle_end_s"], frac_stationary=r["frac_stationary"],
        top_file=r["video"]["top"]["file"], top_from_s=r["video"]["top"]["from_s"], top_to_s=r["video"]["top"]["to_s"],
        side_file=r["video"]["side"]["file"], side_from_s=r["video"]["side"]["from_s"],
        side_to_s=r["video"]["side"]["to_s"], qa="; ".join(r["qa"]),
    ))
pd.DataFrame(rows).to_csv(OUT / "episodes.csv", index=False)

print("HOME(start median)", np.round(HOME, 2).tolist())
for r in recs:
    gtxt = [(round(g_["start"] / FPS, 2), round(g_["t"] / FPS, 2), g_["plateau"], g_["cls"], g_["open_level"],
             (round(g_["lift_t"] / FPS, 2) if g_.get("lift_t") is not None else None)) for g_ in r["grasps"] + r["empties"]]
    rtxt = [(round(x["t"] / FPS, 2), x["pan"], x["plate_cand"], x["plate_final"],
             (round(x["returned_t"] / FPS, 2) if x["returned_t"] is not None else None), x["pose"]) for x in r["releases"]]
    print(f"ep{r['episode_index']:>2} n={r['n_frames']:>4} dur={r['duration_s']:>6.2f} ti={r['task_index']} "
          f"held0={r['start_held']} p0={r['start_plateau']} type={r['type']} {r['verdict']} "
          f"d0={r['start_dist_home']} dEnd={r['end_dist_home']} idle0={r['idle_start_s']} idleEnd={r['idle_end_s']}")
    print(f"     grasps(start,close_end,plat,cls,open,lift)={gtxt}")
    print(f"     releases(onset,pan,cand,final,returned,pose)={rtxt}")
    print(f"     reasons={r['reasons']}")
print("NEEDS", json.dumps({k: {kk: vv for kk, vv in v.items()} for k, v in needs.items()}, default=tolist))
print("OPEN", calib["gripper"]["open_before_grasp"], "CLOSED", calib["gripper"]["closed_on_brick_plateau"])
print("EMPTY", calib["gripper"]["closed_on_nothing_plateau"], "RELOPEN", calib["gripper"]["open_after_release"])
print("DROP", calib["gripper"]["closing_direction"])
print("BLUE", json.dumps(calib["blue_drop"], default=tolist))
print("GREEN", json.dumps(calib["green_drop"], default=tolist))
print("MAT", json.dumps(calib["spot_put_down_mat"], default=tolist))
print("DURS", json.dumps(durs, default=tolist))
print("V3_INVENTORY_OK", len(recs), "episodes")
