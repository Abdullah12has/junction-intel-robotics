"""Replay of ACT model f06a253e (PP-BLUE v1) through the Studio runtime call chain (same chain as app/ovrun/replay.py).

Runs in envs/ovrun-env on OpenVINO GPU. Reads the model export and prepared episode arrays only; writes under W.
Usage: python3 -B -s replay2.py --device GPU
"""
import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np

REPO_SRC = "/home/intel/intel-junction-2026/repos/physicalai/src"
sys.path.insert(0, REPO_SRC)
from physicalai.inference import InferenceModel  # noqa: E402
from physicalai.runtime import ChunkedActionQueue, LerpSmoother, PolicySource, SyncExecution  # noqa: E402

W = "/home/intel/intel-junction-2026/work/P2-B-OVRUN2/E1"
MODEL_ID = "f06a253e-0f64-4d29-bf93-03966d03545b"
EXPORT = f"/home/intel/.local/share/physicalai/models/{MODEL_ID}/exports/openvino"
TRAINED_TEXT = "move the red brick to the blue plate"
CAMS = ["realsense 243722064316", "realsense 243622061421"]
FPS = 30
H, WD = 480, 640
GRIP_T = 6.0
PAN_MIN = 3.0
SET_ORDER = ["trainBlueV1", "heldBlueV2", "heldGreenV2"]


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def decode(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, WD, 3)


def lat(x_s):
    ms = np.asarray(x_s, dtype=np.float64) * 1000.0
    if ms.size == 0:
        return {"n": 0}
    return {"n": int(ms.size), "p50_ms": round(float(np.percentile(ms, 50)), 2),
            "p95_ms": round(float(np.percentile(ms, 95)), 2), "max_ms": round(float(ms.max()), 2)}


def events(g):
    """Gripper events on one trajectory: open (>= GRIP_T), close (first drop below), release (first rise back)."""
    op = np.where(g >= GRIP_T)[0]
    if op.size == 0:
        return None, None, None
    o = int(op[0])
    cl = np.where(g[o:] < GRIP_T)[0]
    if cl.size == 0:
        return o, None, None
    c = o + int(cl[0])
    rl = np.where(g[c:] >= GRIP_T)[0]
    r = (c + int(rl[0])) if rl.size else None
    return o, c, r


def match_best(Sq, Sr, win=300, max_off=200):
    """Smallest mean abs state difference over a window, aligning the two takes at any offset up to max_off."""
    best = (1e9, 0, 0)
    for k in range(0, min(max_off, len(Sq) - win) + 1):
        d = float(np.abs(Sq[k:k + win] - Sr[:win]).mean())
        if d < best[0]:
            best = (d, k, 0)
    for k in range(1, min(max_off, len(Sr) - win) + 1):
        d = float(np.abs(Sq[:win] - Sr[k:k + win]).mean())
        if d < best[0]:
            best = (d, 0, k)
    return best


@dataclass
class Obs:
    joint_positions: np.ndarray
    timestamp: float
    images: object = None

    @property
    def state(self):
        return self.joint_positions


class StudioWorkerPath:
    """ProcessModel -> worker Inference.infer -> InferenceModel, in the order Studio calls them (as replay.py)."""

    def __init__(self, model, frames, image_keys):
        self.model = model
        self.frames = frames
        self.image_keys = image_keys
        self.tick = 0
        self.call_ticks, self.call_infer_s, self.chunks = [], [], []

    def reset(self):
        self.model.reset()

    def predict_action_chunk(self, observation):
        state = np.asarray(observation["state"])[0].tolist()
        task = observation.get("task")
        obs = {"state": np.array([state], dtype=np.float32)}
        for name in self.image_keys:
            obs[name] = self.frames[name][self.tick][np.newaxis]
        if task:
            obs["task"] = [task[0]]
        t0 = time.perf_counter()
        actions = self.model.predict_action_chunk(obs)
        infer_s = time.perf_counter() - t0
        out = np.asarray(np.asarray(actions).tolist(), dtype=np.float32)
        self.call_ticks.append(self.tick)
        self.call_infer_s.append(infer_s)
        self.chunks.append(out)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True)
    args = ap.parse_args()
    started = now()
    man = json.load(open(f"{W}/data/prep2.json"))
    eps = man["episodes"]
    arrs, rec_ev, rel_pan = {}, {}, {"trainBlueV1": [], "heldGreenV2": []}
    for ep in eps:
        key = (ep["tag"], ep["episode_index"])
        d = np.load(ep["npz"])
        arrs[key] = (d["state"], d["action"])
        rec_ev[key] = events(d["action"][:, 5])
        r = rec_ev[key][2]
        if r is not None and ep["tag"] in rel_pan:
            rel_pan[ep["tag"]].append(float(d["action"][r, 0]))
    blue_ref = float(np.median(rel_pan["trainBlueV1"]))
    green_ref = float(np.median(rel_pan["heldGreenV2"]))

    # Is a v2 blue take the same recording as a training take (then it is in-sample, not held-out)?
    train_keys = [k for k in arrs if k[0] == "trainBlueV1"]
    dup, checks = {}, []
    for k in [k for k in arrs if k[0] == "heldBlueV2"]:
        best = None
        for tk in train_keys:
            b = match_best(arrs[k][0], arrs[tk][0])
            checks.append({"query": f"{k[0]} ep{k[1]}", "train": f"{tk[0]} ep{tk[1]}",
                           "mean_abs_state_diff": round(b[0], 4), "offset_query": b[1], "offset_train": b[2]})
            best = b[0] if best is None else min(best, b[0])
        dup[k[1]] = bool(best < 0.05)

    manifest = json.load(open(f"{EXPORT}/manifest.json"))
    image_keys = [f["init_args"]["name"] for f in manifest["model"]["input_features"]
                  if f["init_args"]["ftype"] == "VISUAL"]
    cams = [k.removeprefix("images.") for k in image_keys]
    t0 = time.perf_counter()
    model = InferenceModel(EXPORT, backend="openvino", device=args.device)
    load_s = round(time.perf_counter() - t0, 3)
    print(now(), "loaded", repr(model), "load_s", load_s, flush=True)

    per_ep, abs_ex, abs_hd, upd_q_all, infer_all = [], {}, {}, [], []
    cache_key, frames_by_key, frames_n = None, None, None
    for ep in eps:
        tag, e, L, g0 = ep["tag"], ep["episode_index"], ep["length"], ep["g0"]
        if tag == "trainBlueV1":
            role = "train"
        elif tag == "heldGreenV2":
            role = "heldout_green_negative"
        else:
            role = "v2_trim_of_train_take" if dup[e] else "heldout_blue"
        key = tuple(ep["videos"])
        if key != cache_key:
            t1 = time.perf_counter()
            frames = {cam: decode(vp) for cam, vp in zip(CAMS, ep["videos"])}
            frames_by_key = {k: frames[k.removeprefix("images.")] for k in image_keys}
            frames_n = {cam: int(frames[cam].shape[0]) for cam in cams}
            cache_key = key
            print(now(), "decoded", ep["videos"][0].split("/")[-1], frames_n, round(time.perf_counter() - t1, 2), "s", flush=True)
        assert g0 + L <= min(frames_n.values()), (tag, e, g0, L, frames_n)
        S, A = arrs[(tag, e)]
        path = StudioWorkerPath(model, frames_by_key, image_keys)
        src = PolicySource(path, SyncExecution(request_threshold=0.5), ChunkedActionQueue(smoother=LerpSmoother()),
                           task=TRAINED_TEXT)
        src.connect(bus=None, session_id="")
        E = np.zeros((L, 6), np.float32)
        upd = np.zeros(L)
        for t in range(L):
            path.tick = g0 + t
            t1 = time.perf_counter()
            E[t] = src.update(Obs(S[t], t / FPS), {}, t)
            upd[t] = time.perf_counter() - t1
        src.disconnect()
        q_rel = [tk - g0 for tk in path.call_ticks]
        qs = []
        for tq, chunk in zip(q_rel, path.chunks):
            n = min(chunk.shape[0], L - tq)
            err = np.abs(chunk[:n] - A[tq:tq + n])
            qs.append({"t": int(tq), "n": int(n), "mae_per_joint": [round(float(x), 3) for x in err.mean(0)]})
        ex, hd = np.abs(E - A), np.abs(S - A)
        o_r, c_r, r_r = rec_ev[(tag, e)]
        o_p, c_p, r_p = events(E[:, 5])
        pan = {"side_ok": None}
        if c_r is not None and r_r is not None:
            pg, pr = float(A[c_r, 0]), float(A[r_r, 0])
            eg, er = float(E[c_r, 0]), float(E[r_r, 0])
            tgt = float(np.sign(blue_ref - pg)) if abs(blue_ref - pg) >= PAN_MIN else 0.0
            ps = float(np.sign(er - eg))
            pan = {"rec_grasp_pan": round(pg, 2), "rec_release_pan": round(pr, 2), "rec_disp": round(pr - pg, 2),
                   "pred_grasp_pan": round(eg, 2), "pred_release_pan": round(er, 2), "pred_disp": round(er - eg, 2),
                   "target_side": int(tgt), "pred_side": int(ps),
                   "pred_release_nearer_blue_ref": bool(abs(er - blue_ref) < abs(er - green_ref)),
                   "side_ok": (None if tgt == 0.0 else bool(ps == tgt))}
        dt_c = None if (c_p is None or c_r is None) else int(c_p - c_r)
        dt_r = None if (r_p is None or r_r is None) else int(r_p - r_r)
        row = {"set": tag, "episode_index": int(e), "role": role, "length": int(L), "g0": int(g0),
               "n_queries": len(q_rel),
               "exec_mae_per_joint": [round(float(x), 3) for x in ex.mean(0)],
               "exec_mae_mean": round(float(ex.mean()), 3),
               "hold_mae_per_joint": [round(float(x), 3) for x in hd.mean(0)],
               "hold_mae_mean": round(float(hd.mean()), 3),
               "beats_hold": bool(ex.mean() < hd.mean()),
               "first_chunk_mae_mean": round(float(np.mean(qs[0]["mae_per_joint"])), 3),
               "rec_gripper_events_frames": [o_r, c_r, r_r], "pred_gripper_events_frames": [o_p, c_p, r_p],
               "dt_close_frames": dt_c, "dt_release_frames": dt_r,
               "dt_close_s": None if dt_c is None else round(dt_c / FPS, 2),
               "dt_release_s": None if dt_r is None else round(dt_r / FPS, 2),
               "gripper_action_min_max_rec": [round(float(A[:, 5].min()), 2), round(float(A[:, 5].max()), 2)],
               "gripper_action_min_max_pred": [round(float(E[:, 5].min()), 2), round(float(E[:, 5].max()), 2)],
               "pan": pan,
               "latency_update_at_query_ticks": lat(upd[q_rel]),
               "latency_chunk_infer": lat(path.call_infer_s),
               "queries": qs}
        per_ep.append(row)
        abs_ex.setdefault(role, []).append(ex)
        abs_hd.setdefault(role, []).append(hd)
        upd_q_all.extend(upd[q_rel].tolist())
        infer_all.extend(path.call_infer_s)
        print(now(), tag, "ep", e, role, "L", L, "queries", len(q_rel), "exec", row["exec_mae_mean"],
              "hold", row["hold_mae_mean"], "pan", pan.get("side_ok"), flush=True)

    agg = {}
    for role, lst in abs_ex.items():
        ex = np.concatenate(lst)
        hd = np.concatenate(abs_hd[role])
        agg[role] = {"steps": int(ex.shape[0]), "exec_mae_per_joint": [round(float(x), 3) for x in ex.mean(0)],
                     "exec_mae_mean": round(float(ex.mean()), 3),
                     "hold_mae_per_joint": [round(float(x), 3) for x in hd.mean(0)],
                     "hold_mae_mean": round(float(hd.mean()), 3)}
    train_rows = [r for r in per_ep if r["role"] == "train"]
    ho_blue = [r for r in per_ep if r["role"] == "heldout_blue"]
    reasons = []
    if not all(r["beats_hold"] for r in train_rows):
        reasons.append("exec MAE not below hold-state on every training episode")
    if not all(r["pan"]["side_ok"] is True for r in train_rows):
        reasons.append("pan side not correct on every training episode")
    if not ho_blue:
        reasons.append("no held-out blue episode")
    elif not all(r["pan"]["side_ok"] is True for r in ho_blue):
        reasons.append("pan side wrong on held-out blue episode")
    verdict = "SANE" if not reasons else "NOT SANE"
    compact = []
    for r in per_ep:
        compact.append({k: r[k] for k in ["set", "episode_index", "role", "length", "n_queries", "exec_mae_mean",
                                          "hold_mae_mean", "beats_hold", "exec_mae_per_joint", "hold_mae_per_joint",
                                          "first_chunk_mae_mean", "rec_gripper_events_frames",
                                          "pred_gripper_events_frames", "dt_close_s", "dt_release_s", "pan",
                                          "latency_update_at_query_ticks"]})
    summary = {"model_id": MODEL_ID, "model_name": "PP-BLUE ACT red brick to blue plate v1", "device": args.device,
               "trained_task_text": TRAINED_TEXT, "load_s": load_s, "started_utc": started, "finished_utc": now(),
               "plate_pan_refs": {"blue_median_recorded_release_pan": round(blue_ref, 3),
                                  "green_median_recorded_release_pan": round(green_ref, 3),
                                  "blue_release_pans": [round(x, 3) for x in rel_pan["trainBlueV1"]],
                                  "green_release_pans": [round(x, 3) for x in rel_pan["heldGreenV2"]]},
               "duplicate_check_v2_vs_train": checks, "duplicate_flags_heldBlueV2": dup,
               "aggregate_by_role": agg, "verdict": verdict, "reasons": reasons,
               "latency_update_at_query_ticks_all": lat(upd_q_all), "latency_chunk_infer_all": lat(infer_all),
               "episodes": compact}
    full = dict(summary)
    full["episodes_full"] = per_ep
    json.dump(full, open(f"{W}/results/f06a253e_GPU_full.json", "w"), indent=1)
    json.dump(summary, open(f"{W}/results/f06a253e_GPU_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    print(now(), "WROTE summary", "VERDICT", verdict, "; ".join(reasons), flush=True)


if __name__ == "__main__":
    main()
