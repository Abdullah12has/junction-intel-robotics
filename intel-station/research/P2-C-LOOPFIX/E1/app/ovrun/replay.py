"""Offline replay of model 389da0fd (our Studio ACT export, copied) through the Studio runtime path.

Runs in envs/anomalib-env with the physicalai source at R/repos/physicalai (8381d43) on sys.path.
Call chain replicated (read-only reference in the Studio backend):
  runtime_session/controls/model.py  ModelControl.on_tick -> PolicySource.update(obs(state), {}, step)
  runtime_session/model.py           ProcessModel.predict_action_chunk (state as list, task string)
  workers/model_worker.py            Inference.infer (state float32 (1,6); frame.value[np.newaxis]; task list)
  physicalai.inference.InferenceModel.predict_action_chunk (manifest preprocessors, adapter, runner)
  physicalai.runtime PolicySource + SyncExecution(request_threshold=0.5) + ChunkedActionQueue(LerpSmoother())
Frames: rgb24 decoded from the dataset videos (ffmpeg, libdav1d). The store's latest frame at tick t is frame t.
Usage: python3 -B -s replay.py --device CPU|GPU|NPU
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

W = "/home/intel/intel-junction-2026/work/P2-B-OVRUN/E1"
EXPORT = f"{W}/models/389da0fd/openvino"
DS = "/home/intel/.local/share/physicalai/datasets/a4ba6ede-092c-450f-8c76-559f66991dbd"
MODEL_ID = "389da0fd-e840-4224-b0a0-09aa5be7a8fc"
FPS = 30
H, WD = 480, 640
PROFILE = [0, 1, 5, 10, 25, 50, 75, 99]


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def decode_camera(cam):
    path = f"{DS}/videos/observation.images.{cam}/chunk-000/file-000.mp4"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, WD, 3)


def lat(x_s):
    ms = np.asarray(x_s, dtype=np.float64) * 1000.0
    if ms.size == 0:
        return {"n": 0}
    return {"n": int(ms.size), "mean_ms": float(ms.mean()), "p50_ms": float(np.percentile(ms, 50)),
            "p95_ms": float(np.percentile(ms, 95)), "max_ms": float(ms.max())}


@dataclass
class Obs:
    """What ModelControl hands to PolicySource.update: the follower's measured state only."""

    joint_positions: np.ndarray
    timestamp: float
    images: object = None

    @property
    def state(self):
        return self.joint_positions


class StudioWorkerPath:
    """ProcessModel -> worker Inference.infer -> InferenceModel, in the order Studio calls them."""

    def __init__(self, model, frames, image_keys):
        self.model = model
        self.frames = frames
        self.image_keys = image_keys
        self.tick = 0
        self.call_ticks, self.call_infer_s, self.call_pre_s, self.chunks = [], [], [], []
        self.first_obs = None

    def predict_action_chunk(self, observation):
        state = np.asarray(observation["state"])[0].tolist()
        task = observation.get("task")
        obs = {"state": np.array([state], dtype=np.float32)}
        for name in self.image_keys:
            obs[name] = self.frames[name][self.tick][np.newaxis]
        if task:
            obs["task"] = [task[0]]
        pre_s = self._preprocess_only(obs)
        t0 = time.perf_counter()
        actions = self.model.predict_action_chunk(obs)
        infer_s = time.perf_counter() - t0
        out = np.asarray(np.asarray(actions).tolist(), dtype=np.float32)
        self.call_ticks.append(self.tick)
        self.call_infer_s.append(infer_s)
        self.call_pre_s.append(pre_s)
        self.chunks.append(out)
        if self.first_obs is None:
            self.first_obs = obs
        return out

    def _preprocess_only(self, obs):
        t0 = time.perf_counter()
        x = self.model._order_cameras(obs)
        for proc in self.model.preprocessors:
            x = proc(x)
        return time.perf_counter() - t0

    def reset(self):
        self.model.reset()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True)
    args = ap.parse_args()
    out_path = f"{W}/results/replay_{args.device}.json"
    res = {"device": args.device, "model_id": MODEL_ID, "export_copy": EXPORT, "started_utc": now()}
    prep = json.load(open(f"{W}/data/prep.json"))
    arr = np.load(f"{W}/data/a4ba6ede_arrays.npz")
    state_all, action_all = arr["state"], arr["action"]
    ep_all, task_all = arr["episode_index"], arr["task_index"]
    manifest = json.load(open(f"{EXPORT}/manifest.json"))
    image_keys = [f["init_args"]["name"] for f in manifest["model"]["input_features"]
                  if f["init_args"]["ftype"] == "VISUAL"]
    cams = [k.removeprefix("images.") for k in image_keys]
    res["image_keys_manifest_order"] = image_keys
    try:
        t0 = time.perf_counter()
        model = InferenceModel(EXPORT, backend="openvino", device=args.device)
        res["load_s"] = time.perf_counter() - t0
        res["model_repr"] = repr(model)
        res["preprocessors"] = [type(p).__name__ for p in model.preprocessors]
        res["postprocessors"] = [type(p).__name__ for p in model.postprocessors]
        res["adapter_input_names"] = list(model.adapter.input_names)
        res["compiled_inputs"] = [[str(i.get_any_name()), str(i.get_partial_shape()), str(i.get_element_type())]
                                  for i in model.adapter.compiled_model.inputs]
        res["compiled_outputs"] = [[str(o.get_any_name()), str(o.get_partial_shape())]
                                   for o in model.adapter.compiled_model.outputs]
        t0 = time.perf_counter()
        frames = {cam: decode_camera(cam) for cam in cams}
        res["decode_s"] = time.perf_counter() - t0
        res["frame_counts"] = {cam: int(frames[cam].shape[0]) for cam in cams}
        frames_by_key = {k: frames[k.removeprefix("images.")] for k in image_keys}

        all_upd, all_upd_q, all_upd_nq, all_infer, all_pre = [], [], [], [], []
        eps_out, exec_err, hold_err, first_chunk, rolling = [], [], [], [], []
        sample = None
        for ep in prep["episodes"]:
            e, L = ep["episode_index"], ep["length"]
            rows = np.where(ep_all == e)[0]
            assert len(rows) == L, (e, len(rows), L)
            g0s = {int(round(float(ep[f"videos/observation.images.{c}/from_timestamp"]) * FPS)) for c in cams}
            assert len(g0s) == 1
            g0 = g0s.pop()
            assert g0 + L <= min(res["frame_counts"].values())
            task_text = prep["task_text"][str(int(task_all[rows[0]]))]
            S, A = state_all[rows], action_all[rows]
            path = StudioWorkerPath(model, frames_by_key, image_keys)
            src = PolicySource(path, SyncExecution(request_threshold=0.5),
                               ChunkedActionQueue(smoother=LerpSmoother()), task=task_text)
            src.connect(bus=None, session_id="")
            executed = np.zeros((L, 6), np.float32)
            upd = np.zeros(L)
            for t in range(L):
                path.tick = g0 + t
                t0 = time.perf_counter()
                executed[t] = src.update(Obs(S[t], t / FPS), {}, t)
                upd[t] = time.perf_counter() - t0
            src.disconnect()
            q_rel = [tk - g0 for tk in path.call_ticks]
            qs = []
            for tq, chunk in zip(q_rel, path.chunks):
                n = min(chunk.shape[0], L - tq)
                err = np.abs(chunk[:n] - A[tq:tq + n])
                qs.append({"t": int(tq), "n": int(n), "mae_per_joint": err.mean(0).tolist(),
                           "horizon_mae": {str(k): float(err[k].mean()) for k in PROFILE if k < n}})
            if sample is None:
                sample = {"episode": e, "pred_first_action": path.chunks[0][0].tolist(),
                          "recorded_first_action": A[0].tolist(), "pred_action_k10": path.chunks[0][10].tolist(),
                          "recorded_action_k10": A[10].tolist(), "state_first": S[0].tolist()}
            exec_e = np.abs(executed - A)
            hold_e = np.abs(S - A)
            exec_err.append(exec_e)
            hold_err.append(hold_e)
            rolling.extend(q["mae_per_joint"] for q in qs)
            first_chunk.append(qs[0]["mae_per_joint"])
            upd_q = [upd[t] for t in q_rel]
            upd_nq = [upd[t] for t in range(L) if t not in set(q_rel)]
            all_upd.extend(upd.tolist())
            all_upd_q.extend(upd_q)
            all_upd_nq.extend(upd_nq)
            all_infer.extend(path.call_infer_s)
            all_pre.extend(path.call_pre_s)
            eps_out.append({"episode": int(e), "length": int(L), "task": task_text, "frame_offset": int(g0),
                            "queries_at_ticks": [int(x) for x in q_rel], "n_queries": len(q_rel),
                            "executed_mae_per_joint": exec_e.mean(0).tolist(),
                            "hold_state_mae_per_joint": hold_e.mean(0).tolist(),
                            "first_chunk_mae_per_joint": qs[0]["mae_per_joint"],
                            "rolling_chunk_mae_per_joint": np.mean([q["mae_per_joint"] for q in qs], axis=0).tolist(),
                            "chunks": qs})
            print(now(), "episode", e, "L", L, "queries", len(q_rel), flush=True)
        res["episodes"] = eps_out
        res["sample_first_chunk"] = sample
        ex = np.concatenate(exec_err)
        hd = np.concatenate(hold_err)
        res["aggregate"] = {
            "executed_mae_per_joint": ex.mean(0).tolist(), "executed_mae_mean": float(ex.mean()),
            "hold_state_mae_per_joint": hd.mean(0).tolist(), "hold_state_mae_mean": float(hd.mean()),
            "first_chunk_mae_per_joint_mean_over_episodes": np.mean(first_chunk, axis=0).tolist(),
            "rolling_chunk_mae_per_joint_mean_over_queries": np.mean(rolling, axis=0).tolist(),
            "n_queries_total": len(rolling), "n_steps_total": int(ex.shape[0]),
        }
        res["latency"] = {"update_all": lat(all_upd), "update_at_query_ticks": lat(all_upd_q),
                          "update_non_query_ticks": lat(all_upd_nq), "chunk_infer_incl_preprocess": lat(all_infer),
                          "preprocess_only": lat(all_pre), "first_chunk_infer": lat(all_infer[:1]),
                          "chunk_infer_after_first": lat(all_infer[1:])}
        reps = [np.asarray(model.predict_action_chunk(path.first_obs), dtype=np.float32) for _ in range(5)]
        res["determinism_5_repeats_max_abs_diff"] = float(max(np.abs(r - reps[0]).max() for r in reps[1:]))
        x = model._order_cameras(dict(path.first_obs))
        for proc in model.preprocessors:
            x = proc(x)
        res["parity_observation_into_model"] = {
            "keys_and_shapes": {k: [list(np.shape(v)), str(np.asarray(v).dtype)] for k, v in path.first_obs.items()
                                if k != "task"},
            "task_sent": path.first_obs.get("task"),
            "after_preprocessors": {k: [list(np.shape(v)), str(np.asarray(v).dtype), float(np.min(v)), float(np.max(v))]
                                    for k, v in x.items() if isinstance(v, np.ndarray)},
            "predict_output_shape": list(reps[0].shape),
            "manifest_has_stats_normalizer": any("stats" in str(p).lower() for p in manifest["model"]["preprocessors"]),
        }
        res["ok"] = True
    except Exception as exc:  # record the failure for this device and stop
        res["ok"] = False
        res["error"] = f"{type(exc).__name__}: {str(exc)[:1500]}"
    res["finished_utc"] = now()
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(now(), "WROTE", out_path, "ok", res.get("ok"), res.get("error", ""), flush=True)


if __name__ == "__main__":
    main()
