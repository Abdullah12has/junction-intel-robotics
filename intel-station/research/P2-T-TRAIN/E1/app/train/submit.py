#!/usr/bin/env python3
"""P2-T-TRAIN E1: payload builder, validator, submitter and poller for Hassan's Modal ACT run.

build, jobs, submit, poll and finish are stdlib only and run on the laptop (python3 -I submit.py ...).
validate needs pydantic and the sandbox repo copy; run it with the sandbox venv only.

  build    --dataset-id UUID --name NAME --episodes N --frames TOTAL [--gpu A10G] --out payload.json
  validate PAYLOAD.json
  jobs
  submit   PAYLOAD.json RESPONSE.json
  poll     JOB_ID --log FILE [--max-minutes N]
  finish   JOB_ID OUT.json
"""

import argparse
import datetime
import json
import math
import sys
import time
import types
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:7860"
PROJECT_ID = "ab767473-d6fd-47ff-812e-8239c455dabb"
TRAINER_ID = "ca5568b5-4542-4fe2-9932-c09a4c1bdbaa"
TRAINER_NAME = "Modal"
TEAM_TRAINER_IDS = {
    "57e65864-ac75-4af2-b4d9-0af5246c6023",
    "b963d065-34f1-4740-8448-4bf9fdca2abd",
}
TERMINAL = {"completed", "failed", "canceled"}
BATCH = 8
SRC = "/home/claude/kb/repos/physical-ai-studio/application/backend/src"


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def http(method, path, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode()


def get_json(path):
    status, text = http("GET", path)
    if status != 200:
        raise SystemExit(f"GET {path} -> {status}: {text[:500]}")
    return json.loads(text)


def steps_per_epoch(n_episodes, total_frames, batch_size):
    val_episodes = max(1, math.floor(n_episodes * 0.1))
    expected_train = total_frames * (1 - val_episodes / n_episodes)
    return math.floor(expected_train / batch_size)


def target_steps(n_episodes):
    if n_episodes < 10:
        return 2000
    if 30 <= n_episodes <= 50:
        return 25000
    raise SystemExit(f"{n_episodes} episodes: no step rule in the task spec, ask Opus for a target")


def build(args):
    per_epoch = steps_per_epoch(args.episodes, args.frames, BATCH)
    target = target_steps(args.episodes)
    epochs = max(1, round(target / per_epoch))
    payload = {
        "training_target": "remote",
        "remote_trainer_id": TRAINER_ID,
        "modal_gpu": args.gpu,
        "project_id": PROJECT_ID,
        "dataset_id": args.dataset_id,
        "policy": "act",
        "model_name": args.name,
        "max_epochs": epochs,
        "batch_size": BATCH,
        "num_workers": "auto",
        "auto_scale_batch_size": False,
        "precision": "bf16-mixed",
        "compile_model": False,
        "image_key_reorder_map": {},
        "num_cameras": 0,
        "export_backends": ["openvino"],
        "augment_images": False,
        "lora_enabled": False,
        "lora_rank": 32,
        "lora_alpha": None,
        "lora_dropout": 0.05,
        "lora_use_dora": False,
        "snapflow_enabled": False,
        "snapflow_distill_epochs": 3,
        "val_split": 0.1,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(
        f"episodes={args.episodes} frames={args.frames} steps_per_epoch={per_epoch} "
        f"target={target} max_epochs={epochs} est_steps={per_epoch * epochs}"
    )


def validate(args):
    for name in ("schemas", "training"):
        pkg = types.ModuleType(name)
        pkg.__path__ = [f"{SRC}/{name}"]
        sys.modules[name] = pkg
    sys.path.insert(0, SRC)
    from pydantic import ValidationError
    from schemas.job import RemoteTrainJobPayload, TrainJobPayloadAdapter

    with open(args.payload) as f:
        data = json.load(f)
    try:
        obj = TrainJobPayloadAdapter.validate_python(data)
    except ValidationError as err:
        raise SystemExit(f"INVALID: {err}")
    assert isinstance(obj, RemoteTrainJobPayload)
    out = obj.model_dump(mode="json")
    print(
        "VALID",
        f"max_epochs={out['max_epochs']}",
        f"max_steps={out['max_steps']}",
        f"export_backends={out['export_backends']}",
        f"modal_gpu={out['modal_gpu']}",
        f"precision={out['precision']}",
        f"device={out['device']}",
        f"remote_trainer_id={out['remote_trainer_id']}",
    )


def jobs(args):
    for job in get_json("/api/jobs"):
        p = job["payload"]
        print(
            job["id"],
            job["status"],
            job["progress"],
            p.get("training_target"),
            p.get("remote_trainer_name"),
            p.get("model_name"),
            sep=" | ",
        )


def submit(args):
    with open(args.payload) as f:
        body = json.load(f)
    if body.get("training_target") != "remote" or body.get("remote_trainer_id") != TRAINER_ID:
        raise SystemExit("refusing: payload is not for Hassan's Modal trainer")
    if body["remote_trainer_id"] in TEAM_TRAINER_IDS:
        raise SystemExit("refusing: team trainer")
    existing = [j["id"] for j in get_json("/api/jobs") if j["payload"].get("model_name") == body["model_name"]]
    if existing:
        raise SystemExit(f"refusing: a job named {body['model_name']!r} already exists: {existing}")
    try:
        status, text = http("POST", "/api/jobs:train", body, timeout=120)
    except Exception as err:
        raise SystemExit(f"POST outcome unknown ({type(err).__name__}); check jobs before any retry")
    with open(args.response, "w") as f:
        f.write(text)
    if status != 200:
        raise SystemExit(f"POST /api/jobs:train -> {status}: {text[:800]}")
    job = json.loads(text)
    trainer_name = job["payload"].get("remote_trainer_name")
    print(
        utc(),
        "submitted",
        job["id"],
        job["status"],
        trainer_name,
        job["payload"].get("modal_gpu"),
        job["payload"].get("max_epochs"),
        sep=" | ",
    )
    if trainer_name != TRAINER_NAME:
        print(f"WARNING: trainer name is {trainer_name!r}, expected {TRAINER_NAME!r}")


def poll(args):
    deadline = time.time() + args.max_minutes * 60
    while True:
        job = None
        try:
            job = get_json(f"/api/jobs/{args.job_id}")
            line = (
                f"{utc()} status={job['status']} progress={job['progress']} "
                f"message={job.get('message')!r} extra={json.dumps(job.get('extra_info'))[:400]}"
            )
        except (SystemExit, OSError, ValueError) as err:
            line = f"{utc()} poll-error {type(err).__name__}: {str(err)[:300]}"
        print(line, flush=True)
        with open(args.log, "a") as f:
            f.write(line + "\n")
        if job is not None and job["status"] in TERMINAL:
            return
        if time.time() >= deadline:
            print("max-minutes reached; job not terminal", flush=True)
            return
        time.sleep(60)


def read_metrics(job_id, seconds=8):
    lines = []
    req = urllib.request.Request(
        BASE + f"/api/jobs/{job_id}/model_metrics", headers={"Accept": "text/event-stream"}
    )
    try:
        with urllib.request.urlopen(req, timeout=seconds) as resp:
            end = time.time() + seconds
            while time.time() < end:
                raw = resp.readline()
                if not raw:
                    break
                lines.append(raw.decode(errors="replace").rstrip())
    except Exception as err:
        lines.append(f"[stream closed: {type(err).__name__}]")
    return lines


def finish(args):
    job = get_json(f"/api/jobs/{args.job_id}")
    models = get_json(f"/api/projects/{PROJECT_ID}/models")
    rows = models if isinstance(models, list) else models.get("models", [])
    mine = [m for m in rows if m.get("train_job_id") == args.job_id]
    result = {"job": job, "metrics_tail": read_metrics(args.job_id)[-30:], "models": mine}
    if mine:
        result["model_detail"] = get_json(f"/api/models/{mine[0]['id']}")
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("status", job["status"], "models", [m["id"] for m in mine])
    if mine:
        detail = result["model_detail"]
        model = detail.get("model", detail)
        print("available_backends", model.get("available_backends"), "training_summary", detail.get("training_summary"))


def main(argv):
    ap = argparse.ArgumentParser(description="P2-T-TRAIN E1 Studio ACT training helper")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--dataset-id", required=True)
    b.add_argument("--name", required=True)
    b.add_argument("--episodes", type=int, required=True)
    b.add_argument("--frames", type=int, required=True)
    b.add_argument("--gpu", default="A10G", choices=["A10G", "A100-80GB", "H100"])
    b.add_argument("--out", required=True)
    v = sub.add_parser("validate")
    v.add_argument("payload")
    sub.add_parser("jobs")
    s = sub.add_parser("submit")
    s.add_argument("payload")
    s.add_argument("response")
    p = sub.add_parser("poll")
    p.add_argument("job_id")
    p.add_argument("--log", required=True)
    p.add_argument("--max-minutes", type=int, default=600)
    f = sub.add_parser("finish")
    f.add_argument("job_id")
    f.add_argument("out")
    args = ap.parse_args(argv)
    handlers = {"build": build, "validate": validate, "jobs": jobs, "submit": submit, "poll": poll, "finish": finish}
    handlers[args.cmd](args)


if __name__ == "__main__":
    main(sys.argv[1:])
