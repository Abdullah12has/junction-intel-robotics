#!/usr/bin/env python3
"""Submit one training job to Hassan's 'Modal' remote trainer (ca5568b5) for a NEW curated dataset, found by name.

Usage: python3 -I submit.py <dataset_name> <act|smolvla> <model_name> <max_epochs>
Only POST /api/jobs:train is written. SmolVLA starts from its pretrained weights (base_model_id null).
"""
import json
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:7860"
PROJECT = "ab767473-d6fd-47ff-812e-8239c455dabb"
TRAINER = "ca5568b5-4542-4fe2-9932-c09a4c1bdbaa"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get(path):
    with OPENER.open(urllib.request.Request(BASE + path), timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    name, policy, model_name, epochs = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    projects = [p for p in get("/api/projects") if p.get("id") == PROJECT]
    matches = [d for d in projects[0].get("datasets", []) if d.get("name") == name]
    if len(matches) != 1:
        raise SystemExit(f"dataset name not unique or missing: {name} ({len(matches)} matches)")
    ds_id = matches[0]["id"]
    payload = {
        "project_id": PROJECT,
        "dataset_id": ds_id,
        "policy": policy,
        "model_name": model_name,
        "max_epochs": epochs,
        "max_steps": None,
        "batch_size": 8,
        "num_workers": "auto",
        "auto_scale_batch_size": False,
        "base_model_id": None,
        "val_split": 0.0,
        "device": None,
        "precision": "bf16-mixed",
        "compile_model": False,
        "augment_images": False,
        "lora_enabled": False,
        "snapflow_enabled": False,
        "export_backends": ["openvino"],
        "image_key_reorder_map": {},
        "num_cameras": 0,
        "training_target": "remote",
        "remote_trainer_id": TRAINER,
        "modal_gpu": "A10G",
    }
    if policy == "smolvla":
        payload["image_key_reorder_map"] = {"realsense 243622061421": 0, "realsense 243722064316": 1}
        payload["num_cameras"] = 2
    req = urllib.request.Request(
        BASE + "/api/jobs:train",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with OPENER.open(req, timeout=120) as resp:
            out = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        raise SystemExit(f"HTTP {err.code}: {err.read().decode('utf-8', 'replace')[:600]}")
    print(json.dumps({
        "job_id": out.get("id"), "status": out.get("status"), "dataset_id": ds_id, "dataset_name": name,
        "policy": policy, "max_epochs": epochs, "model_name": model_name,
    }))


if __name__ == "__main__":
    main()
