#!/usr/bin/env python3
"""Wait for a Studio dataset import job to reach review, finalize it as a NEW dataset, wait until completed.

Usage: python3 -I import_wait.py <import_job_id> <default_task_text>
Finalize is the only write (POST .../imports/datasets/<job>:finalize); status is read with GET /api/jobs/<job>.
"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:7860"
PROJECT = "ab767473-d6fd-47ff-812e-8239c455dabb"
ENV = "72206569-df16-4a8b-a98f-247bac4de63f"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call(method, path, body=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    try:
        with OPENER.open(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "null")
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode("utf-8", "replace")[:500]


def job(job_id):
    _, body = call("GET", f"/api/jobs/{job_id}")
    return body


def main():
    job_id, task = sys.argv[1], sys.argv[2]
    deadline = time.time() + 900
    while True:
        j = job(job_id)
        step = (j.get("payload") or {}).get("step")
        print("status", j.get("status"), "step", step, "progress", j.get("progress"), "msg", j.get("message"),
              flush=True)
        if j.get("status") in ("failed", "canceled"):
            raise SystemExit("import failed before review: " + json.dumps(j.get("issues"))[:800])
        if step == "awaiting_user_review" or j.get("status") == "awaiting_user_review":
            break
        if time.time() > deadline:
            raise SystemExit("timeout waiting for review")
        time.sleep(4)
    report = (j.get("payload") or {}).get("validation_report")
    print("validation_report:", json.dumps(report)[:600], "issues:", json.dumps(j.get("issues"))[:400], flush=True)
    code, body = call("POST", f"/api/projects/{PROJECT}/imports/datasets/{job_id}:finalize",
                      {"environment_id": ENV, "default_task": task})
    print("finalize http", code, str(body)[:300], flush=True)
    while True:
        j = job(job_id)
        payload = j.get("payload") or {}
        if j.get("status") == "completed" or payload.get("step") == "completed":
            manifest = payload.get("dataset_manifest_draft") or {}
            print("COMPLETED", json.dumps({
                "result_dataset_id": payload.get("result_dataset_id"),
                "statistics": manifest.get("statistics"),
                "tasks": (manifest.get("dataset_schema") or {}).get("tasks"),
            }), flush=True)
            return
        if j.get("status") in ("failed", "canceled"):
            raise SystemExit("import failed after finalize: " + json.dumps(j.get("issues"))[:800])
        if time.time() > deadline:
            raise SystemExit("timeout after finalize")
        time.sleep(4)


if __name__ == "__main__":
    main()
