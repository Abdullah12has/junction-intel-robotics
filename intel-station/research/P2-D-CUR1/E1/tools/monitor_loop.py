#!/usr/bin/env python3
"""Poll our training jobs every 60 s (read-only GET /api/jobs/<id>) and append one line per job per poll.

Usage: python3 -I monitor_loop.py <log_path> <job_id> [<job_id> ...]
Stops when every job is completed, failed or canceled.
"""
import datetime
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:7860"
DONE = {"completed", "failed", "canceled"}
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get(path):
    with OPENER.open(urllib.request.Request(BASE + path), timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    log_path, jobs = sys.argv[1], sys.argv[2:]
    pending = set(jobs)
    while pending:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S")
        for jid in sorted(pending):
            try:
                j = get(f"/api/jobs/{jid}")
                line = f"{stamp}Z {jid} {j.get('status')} progress={j.get('progress')} msg={j.get('message')}"
                if j.get("status") in DONE:
                    pending.discard(jid)
            except Exception as exc:  # keep polling on transient errors
                line = f"{stamp}Z {jid} poll-error {exc}"
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        time.sleep(60)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write("ALL-DONE\n")


if __name__ == "__main__":
    main()
