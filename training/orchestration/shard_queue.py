"""Run a disjoint campaign shard, optionally adopting existing Linux worker PIDs.

Keep this outside the immutable training code directory used by resume contracts.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def process(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0], fields[19]
    except FileNotFoundError:
        return None


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    a = p.parse_args()
    m = json.loads(a.manifest.read_text())
    root = Path(m["root"])
    lock = (root / "shard_queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ids = m["experiments"]
    assert len(ids) == len(set(ids))
    active = {}
    finished = []
    status_path = root / "shard_status.json"
    if status_path.exists():
        raise RuntimeError(
            "Existing shard status: inspect results and make an explicit recovery manifest"
        )
    for job in m.get("adopt", []):
        assert job["id"] in ids
        live = process(job["pid"])
        assert live and live[1] == job["start_ticks"], "Adopted PID identity changed"
        active[job["id"]] = {**job, "proc": None, "log": None}
    pending = [i for i in ids if i not in active]
    (root / "logs").mkdir(exist_ok=True)
    while active or pending:
        supervisor = m.get("paused_supervisor")
        if supervisor:
            live = process(supervisor["pid"])
            if live and (
                live[1] != supervisor["start_ticks"] or live[0] not in ["T", "t"]
            ):
                raise RuntimeError(
                    "Original scheduler is not paused; refusing duplicate dispatch"
                )
        for name, job in list(active.items()):
            if job["proc"] is not None:
                code = job["proc"].poll()
            else:
                live = process(job["pid"])
                if live and live[1] == job["start_ticks"] and live[0] != "Z":
                    continue
                # Adopted processes have another parent. Verify terminal artifact and log.
                spec = json.loads((root / "experiments" / f"{name}.json").read_text())
                output = root / spec["output_dir"]
                checkpoints = list(
                    output.glob("checkpoints/*/training_state/training_step.json")
                )
                complete = any(
                    json.loads(f.read_text()).get("step") == spec["steps"]
                    for f in checkpoints
                )
                log = (root / "logs" / f"{name}.log").read_text()
                code = 0 if complete and "End of training" in log else 1
            if code is None:
                continue
            if job["log"]:
                job["log"].close()
            finished.append({"id": name, "exit_code": code, "finished_at": time.time()})
            del active[name]
        while pending and len(active) < m["concurrency"]:
            name = pending.pop(0)
            spec = json.loads((root / "experiments" / f"{name}.json").read_text())
            if (root / spec["output_dir"]).exists():
                raise RuntimeError(f"Output exists: {name}; explicit resume required")
            log = (root / "logs" / f"{name}.log").open("a")
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "0",
                "OMP_NUM_THREADS": "4",
                "TOKENIZERS_PARALLELISM": "false",
            }
            proc = subprocess.Popen(
                [
                    sys.executable,
                    m["runner"],
                    "--root",
                    str(root),
                    "--experiment",
                    name,
                    "--train",
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            active[name] = {"pid": proc.pid, "proc": proc, "log": log}
        save(
            status_path,
            {
                "updated_at": time.time(),
                "shard": m["shard"],
                "running": [{"id": n, "pid": j["pid"]} for n, j in active.items()],
                "queued": pending,
                "finished": finished,
            },
        )
        time.sleep(2)
    # Existing children have ended; retiring their paused parent cannot interrupt them.
    if m.get("paused_supervisor"):
        s = m["paused_supervisor"]
        live = process(s["pid"])
        if live and live[1] == s["start_ticks"]:
            os.kill(s["pid"], signal.SIGKILL)
    if any(x["exit_code"] for x in finished):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
