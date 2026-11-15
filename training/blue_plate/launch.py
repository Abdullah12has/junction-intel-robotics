"""Local queue for one provisioned multi-GPU host. Dry run unless --execute."""

import argparse, json, os, subprocess, sys, time, fcntl
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--devices", default="0")
    p.add_argument("--execute", action="store_true")
    p.add_argument(
        "--jobs-per-gpu",
        type=int,
        default=1,
        help="Use >1 only after concurrent GPU memory preflight",
    )
    p.add_argument("--domain", choices=["R", "M", "all"], default="all")
    a = p.parse_args()
    root = a.root.resolve()
    devices = a.devices.split(",")
    if len(set(devices)) != len(devices):
        raise ValueError(
            "One worker per distinct GPU; profile before adding shared-GPU workers"
        )
    if a.jobs_per_gpu < 1:
        p.error("--jobs-per-gpu must be positive")
    slots = [(gpu, i) for gpu in devices for i in range(a.jobs_per_gpu)]
    runs = [
        r
        for r in json.loads((root / "experiments.json").read_text())
        if a.domain == "all" or r["domain"] == a.domain
    ]
    # Interleave models so shared GPU workers start ACT and SmolVLA together.
    runs.sort(key=lambda r: (r["conditioning"], r["augmentation"], r["model"]))
    commands = [
        (
            r["id"],
            [
                sys.executable,
                str(Path(__file__).with_name("run.py")),
                "--root",
                str(root),
                "--experiment",
                r["id"],
                "--train",
            ],
        )
        for r in runs
    ]
    if not a.execute:
        print(
            json.dumps(
                {
                    "concurrency": len(slots),
                    "jobs": len(commands),
                    "commands": [{"id": name, "argv": cmd} for name, cmd in commands],
                },
                indent=2,
            )
        )
        return
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    lock = (root / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active = {}
    finished = []
    try:
        while commands or active:
            for slot in slots:
                gpu = slot[0]
                if slot not in active and commands:
                    name, cmd = commands.pop(0)
                    log = (logs / f"{name}.log").open("a")
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = gpu
                    env["TOKENIZERS_PARALLELISM"] = "false"
                    env["OMP_NUM_THREADS"] = "4"
                    active[slot] = (
                        name,
                        subprocess.Popen(
                            cmd, env=env, stdout=log, stderr=subprocess.STDOUT
                        ),
                        log,
                    )
            for slot, (name, proc, log) in list(active.items()):
                code = proc.poll()
                if code is not None:
                    log.close()
                    finished.append(
                        {"id": name, "gpu": slot[0], "slot": slot[1], "exit_code": code}
                    )
                    del active[slot]
                    (root / "queue_results.json").write_text(
                        json.dumps(finished, indent=2)
                    )
            status = {
                "updated_at": time.time(),
                "running": [
                    {"id": name, "pid": proc.pid, "gpu": slot[0]}
                    for slot, (name, proc, log) in active.items()
                ],
                "queued": [name for name, cmd in commands],
                "finished": finished,
            }
            temporary = root / "queue_status.tmp"
            temporary.write_text(json.dumps(status, indent=2))
            temporary.replace(root / "queue_status.json")
            time.sleep(1)
    finally:
        for name, proc, log in active.values():
            proc.terminate()
            proc.wait()
            log.close()
    if any(r["exit_code"] for r in finished):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
