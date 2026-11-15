"""Prepare an explicit, local-only SmolVLA command after episode review.

Does not upload data, download models, or launch GPU jobs. Run the generated
argv on an isolated CUDA trainer with lerobot[smolvla]==0.7.0 installed.
"""
import argparse
import csv
import json
from pathlib import Path
import shlex

from audit_dataset import audit, STATION


def select_episodes(rows, available):
    approved = [r for r in rows if r["accepted"].strip().lower() == "yes"]
    if len(approved) < 50:
        raise ValueError("Review and accept at least 50 successful real episodes first")
    ids = [int(r["episode_index"]) for r in approved]
    if len(set(ids)) != len(ids) or not set(ids).issubset(available):
        raise ValueError("Accepted episode IDs must be unique and present in this dataset")
    if any(r["split"] not in ("train", "validation") for r in approved):
        raise ValueError("Every accepted episode must have train or validation split")
    train = [int(r["episode_index"]) for r in approved if r["split"] == "train"]
    val = [int(r["episode_index"]) for r in approved if r["split"] == "validation"]
    if len(train) < 40 or len(val) < 10:
        raise ValueError("Need at least 40 training and 10 validation episodes")
    return train, val


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=Path("sorting-recording-plan.csv"))
    p.add_argument("--task-mode", choices=("inspection", "sorting"), default="sorting")
    p.add_argument("--output", type=Path, default=Path("outputs/training-plan"))
    p.add_argument("--checkpoint", default="lerobot/smolvla_base")
    a = p.parse_args()
    report = audit(a.dataset, a.task_mode)
    if report["errors"] or any("exceeds" in w for w in report["warnings"]):
        raise ValueError(f"Resolve data issues first: {report['errors'] + report['warnings']}")
    with a.manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))
        train, val = select_episodes(rows, {e["episode_index"] for e in report["episodes"]})
    if a.task_mode == "sorting":
        tasks = {e['episode_index']: e['task'] for e in report['episodes']}
        for row in rows:
            if row['accepted'].strip().lower() == 'yes' and row.get('task') != tasks[int(row['episode_index'])]:
                raise ValueError("Recording manifest instruction differs from recorded episode task")
    # No simulator actions may enter this path: use reviewed original real data.
    argv = ["lerobot-train", f"--policy.path={a.checkpoint}",
            f"--dataset.repo_id=local/junction-{a.task_mode}-real", f"--dataset.root={a.dataset.resolve()}",
            f"--dataset.episodes={json.dumps(train)}", "--dataset.video_backend=pyav",
            f"--rename_map={json.dumps(STATION['camera_rename_map'])}",
            "--policy.device=cuda", "--policy.push_to_hub=false", "--batch_size=8",
            "--num_workers=4", "--steps=20000", "--save_freq=2000", "--eval_freq=0",
            "--seed=42", f"--output_dir=outputs/train/smolvla-{a.task_mode}-real", f"--job_name=junction-{a.task_mode}-real"]
    a.output.mkdir(parents=True, exist_ok=False)
    plan = {"lerobot_version": "0.7.0", "argv": argv, "train_episodes": train,
            "validation_episodes": val, "validation_is_not_automatically_run": True,
            "source": "real", "checkpoint": a.checkpoint, "audit": report}
    (a.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (a.output / "command.txt").write_text(shlex.join(argv) + "\n")
    print(f"Prepared command at {a.output}. No training started. Adjust dataset.root on the CUDA host.")


if __name__ == "__main__":
    main()
