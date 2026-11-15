"""Materialize all 32 run specifications; no provisioning or training."""

import argparse, itertools, hashlib, importlib.metadata
from pathlib import Path
from prepare import dump


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    runs = []
    for model, marked, aug, domain in itertools.product(
        ["smolvla", "act"], ["N", "X"], range(4), ["R", "M"]
    ):
        name = f"{model}-{marked}-A{aug}-{domain}-seed0"
        spec = {
            "id": name,
            "model": model,
            "conditioning": marked,
            "augmentation": aug,
            "domain": domain,
            "seed": 0,
            "steps": 20000 if model == "smolvla" else 100000,
            "batch_size": 32 if model == "smolvla" else 8,
            "real_dataset": "real_corrected",
            "real_markers": "real_markers.npz",
            "sim_dataset": "sim_corrected",
            "sim_markers": "sim_markers.npz",
            "sim_fraction": 0.2 if domain == "M" else 0.0,
            "split_manifest": "splits.json",
            "cameras": ["overhead", "wrist"],
            "sample_policy": "domain_then_uniform_episode_then_uniform_frame",
            "output_dir": f"runs/{name}",
            "status": "prepared_pending_batch_validation",
            "checkpoint_every": 2000 if model == "smolvla" else 10000,
        }
        dump(root / "experiments" / f"{name}.json", spec)
        runs.append(spec)
    dump(root / "experiments.json", runs)
    import lerobot

    package = Path(lerobot.__file__).parent
    files = [
        "scripts/lerobot_train.py",
        "datasets/lerobot_dataset.py",
        "datasets/sampler.py",
        "configs/train.py",
    ]
    dump(
        root / "environment.json",
        {
            "python": "3.12",
            "lerobot": importlib.metadata.version("lerobot"),
            "torch": importlib.metadata.version("torch"),
            "source_hashes": {
                f: hashlib.sha256((package / f).read_bytes()).hexdigest() for f in files
            },
            "note": "Check these source hashes on CUDA host; do not rely on package version alone.",
        },
    )
    print(f"Prepared {len(runs)} experiments at {root}")


if __name__ == "__main__":
    main()
