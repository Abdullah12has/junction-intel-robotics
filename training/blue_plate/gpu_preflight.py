"""Disposable native CUDA training preflight at the campaign batch size."""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import torch
from run import configuration, datasets
from runtime import DomainSampler
from prepare import dump

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--model", choices=["act", "smolvla"], required=True)
p.add_argument("--steps", type=int, default=100)
a = p.parse_args()
sys.argv = [sys.argv[0]]
root = a.root.resolve()
spec = json.loads((root / "experiments" / f"{a.model}-X-A3-R-seed0.json").read_text())
cfg = configuration(spec, root)
cfg.steps = a.steps
cfg.save_freq = a.steps
cfg.eval_steps = 0
cfg.log_freq = 5
from lerobot.scripts import lerobot_train as trainer

with tempfile.TemporaryDirectory(prefix="gpu-preflight-", dir=root.parent) as temp:
    cfg.output_dir = Path(temp) / "run"
    cfg.validate()
    ds, val = datasets(spec, root, cfg)
    trainer.make_train_eval_datasets = lambda cfg: (ds, None)
    trainer.EpisodeAwareSampler = lambda *args, **kwargs: DomainSampler(ds, 0, 0)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    trainer.train.__wrapped__(cfg)
    torch.cuda.synchronize()
    result = {
        "model": a.model,
        "batch_size": spec["batch_size"],
        "steps": a.steps,
        "elapsed_s": time.monotonic() - start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "device": torch.cuda.get_device_name(),
        "checkpoint_saved": True,
    }
    dump(root / "validation" / f"{a.model}-gpu-preflight.json", result)
    print(json.dumps(result))
