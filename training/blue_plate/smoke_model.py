"""A local forward/backward validation; never a campaign training run."""

import argparse, json, sys, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from run import configuration, datasets
from runtime import DomainSampler
from prepare import dump


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--model", choices=["act", "smolvla"], default="act")
    a = p.parse_args()
    root = a.root.resolve()
    sys.argv = [sys.argv[0]]
    torch.set_num_threads(4)
    spec = json.loads(
        (root / "experiments" / f"{a.model}-X-A3-R-seed0.json").read_text()
    )
    cfg = configuration(spec, root, device=a.device)
    cfg.validate()
    ds, _ = datasets(spec, root, cfg)
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    t0 = time.monotonic()
    policy = make_policy(cfg.policy, ds_meta=ds.meta, rename_map=cfg.rename_map)
    # Use the same processor overrides as the native training entry point.
    kwargs = {"dataset_stats": ds.meta.stats}
    if cfg.policy.pretrained_path:
        kwargs.update(
            preprocessor_overrides={
                "device_processor": {"device": a.device},
                "normalizer_processor": {
                    "stats": ds.meta.stats,
                    "features": {
                        **policy.config.input_features,
                        **policy.config.output_features,
                    },
                    "norm_map": policy.config.normalization_mapping,
                },
                "rename_observations_processor": {"rename_map": cfg.rename_map},
            },
            postprocessor_overrides={
                "unnormalizer_processor": {
                    "stats": ds.meta.stats,
                    "features": policy.config.output_features,
                    "norm_map": policy.config.normalization_mapping,
                }
            },
        )
    pre, post = make_pre_post_processors(
        cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        pretrained_revision=cfg.policy.pretrained_revision,
        **kwargs,
    )
    batch = next(
        iter(
            DataLoader(
                ds,
                batch_size=a.batch_size,
                sampler=DomainSampler(ds, 0, 0),
                num_workers=0,
            )
        )
    )
    for k in ds.meta.camera_keys:
        batch[k] = batch[k].float() / 255
    processed = pre(batch)
    policy.train()
    for _ in range(a.iterations):
        policy.zero_grad(set_to_none=True)
        loss, _ = policy.forward(processed)
        assert torch.isfinite(loss)
        loss.backward()
    if a.device == "cuda":
        torch.cuda.synchronize()
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    result = {
        "model": a.model,
        "forward_loss_finite": True,
        "backward_gradients_finite": True,
        "loss": float(loss.detach()),
        "elapsed_s": time.monotonic() - t0,
        "device": a.device,
        "batch_size": a.batch_size,
        "iterations": a.iterations,
        "peak_vram_bytes": torch.cuda.max_memory_allocated()
        if a.device == "cuda"
        else None,
        "optimizer_steps": 0,
        "note": "No campaign training or CUDA throughput measured",
    }
    dump(root / "validation" / f"{a.model}-{a.device}-model-smoke.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
