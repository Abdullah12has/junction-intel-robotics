"""Two CPU optimizer updates + checkpoint reload/resume check in temporary storage."""

import argparse, json, sys, tempfile
from pathlib import Path
import torch
from run import configuration, datasets
from runtime import DomainSampler
from prepare import dump


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--model", choices=["act", "smolvla"], default="act")
    a = p.parse_args()
    root = a.root.resolve()
    sys.argv = [sys.argv[0]]
    torch.set_num_threads(4)
    spec = json.loads(
        (root / "experiments" / f"{a.model}-X-A3-R-seed0.json").read_text()
    )
    cfg = configuration(spec, root, device="cpu")
    cfg.batch_size = 1
    cfg.steps = 1
    cfg.save_freq = 1
    cfg.eval_steps = 0
    cfg.num_workers = 0
    cfg.log_freq = 1
    from lerobot.scripts import lerobot_train as trainer
    from lerobot.configs.train import TrainPipelineConfig

    with tempfile.TemporaryDirectory(prefix="blue-plate-integration-") as temp:
        cfg.output_dir = Path(temp) / "run"
        cfg.validate()
        ds, val = datasets(spec, root, cfg)
        trainer.make_train_eval_datasets = lambda cfg: (ds, None)
        trainer.EpisodeAwareSampler = lambda *args, **kwargs: DomainSampler(ds, 0, 0)
        trainer.train.__wrapped__(cfg)
        checkpoint = cfg.output_dir / "checkpoints/000001"
        config_path = checkpoint / "pretrained_model/train_config.json"
        loaded = TrainPipelineConfig.from_pretrained(config_path)
        loaded.resume = True
        loaded.steps = 2
        loaded.policy.device = "cpu"
        sys.argv = [sys.argv[0], f"--config_path={config_path}"]
        trainer.train.__wrapped__(loaded)
        final = json.loads(
            (
                cfg.output_dir / "checkpoints/000002/training_state/training_step.json"
            ).read_text()
        )
        assert final["step"] == 2
        result = {
            "model": a.model,
            "native_training_loop": True,
            "checkpoint_reload": True,
            "optimizer_resume_to_step": 2,
            "custom_sampler_resume_test": "separate exact-index unit test passed",
            "device": "cpu",
            "temporary_checkpoints_removed": True,
            "campaign_training_started": False,
            "note": "Two disposable CPU integration-test optimizer updates; not a policy evaluation or CUDA spot recovery test",
        }
    dump(root / "validation" / f"{a.model}-training-integration.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
