"""Validate or launch exactly one campaign experiment on its assigned GPU.

No cloud APIs or physical robot interfaces. Dry-run is default; --train is explicit.
"""

import argparse, copy, json, sys, hashlib
from pathlib import Path
import numpy as np
from prepare import OVER, WRIST, dump
from runtime import VariantDataset, DomainSampler


def configuration(spec, root, device="cuda"):
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.default import DatasetConfig
    from lerobot.policies.act.configuration_act import ACTConfig

    if spec["model"] == "smolvla":
        from lerobot.configs.policies import PreTrainedConfig

        policy = PreTrainedConfig.from_pretrained(root / "base_config")
        policy.input_features.pop("observation.images.camera3", None)
        policy.device = device
        policy.push_to_hub = False
        policy.pretrained_path = Path("lerobot/smolvla_base")
    else:
        policy = ACTConfig(device=device, push_to_hub=False)
    if spec["model"] == "smolvla":
        policy.pretrained_revision = json.loads(
            (root / "base_checkpoint.json").read_text()
        )["revision"]
    split = json.loads((root / "splits.json").read_text())["episodes"]
    return TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id="local/blue-plate-corrected",
            root=str(root / spec["real_dataset"]),
            episodes=split["train"],
            video_backend="pyav",
            use_imagenet_stats=True,
            eval_split=0.15,
        ),
        policy=policy,
        output_dir=root / spec["output_dir"],
        job_name=spec["id"],
        seed=spec["seed"],
        batch_size=spec["batch_size"],
        steps=spec["steps"],
        num_workers=6,
        save_freq=spec["checkpoint_every"],
        env_eval_freq=0,
        eval_steps=2000,
        max_eval_samples=512,
        rename_map={
            OVER: "observation.images.camera1",
            WRIST: "observation.images.camera2",
        }
        if spec["model"] == "smolvla"
        else {},
    )


def datasets(spec, root, cfg):
    from lerobot.datasets.factory import make_dataset

    real = make_dataset(cfg)
    valcfg = copy.deepcopy(cfg)
    valcfg.dataset.episodes = json.loads((root / "splits.json").read_text())[
        "episodes"
    ]["validation"]
    val = make_dataset(valcfg)
    markers = np.load(root / spec["real_markers"])["xy"]
    sim = None
    sm = None
    if spec["domain"] == "M":
        scfg = copy.deepcopy(cfg)
        scfg.dataset.root = str(root / spec["sim_dataset"])
        scfg.dataset.repo_id = "local/blue-plate-sim"
        scfg.dataset.episodes = json.loads((root / "sim_splits.json").read_text())[
            "train"
        ]
        sim = make_dataset(scfg)
        sm = np.load(root / spec["sim_markers"])["xy"]
    ds = VariantDataset(
        real, markers, spec["augmentation"], spec["conditioning"] == "X", sim, sm
    )
    validation = VariantDataset(
        val, markers, 0, spec["conditioning"] == "X", training=False
    )
    from torch.utils.data import Subset

    validation = Subset(
        validation,
        np.linspace(
            0, len(validation) - 1, min(512, len(validation)), dtype=int
        ).tolist(),
    )
    return ds, validation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--check-batch", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--resume", type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    spec = json.loads((root / "experiments" / f"{a.experiment}.json").read_text())
    cfg = configuration(spec, root)
    # No shell interpolation. Keep parser wrapper from consuming campaign-specific flags.
    sys.argv = [sys.argv[0]]
    if a.resume:
        from lerobot.configs.train import TrainPipelineConfig

        checkpoint = a.resume.resolve()
        config_path = checkpoint / "pretrained_model/train_config.json"
        saved = TrainPipelineConfig.from_pretrained(config_path)
        if (
            saved.job_name != spec["id"]
            or saved.batch_size != spec["batch_size"]
            or saved.steps != spec["steps"]
        ):
            raise ValueError("Resume contract differs from experiment")
        saved.resume = True
        saved.dataset.root = str(root / spec["real_dataset"])
        saved.output_dir = root / spec["output_dir"]
        saved.policy.device = "cuda"
        cfg = saved
        sys.argv = [sys.argv[0], f"--config_path={config_path}"]
    cfg.validate()
    dump(root / "resolved_configs" / f"{spec['id']}.json", cfg.to_dict())
    required = [
        root / spec["real_dataset"] / "meta/info.json",
        root / spec["real_markers"],
    ]
    if spec["domain"] == "M":
        required += [
            root / spec["sim_dataset"] / "meta/info.json",
            root / spec["sim_markers"],
            root / "sim_splits.json",
        ]
    missing = [str(x) for x in required if not x.exists()]
    report = {
        "id": spec["id"],
        "config_valid": True,
        "missing": missing,
        "training_started": False,
    }
    if missing:
        print(json.dumps(report))
        if a.train or a.check_batch:
            raise SystemExit(2)
        return
    ds, val = datasets(spec, root, cfg)
    if a.check_batch:
        from torch.utils.data import DataLoader

        sampler = DomainSampler(ds, spec["seed"], spec["sim_fraction"])
        batch = next(iter(DataLoader(ds, batch_size=2, sampler=sampler, num_workers=0)))
        report["batch_shapes"] = {
            k: list(v.shape) for k, v in batch.items() if hasattr(v, "shape")
        }
        report["tasks"] = batch["task"]
        report["train_frames"] = len(ds)
        report["validation_frames"] = len(val)
    dump(root / "validation" / f"{spec['id']}.json", report)
    if not a.train:
        print(json.dumps(report))
        return
    import lerobot, torch

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires assigned CUDA GPU")
    expected = json.loads((root / "environment.json").read_text())["source_hashes"]
    pkg = Path(lerobot.__file__).parent
    for f, sha in expected.items():
        if hashlib.sha256((pkg / f).read_bytes()).hexdigest() != sha:
            raise RuntimeError(f"Unvalidated LeRobot source revision: {f}")
    readiness = json.loads((root / "readiness.json").read_text())
    for gate in (
        ["real_data_review", "batch_validation"]
        + (["marker_review"] if spec["conditioning"] == "X" else [])
        + (["simulation_compatibility"] if spec["domain"] == "M" else [])
    ):
        if not readiness.get(gate, False):
            raise RuntimeError(f"Unresolved preparation gate: {gate}")
    contract_files = [
        "splits.json",
        "real_markers.npz",
        "real_corrected/meta/stats.json",
        "provenance.json",
    ]
    if spec["domain"] == "M":
        contract_files += ["sim_splits.json", "sim_markers.npz", "sim_provenance.json"]
    contract = {
        "spec": spec,
        "files": {
            f: hashlib.sha256((root / f).read_bytes()).hexdigest()
            for f in contract_files
        },
        "code": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in Path(__file__).parent.glob("*.py")
        },
    }
    contract_path = root / "launch_contracts" / f"{spec['id']}.json"
    if a.resume:
        if json.loads(contract_path.read_text()) != contract:
            raise RuntimeError("Dataset/code/experiment changed since original launch")
    else:
        dump(contract_path, contract)
    from lerobot.scripts import lerobot_train as trainer

    trainer.make_train_eval_datasets = lambda cfg: (ds, val)
    trainer.EpisodeAwareSampler = lambda *args, **kwargs: DomainSampler(
        ds, spec["seed"], spec["sim_fraction"]
    )
    trainer.train.__wrapped__(cfg)


if __name__ == "__main__":
    main()
