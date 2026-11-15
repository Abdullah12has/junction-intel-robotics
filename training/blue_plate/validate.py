"""Check sources, splits, all configs and all real/mixed sample paths without training."""

import argparse, json, hashlib, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch
from PIL import Image, ImageDraw
from prepare import OVER, WRIST, dump
from run import configuration, datasets
from runtime import DomainSampler, VariantDataset
from torch.utils.data import DataLoader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    sys.argv = [sys.argv[0]]
    root = a.root.resolve()
    splits = json.loads((root / "splits.json").read_text())
    ids = splits["episodes"]
    assert not (set(ids["train"]) & set(ids["test"]))
    assert not (set(ids["train"]) & set(ids["validation"]))
    assert not (set(ids["test"]) & set(ids["validation"]))
    assert sorted(sum(ids.values(), [])) == list(range(64))
    for group in splits["groups"]:
        assert sum(bool(set(group) & set(v)) for v in ids.values()) == 1
    provenance = json.loads((root / "provenance.json").read_text())
    src = Path(provenance["source"])
    hashes = {
        str(p.relative_to(src)): hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
        for p in src.rglob("*")
        if p.is_file()
    }
    assert hashes == provenance["source_sha256"]
    for rel, sha in hashes.items():
        if rel.startswith(("data/", "videos/")):
            assert (
                hashlib.file_digest(
                    (root / "real_corrected" / rel).open("rb"), "sha256"
                ).hexdigest()
                == sha
            )
    data = pq.read_table(root / "real_corrected/data").to_pydict()
    mask = np.isin(data["episode_index"], ids["train"])
    stats = json.loads((root / "real_corrected/meta/stats.json").read_text())
    for k in ["action", "observation.state"]:
        np.testing.assert_allclose(
            np.mean(np.array(data[k])[mask], axis=0), stats[k]["mean"], rtol=1e-6
        )
    assert (
        pq.read_table(root / "real_corrected/meta/tasks.parquet")
        .to_pylist()[0]["task"]
        .endswith("blue plate.")
    )
    results = []
    cache = {}
    sample_images = []
    for spec in json.loads((root / "experiments.json").read_text()):
        cfg = configuration(spec, root, device="cpu")
        cfg.validate()
        dump(root / "resolved_configs" / f"{spec['id']}.json", cfg.to_dict())
        report = {"id": spec["id"], "config_valid": True, "missing": []}
        if (
            spec["domain"] == "M"
            and not (root / "sim_corrected/meta/info.json").exists()
        ):
            report.update(batch_valid=False, missing=["sim_corrected export"])
            results.append(report)
            continue
        key = (spec["model"], spec["domain"])
        if key not in cache:
            cache[key] = datasets(spec, root, cfg)
        base, val = cache[key]
        ds = VariantDataset(
            base.real,
            base.markers[0],
            spec["augmentation"],
            spec["conditioning"] == "X",
            base.sim,
            base.markers[1],
        )
        sampler = DomainSampler(ds, spec["seed"], spec["sim_fraction"])
        indices = list(iter(sampler))[:2]
        batch = next(iter(DataLoader(ds, batch_size=2, sampler=indices, num_workers=0)))
        assert batch["action"].shape == (2, cfg.policy.chunk_size, 6)
        for k in [OVER, WRIST]:
            assert batch[k].shape == (2, 3, 480, 640) and batch[k].dtype == torch.uint8
        assert torch.isfinite(batch["action"]).all()
        assert torch.isfinite(batch["observation.state"]).all()
        # Verify transforms do not alter demonstrated commands and episode-tail padding.
        for j, (idx, seed) in enumerate(indices):
            underlying = ds.real if idx < len(ds.real) else ds.sim
            ii = idx if idx < len(ds.real) else idx - len(ds.real)
            raw = underlying[ii]
            assert torch.equal(batch["action"][j], raw["action"])
            assert torch.equal(batch["action_is_pad"][j], raw["action_is_pad"])
        # Explicitly exercise the sim branch even if the first random draws happen to be real.
        if ds.sim is not None:
            simitem = ds[(len(ds.real), 123)]
            assert simitem["action"].shape == (cfg.policy.chunk_size, 6)
        report.update(
            batch_valid=True,
            train_frames=len(ds),
            validation_frames=len(val),
            batch_shapes={
                k: list(batch[k].shape)
                for k in [OVER, WRIST, "action", "observation.state"]
            },
        )
        dump(root / "validation" / f"{spec['id']}.json", report)
        results.append(report)
        if spec["model"] == "act" and spec["domain"] == "R":
            sample_images.append(
                (spec["id"], ds[(0, 1234)][OVER].permute(1, 2, 0).numpy())
            )
    sheet = Image.new("RGB", (4 * 320, 2 * 270), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (name, im) in enumerate(sample_images):
        x = (i % 4) * 320
        y = (i // 4) * 270
        sheet.paste(Image.fromarray(im).resize((320, 240)), (x, y + 25))
        draw.text((x + 4, y + 4), name, fill="black")
    sheet.save(root / "qa/augmentation-matrix.jpg")
    report = {
        "original_checksums_unchanged": True,
        "derived_data_and_videos_unchanged": True,
        "group_split_no_leakage": True,
        "train_only_stats_verified": True,
        "configurations": len(results),
        "batch_passed": sum(r.get("batch_valid", False) for r in results),
        "runs": results,
        "scope": "CPU dataset/config validation, no CUDA throughput or training/model validation implied",
    }
    dump(root / "validation_summary.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}))


if __name__ == "__main__":
    main()
