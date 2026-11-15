"""Snapshot prepared inputs and their validation status; never launches training."""

import argparse, json, hashlib, shutil
from pathlib import Path
from prepare import dump


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    code = Path(__file__).parent
    checks = json.loads((root / "validation_summary.json").read_text())
    assert checks["configurations"] == checks["batch_passed"] == 32
    sim = json.loads((root / "sim_provenance.json").read_text())
    real = json.loads((root / "real_corrected/meta/info.json").read_text())
    splits = json.loads((root / "splits.json").read_text())["episodes"]
    ss = json.loads((root / "sim_splits.json").read_text())
    runs = json.loads((root / "experiments.json").read_text())
    for r in runs:
        r["status"] = (
            "prepared_for_cuda_preflight"
            if r["domain"] == "R"
            else "prepared_candidate_blocked_on_sim_camera_contact_alignment"
        )
        r["batch_validation_passed"] = True
        dump(root / "experiments" / f"{r['id']}.json", r)
    dump(root / "experiments.json", runs)
    # Copy source without bytecode. This snapshot is sufficient to run real/mixed exports;
    # generation additionally depends on the versioned simulator identified by its hashes.
    dst = root / "source"
    dst.mkdir(exist_ok=True)
    for f in code.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    for name in [
        "BLUE-PLATE-PARALLEL-EXPERIMENTS.md",
        "BLUE-PLATE-VERDA-COST.md",
        "MUJOCO-SYNTHETIC-DATA-RESEARCH.md",
    ]:
        shutil.copy2(code.parents[1] / "docs" / name, root / name)
    (root / "cost-evidence").mkdir(exist_ok=True)
    for f in (code.parents[1] / "evidence/blue-plate-campaign").glob("verda*.json"):
        shutil.copy2(f, root / "cost-evidence" / f.name)
    manifest = {
        str(p.relative_to(root)): hashlib.file_digest(
            p.open("rb"), "sha256"
        ).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
        and p.name not in ["bundle-checksums.json", "PREPARATION-STATUS.md"]
        and "__pycache__" not in p.parts
    }
    dump(root / "bundle-checksums.json", manifest)
    text = f"""# Blue-plate preparation status

Prepared 15 November 2026. **All 32 specifications and input pipelines exist and pass CPU batch validation. Sixteen real-only runs are prepared for CUDA preflight. Sixteen mixed runs remain blocked on simulator camera/contact alignment.** No paid resources or campaign training were started.

## Data

- Original: unchanged, SHA-256 rechecked. Independent corrected copy: {real["total_episodes"]} episodes, {real["total_frames"]} frames; corrected blue-plate task text and overhead + wrist model inputs.
- Real split: {len(splits["train"])} training / {len(splits["validation"])} validation / {len(splits["test"])} test episodes; 23 whole reset/layout families; no family crosses partitions. No trimming was applied. Outcome review was sampled, not frame-by-frame certification.
- Target identity and initial image points annotated for all 64 episodes. `real_markers.npz` contains the causal cue. The X is suppressed at occlusion/appearance loss or displacement >8 pixels; it is not a perfect tracker following hidden/held objects. Runtime must use the same click/tracker/render pipeline.
- Four on-the-fly augmentation recipes, applied through `source/runtime.py`; originals and action chunks unchanged. No duplicated pseudo-demonstrations.
- MuJoCo pilot: {len(sim["episodes"])} accepted blue-plate candidates, {sim["total_frames"]} frames, both overhead and mounted-wrist videos; {len(ss["train"])} training / {len(ss["validation"])} validation episodes grouped by simulation seed. This is the 100-episode pilot, not the proposed 500–1,000 scale-up. Do not scale the mismatched camera configuration yet.
- Mixed sampling is explicit 80% real / 20% simulated, then uniform episode and frame; normalization always comes from real training data. Model validation uses 512 fixed samples spanning the real validation partition.

## Validation completed

- All 32 config schemas and batch paths pass; mixed branches explicitly exercised.
- Derived data/action/video bytes match original files; corrected task metadata verified; train-only numeric stats independently checked.
- Six targeted tests pass: sample-domain proportions, exact sample-order resume, marker suppression/no reacquisition, deterministic augmentation and invalid-marker handling.
- Both real ACT and pinned SmolVLA models passed forward/backward checks with finite losses/gradients.
- Native checkpoint save/load and optimizer resume passed for both models via two disposable CPU updates each. Temporary weights were removed. These are integration tests, not trained campaign models, policy success evidence or CUDA benchmarks.
- All 200 exported synthetic videos decoded with expected dimensions, FPS and frame counts. Accepted traces verify bilateral contact/lift, release and stable placement; failed attempts are excluded.

## Remaining launch conditions

1. **Simulation camera/contact alignment:** the mounted-wrist camera is a provisional model view and differs visibly from the recorded USB camera. Nominal rigid-jaw mapping also differs from the physical flexible fingers. `readiness.json` deliberately keeps `simulation_compatibility=false`; mixed runs fail before training. This is an observed preparation limitation, not a permission request. Matching the physical mount/intrinsics and validating contact/action alignment remain unfinished.
2. **CUDA preflight:** benchmark the exact models/batches/decoding/augmentation and checkpoint overhead on the chosen instance. Local tests do not establish GPU memory usage, concurrency, elapsed training time or spot recovery.
3. Live deployment of any X model requires the same selector/tracker; the scripts were tested on recordings, not connected to a live robot.

## Costs

Account-visible quote: currently listed single RTX PRO 6000 CC 96 GB, **$1.446/hour spot or $2.893/hour on-demand**. Eight RTX PRO 6000 GPUs: **$11.34/hour spot or $22.69/hour on-demand**, but unavailable in the checked capacity snapshot. Rates/availability can change.

For 16 SmolVLA runs at 20k updates plus 16 ACT runs at 100k, illustrative sensitivity scenarios are **$38–99 spot / $76–198 on-demand compute**. With 750 storage units to retain roughly 516 GB of scheduled full checkpoints, **about $43–113 spot / $81–212 on-demand** over those assumed durations. These are not bounds or forecasts: ACT GPU throughput is unmeasured, and setup, retries, tax, transfer and longer storage retention are extra. See BLUE-PLATE-VERDA-COST.md and cost-evidence/ for exact assumptions and API results. No instance was provisioned.

## Where to look

- `experiments.json`, `experiments/`: all 32 experiment specifications.
- `real_corrected/`, `real_markers.npz`: corrected real dataset and X sidecar.
- `sim_corrected/`, `sim_markers.npz`, `sim_candidates/`: candidate simulated dataset and provenance.
- `qa/`: all target annotations, tracker checks and augmentation examples.
- `source/README.md`: reproducible preparation and future launch instructions.
- `validation_summary.json`, `validation/`: concrete test evidence.
- `environment.json`, `requirements-observed.txt`, `base_checkpoint.json`: versions and source/model pins.
- `bundle-checksums.json`: artifact hashes. Verify after transfer.

Use the campaign adapter: vanilla `lerobot-train` against `real_corrected/` alone does not apply X markers, online augmentation, explicit grouped validation or mixed-domain sampling. No Git commits or physical robot actions were performed.
"""
    (root / "PREPARATION-STATUS.md").write_text(text)
    print(
        json.dumps(
            {
                "manifest_files": len(manifest),
                "runs": 32,
                "real_prepared": 16,
                "mixed_candidates": 16,
                "sim_frames": sim["total_frames"],
            }
        )
    )


if __name__ == "__main__":
    main()
