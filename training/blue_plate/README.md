# Blue-plate dataset and experiment preparation

This folder contains the campaign implementation. Large datasets and private workspace imagery live outside Git. Supply their location using `--root`; local station paths in dated operations notes are examples, not portable defaults.

Run with the workspace `.venv-vla/bin/python` for data preparation and validation. Simulation collection uses `simulation/.venv/bin/python`. No script in this folder controls real robot hardware or provisions cloud resources.

## Contents and behavior

- `prepare.py`: independent corrected LeRobot v3 copy, train-only normalization, fixed grouped splits, episode/target annotations and initial target previews. No idle trimming or original edits. All original state/action/video bytes remain unchanged.
- `track.py`, `variants.py`: causal initial-target marker. A human-annotated initial point initializes a fixed-template local tracker. X disappears at the first poor appearance match or displacement over eight source pixels; it does not reappear until a new selection. This is deliberately an approach-selection cue, not claimed continuous object tracking through grasp/transport. Runtime must call the same tracker from a user click.
- `runtime.py`: online augmentations and reproducible domain→episode→frame sampling. Transform RNG travels with each sample index, so worker scheduling does not change augmentation. Action chunks and tail padding remain unchanged.
- `matrix.py`: 32 specifications, with relative paths under the campaign root. Four augmentation recipes are cumulative. There are no 32 duplicated video datasets.
- `collect_sim.py`: physical MuJoCo expert candidates with overhead + mounted-wrist streams, blue destination only. No-render preflight skips failures; accepted attempts are rerun with RGB recording and checked again. Failed attempts remain logged. Uses current simulator model, so source hashes and known transfer limitations are preserved.
- `export_sim.py`: accepted-candidate v3 export, scene-seed split, real-only normalization, causal marker sidecar. Original H.264 streams are copied, not re-encoded. This export does not certify camera/contact matching to reality.
- `run.py`: actual native LeRobot trainer adapter. Dry-run default. `--check-batch` verifies batch loading; `--train` explicitly launches on an assigned CUDA GPU and enforces readiness checks. SmolVLA uses a pinned base checkpoint and explicit canonical overhead/wrist feature names; ACT learns with actual dataset camera keys. Validation is 512 fixed evenly spaced real validation frames, never simulated data. `dataset.eval_split=.15` satisfies the native config validator only; the factory is replaced with the explicit grouped split, so no second random split occurs.
- `launch.py`: one independent job per listed GPU, queue otherwise. Dry-run default; no cloud API. Failed readiness checks produce failed jobs, not silent synthetic-to-real fallback.
- `validate.py`: hashes, splits, statistics, all config schemas and real/synthetic batch paths. `test_variants.py` checks sampler proportions/resume, deterministic transforms and marker loss. `smoke_model.py` checks actual policy forward/backward with zero optimizer updates. `test_training_integration.py` exercises native ACT checkpoint save/load and optimizer resume with two disposable CPU updates, removed afterward.

## Reproduce preparation

From the repository root, using the campaign-compatible Python environment (replace the data paths):

```bash
python training/blue_plate/prepare.py \
  --source /data/blue-plate-original \
  --output /data/blue-plate
python training/blue_plate/track.py \
  --root /data/blue-plate
python training/blue_plate/matrix.py \
  --root /data/blue-plate
python training/blue_plate/validate.py \
  --root /data/blue-plate
```

Initial target annotations were reviewed on all 64 first frames. Episode outcomes were reviewed by sampled video frames, not exhaustive frame-level certification. All complete demonstrations are retained, including recording lead-ins. Reset families are conservatively grouped; 41/12/11 episodes are train/validation/test. The original backup is preserved, and the unused fixed-close videos remain in the copied dataset for provenance but are omitted from the model feature schema.

`base_checkpoint.json` pins SmolVLA revision; `base_config/config.json` stores its configuration. `environment.json` pins critical LeRobot source hashes in addition to package versions. Copy the campaign and this source folder to a CUDA host; pass the new absolute `--root`. `source/` inside the campaign is a snapshot of these scripts. The external original-source hash check requires the original Mac path; other dataset checks are portable.

## Launch later on a provisioned host

```bash
python training/blue_plate/run.py --root /data/blue-plate --experiment act-N-A0-R-seed0 --check-batch
python training/blue_plate/launch.py --root /data/blue-plate --devices 0,1,2,3,4,5,6,7
# Explicit execution, only after readiness and capacity checks:
python training/blue_plate/launch.py --root /data/blue-plate --devices 0,1,2,3,4,5,6,7 --execute
```

The first launch should profile the exact batch sizes, video-decoding load, memory and checkpoint overhead on the selected GPU. The local CPU checks establish functionality, not CUDA throughput. Avoid putting eight training processes on a single GPU by treating logical device names as separate physical GPUs.

Full native checkpoints include optimizer/RNG state. Resume with `run.py ... --train --resume /data/blue-plate/runs/RUN/checkpoints/STEP`; unchanged spec, code and data contracts are required. Local ACT and SmolVLA optimizer resume and exact sampler replay were tested; CUDA/spot recovery remains untested. The queue does not recreate an evicted VM or automatically retry failed jobs. Backups and explicit recovery are still needed.

Read `readiness.json` and `PREPARATION-STATUS.md` in the campaign directory before any paid launch. Synthetic data remains a separate experimental branch while its wrist camera/contact mismatch is unresolved. Do not turn readiness booleans on merely to bypass a failing check.
