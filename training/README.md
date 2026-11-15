# Training

The current implementation is [blue_plate/](blue_plate/README.md). It prepares immutable-source-derived data and runs native LeRobot ACT/SmolVLA training. [orchestration/shard_queue.py](orchestration/shard_queue.py) divides an experiment queue across provisioned hosts; it does not provision GPUs.

## Pipeline

1. `blue_plate/prepare.py`: corrected task text, grouped train/validation/test split and train-only normalization.
2. `blue_plate/track.py`: causal target-X sidecars initialized from reviewed annotations.
3. `blue_plate/matrix.py`: 32 experiment specifications without duplicating videos.
4. `blue_plate/validate.py`: source, split, statistics and readiness checks.
5. `blue_plate/run.py`: dry run, batch validation, explicit training and checkpoint resume.
6. `blue_plate/launch.py`: queue on available GPUs; `orchestration/shard_queue.py` supports host-specific queues.
7. `blue_plate/finalize.py`: campaign finalization checks.

For simulation data, collection and conversion are separate (`collect_sim.py`, `export_sim.py`). A converted dataset is not proof of transfer quality: mixed-data runs remain gated until camera/contact mismatch is resolved.

## Reproducibility

Supply the campaign directory explicitly with `--root`. It contains data, split manifests, model revision, environment/source hashes, readiness and run artifacts. The original recording source and GPU environments are external to Git. Setup details and examples are in the component README; root `requirements-checks.txt` installs only offline-test dependencies, not a supported training environment.

A running campaign uses frozen source snapshots and validates them on resume. Repository cleanup must not be copied over active training code. Preserve the campaign's recorded environment and source contract when resuming.

## Evaluation

64 demonstrations are split into 41 training, 12 validation and 11 test episodes by reset family. Augmentation is applied online to training samples. Validation uses fixed real-data frames; evaluate real task success separately from action loss. The X-conditioned model requires the same causal target selection at inference. See [submission results and limitations](../docs/SUBMISSION.md).
