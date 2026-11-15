> Coordinates (2026-11-14): read [NORMALIZED-INTERFACE.md](NORMALIZED-INTERFACE.md). Collection, export and training use station-normalized units through a fixed-scale nominal mapping (signs checked on the station; 3.6-unit gripper gap at grasp). 146 of the 150 radian episodes are converted in `outputs/dataset-150-normalized`. The radian artifacts described below are unchanged.

Offline normalized interface: [usage and limitations](NORMALIZED-INTERFACE.md).

Current moving simulation and training pilot: [TRAINING-STATUS.md](TRAINING-STATUS.md).

# SO-101 simulation and VLA preparation

**Camera-matched station scene:** [reconstruction notes and controls](STATION-RECONSTRUCTION.md).
The operator confirmed larger ~3.2 cm-wide 2×2 blocks; the sorting scene now
uses that scale. The smaller standard LEGO geometry remains only in the legacy
inspection scene. C switches camera views, V restores the reference arrangement.

**Current task: [colored bricks to blue/green plates](SORTING.md).** Run
`uv run python view.py` to see it; R randomizes, B/G changes destination.
Use `sorting-recording-plan.csv` for new demonstrations. The inspection-task
notes below describe the previous experiment; its scene remains available via
`--scene inspection`. Training preparation now defaults to sorting, so the old
inspection recipe needs `--task-mode inspection --manifest recording-plan.csv`.

Isolated MuJoCo workspace for **move the brick to the inspection spot**.
This directory cannot control the physical station: it imports no robot drivers
and opens no serial devices. It does not modify Studio or its Python environment.

## Run

```sh
cd /Users/abdullah/Desktop/junction/junction-intel-robotics/simulation
uv sync --locked --python 3.12
uv run python setup_assets.py
uv run python smoke.py
uv run pytest -q
```

The downloader pins the Menagerie commit and retains its Apache-2.0 license,
README and SHA256 manifest under ignored `assets/robotstudio_so101/`.
`uv.lock` pins the simulator dependencies. The simulation renders two 640×480
views to `outputs/2x4/overhead.png` and `outputs/2x4/wrist.png` and writes `smoke.json`.
The smoke run holds a pose; it does not demonstrate pick-and-place.

Open the interactive native viewer (the wrapper handles macOS's MuJoCo launcher
and the uv-managed Python shared-library lookup):

```sh
uv run python view.py --brick 2x4
```

Use `--brick 2x2` to inspect the smaller brick. For headless Linux rendering,
configure EGL or OSMesa as appropriate for that host. Neither path needs Studio.

## Design and present scope

Use the actual SO-101 MJCF model, a table, a free brick, and a fixed inspection
region. Physics runs at 600 Hz and actions/observations at 30 Hz. Seeded resets
vary pickup position, yaw, color and friction. Absolute-radian actions are
validated against the **simulation model's** joint ranges. Success requires
the brick to have been lifted above 5 cm, then rest inside the target, with no
arm contact, for 0.5 seconds.
Episodes end after success or 30 seconds.

This is an initial task scaffold, not a measured digital twin. The target and
camera transforms are provisional scene geometry. Object dimensions now use
nominal full-height LEGO System 2x2 and 2x4 bricks from `bricks.json`. The simulator
does not translate or replay real normalized actions. `station.json` records
the exact real joint order and camera names, and explicitly leaves the
real-to-sim mapping unset. Never use model ranges as motor calibration.

MuJoCo was chosen because the robot description exists and the task can run
locally. An Isaac-based workflow would add a second simulator stack; the current
task does not yet justify that. SmolVLA is the first VLA; retain ACT as a baseline.

## Record 50 successful real demonstrations

See [RECORDING-50.md](RECORDING-50.md). Fill `recording-plan.csv` with the actual
episode IDs and accept only reviewed successful episodes. The initial rows are
recording slots, not claims that any demonstrations exist.

Audit a finalized local dataset without changing it:

```sh
uv run python audit_dataset.py /absolute/path/to/dataset
```

This checks metadata, joint order, finite values, normalized ranges, episode
counts, frame indices, timestamps, task labels and video file presence.
It does **not** decode videos or certify camera synchronization or physical
success. Review video playback separately. Audit output stays in `outputs/`.

The seven-episode backup passes these structural checks, with a data-count
warning. Do not replace its metadata or mix fake episodes into it.

## Prepare the real-data SmolVLA baseline

After reviewing 50 episodes (40 train / 10 validation):

```sh
uv run python prepare_training.py \
  --dataset /absolute/path/to/finalized/dataset \
  --manifest recording-plan.csv
```

This writes JSON argv plus a shell-quoted command; it starts no job and uploads
nothing. It refuses missing, duplicate or insufficient approved episodes and
data outside the documented ranges. The command uses `lerobot/smolvla_base`,
camera1=overhead and camera2=arm, batch 8, 20,000 steps, checkpointing every 2,000,
CUDA, no Hub publishing, and only the training episode IDs. These hyperparameters
are starting settings, not validated optimal values.

The intended separate CUDA environment is Python 3.12 with
`lerobot[smolvla]==0.7.0`; the CLI fields were checked against its released wheel.
The full VLA dependencies/model have not been installed or trained here. The
generated command is for a standalone LeRobot trainer, **not** Studio's job API.
For a remote host, transfer the reviewed dataset privately and change
`--dataset.root` to its remote location. Preserve the plan and split with results.

Studio also has an existing Modal target; see [MODAL-SETUP.md](../docs/MODAL-SETUP.md).
Check that target's readiness and use its own supported configuration if training
through Studio. Do not create a duplicate target or treat this command as a
verified Studio checkpoint/export integration.

The validation IDs are saved but validation is not automatically run by the
generated command (`eval_freq=0`). A held-out validation evaluator and physical
success trials are required before selecting/deploying a checkpoint. Do not
mistake lower training loss for successful manipulation.

## Next implementation gates

1. Measure brick, target and camera poses; verify inferred camera identities.
2. Fit/verify joint signs and zero offsets from known real poses and the existing
   calibration, without rewriting calibration or limits. Validate replay in sim.
3. Build and validate a scripted/IK grasp-and-place expert. No successful expert
   or synthetic training episodes have been produced yet.
4. Export only successful simulated episodes with provenance and the verified
   action convention. Add explicit domain sampling before mixing sim and real.
5. Compare real-only, sim-only, and sim-then-real checkpoints using held-out
   placements, then supervised physical trials through the existing controller.

Public references: [MuJoCo](https://mujoco.readthedocs.io/en/stable/overview.html),
[Menagerie SO-101](https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9/robotstudio_so101),
[SmolVLA](https://huggingface.co/docs/lerobot/smolvla).

## LEGO objects

Use `uv run python smoke.py --brick 2x2` or `--brick 2x4`; outputs are separated
by variant. Body sizes are 15.8 × 15.8 × 9.6 mm and 31.8 × 15.8 × 9.6 mm.
Studs are 4.8 mm diameter, 8 mm apart, and 1.8 mm tall (11.4 mm total height).
The 2x2 footprint is derived from the same stud grid. Source: [Bartneck drawing](https://www.bartneck.de/wp-content/uploads/2019/04/lego-2x4-brick-dimensions-measurements-3001.pdf).
These are nominal dimensions, not measurements of the station pieces. The solid
collision body omits underside cavities and clutch engagement. Approximate
masses (1.2 g / 2.4 g) and friction remain simulation assumptions.
