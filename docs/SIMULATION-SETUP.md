# MuJoCo / SmolVLA setup — 14 November 2026

## Current status

150 successful simulated pick-and-place episodes have been recorded and audited
(54,474 frames per camera; 120 train / 30 held out by scene). SmolVLA fine-tuning
completed 2,000 steps on Verda using the existing LeRobot 0.6.0 trainer.
Held-out offline loss is 0.0644; closed-loop policy success remains untested. See
[simulation/TRAINING-STATUS.md](../simulation/TRAINING-STATUS.md) for current
artifacts, the moving viewer and limitations. This simulation workflow does not
control the physical station.

## Initial setup notes (superseded where noted above)

The user chose MuJoCo and can record 50 real demonstrations. The isolated
workspace is [simulation/](../simulation/README.md) on the Mac. It uses Python
3.12, uv.lock, MuJoCo 4.0.1 and a pinned SO-101 Menagerie snapshot. No Intel
packages, live robot sessions, serial devices or motor calibration were changed.

Implemented: seeded brick-placement scene, 600 Hz physics / 30 Hz control,
two rendered cameras, strict simulation-action checks, lift/place/release
success predicate, timeout, dataset structural audit, 50-slot recording CSV,
and a command generator for a real-only SmolVLA baseline with 40/10 split.

Verified locally: simulation loads/renders and holds a pose for 90 steps;
seven-episode backup passes structure/value/timestamp checks (4,366 frames).
Tests cover deterministic reset, control timing, invalid actions, termination,
placement success conditions, and reviewed non-overlapping dataset splits.
Screenshots and machine-readable reports are under ignored simulation/outputs/.

Not established: measured camera/object geometry, real normalized-unit to MJCF
radian conversion, successful scripted expert, synthetic demonstration export,
VLA training/inference, automated held-out validation or hardware policy transfer.
Do not call the scene a calibrated digital twin or train on its radian actions
as though they were the station's normalized actions. station.json leaves that
mapping unset deliberately. No training job or dataset upload was started.

The standalone training recipe targets lerobot[smolvla]==0.7.0 in a separate CUDA
environment; its CLI fields were inspected in the released wheel, but the full
VLA environment/model is not installed here. Studio's separately configured
Modal trainer is documented in MODAL-SETUP.md and must be checked for readiness
before use; the standalone CLI is not a verified Studio import/export workflow.

Next: operator follows [recording instructions](../simulation/RECORDING-50.md),
reviews 50 successful episodes, records real object/target dimensions, and verifies
camera views. Then fit the action mapping, implement a successful simulated
expert, and compare real-only against sim-then-real training.
