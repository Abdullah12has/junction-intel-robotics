# Junction Intel Robotics

A robot learning workbench for the Junction 2026 Intel Physical AI challenge: collect SO-101 demonstrations, train ACT and SmolVLA, inspect policy outputs, and evaluate on an Intel robot station.

The current task is **pick up a block and place it on the blue plate**. Our contribution combines a customized Physical AI Studio dashboard, a reproducible 32-experiment training pipeline, MuJoCo data generation, and guarded deployment utilities. Earlier inspection and anomaly-detection experiments remain available as supporting work.

**Submission:** [architecture and results](docs/SUBMISSION.md) · [code map](docs/CODE-MAP.md) · [evidence](evidence/README.md)

## What is implemented

| Component | Source | Purpose |
|---|---|---|
| Station dashboard | [dashboard/](dashboard/README.md) | Camera previews, demonstrations, training and robot session controls |
| Robot deployment | [cell/](cell/README.md) | Intel/OpenVINO utilities, policy output checks, bounded control and inspection |
| Training campaign | [training/](training/README.md) | Corrected labels, grouped splits, target markers, augmentation and ACT/SmolVLA jobs |
| Simulation | [simulation/](simulation/README.md) | SO-101 MuJoCo scene, expert collection, conversion and policy evaluation |
| Intel station source | [intel-station/](intel-station/README.md) | Captured quality-sorting app, deployment workers, diagnostics and research |
| Backend integration | [backend-patches/](backend-patches/README.md) | Reproducible manual-jog changes against pinned upstream Studio |

The campaign defines **2 policies × 2 target-cue variants × 4 augmentation recipes × 2 data sources = 32 experiments**. Real-data jobs have been launched; synthetic-mixed jobs remain gated on camera/contact validation. Physical motion has been demonstrated, but reliable autonomous pick-and-place success has **not** been established. See the dated evidence rather than treating training loss as task success.

## Get the source

```sh
git clone --recurse-submodules https://github.com/Abdullah12has/junction-intel-robotics.git
cd junction-intel-robotics
```

Physical AI Studio is pinned to `17714488fa9c7d098a056da83f7c220cf983957d`. Use `git submodule update --init --recursive` on an existing clone. Git LFS is needed only to retrieve upstream LFS media. Preserve the recorded submodule pin for reproducible setup.

## Verify without hardware

With Python 3.12:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-checks.txt
.venv/bin/python scripts/check.py
```

This explicitly selected suite checks action bounds, visual-servo logic and dataset transforms/sampling. It does not connect to the robot, launch training or provision GPUs. Some historical `cell/test_*.py` files are physical trial programs; do not run blanket test discovery there.

Each component has its own setup instructions. Training also requires the external campaign data and its recorded LeRobot environment; model weights and demonstration videos are not bundled in Git. Simulation dependencies are locked in `simulation/uv.lock`; the dashboard has its own npm lockfile.

## Operate the station

Read the [station handoff](docs/ROBOT-STATION-HANDOFF.md), [runbook](docs/ROBOT-STATION-RUNBOOK.md), and [dashboard guide](docs/ROBOT-DASHBOARD.md) before connecting. Check live state: device paths, sessions and calibration are station-specific. Only one process may own an arm. Never replace measured calibration with guessed limits.

## Documentation and attribution

[Documentation index](docs/README.md) separates submission material, research and dated operations notes. Third-party source and licenses are retained; see [attribution](THIRD_PARTY_NOTICES.md). Generated datasets, checkpoints, local environments and credentials are excluded from the submission source tree.
