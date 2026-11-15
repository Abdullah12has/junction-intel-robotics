# Code map

The repository keeps runtime paths stable so existing training snapshots and station scripts remain identifiable. Entry points are grouped by responsibility:

| Directory | Entry points | Supporting code | Setup |
|---|---|---|---|
| `dashboard/` | npm `start`, `build`, `type-check` | React/TypeScript UI and generated API schema | [README](../dashboard/README.md) |
| `cell/` | `run_cell.py`, `run_vla_real.py` | Export, inspection, policy bounds, visual servo and diagnostics | [README](../cell/README.md) |
| `training/blue_plate/` | `prepare.py`, `matrix.py`, `validate.py`, `run.py`, `launch.py` | Tracking, transforms, sampling and simulation conversion | [README](../training/blue_plate/README.md) |
| `training/orchestration/` | `shard_queue.py` | Host-specific queue management | [Training overview](../training/README.md) |
| `simulation/` | `view.py`, `collect.py`, `export_lerobot.py`, `rollout_policy.py` | Scene, expert, camera fitting and normalized coordinates | [README](../simulation/README.md) |
| `backend-patches/` | `manual-jog.patch` | Upstream backend changes and their tests | [README](../backend-patches/README.md) |
| `intel-station/` | Captured Intel application and deployment entry points | Separate source snapshots with hash manifest | [README](../intel-station/README.md) |
| `scripts/` | `check.py` | Explicit hardware-free verification | [Root setup](../README.md) |

## Source versus artifacts

- `evidence/` holds dated observations and measurements. It is not executable configuration or a live status API.
- `docs/` holds submission, research and operations material. Use the submission overview for current claims; earlier plans describe intended work.
- `simulation/assets/`, `simulation/outputs/`, local environments, recordings and checkpoints are ignored or stored outside the repository.
- `vendor/physical-ai-studio/` retains upstream history and licensing; local integration is captured in `backend-patches/`.
- Deployment and training hosts use frozen copies. Changing this checkout does not update those hosts.

## Verification boundaries

`python scripts/check.py` syntax-checks first-party Python and runs named, offline tests only. Full dashboard builds, native LeRobot checkpoint integration, simulation rendering and physical trials require their respective environments and are separate checks. In particular, historical `cell/test_*.py` trial runners must not be included in automatic test discovery.
