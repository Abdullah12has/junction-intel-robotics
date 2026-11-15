# Robot deployment and inspection

These utilities integrate with the real SO-101 station. Read the [runbook](../docs/ROBOT-STATION-RUNBOOK.md) before using any hardware entry point. Existing deployments may use frozen copies outside this repository; copying code here does not update a live station.

| Area | Files | Role |
|---|---|---|
| Cell orchestration | `run_cell.py`, `inspection.py`, `config.example.json` | Earlier inspection workflow and station configuration template |
| Native policy trial | `run_vla_real.py`, `run_vla_real_from_mac.sh` | Supervised real-arm VLA entry points; inspect station-specific configuration first |
| Output validation | `policy_action_bounds.py`, `audit_policy_contract.py` | Small-overshoot projection, rejection and boundary-dwell checks; offline contract audit |
| Visual feedback | `visual_servo.py`, `vision_diagnostics.py`, `capture_servo_calibration.py` | Measured local alignment, diagnostics and calibration collection |
| Export and verification | `bound_*_export.py`, `verify_*_export.py`, `train_export.py` | ACT/SmolVLA export utilities and recorded-input verification |
| Intel measurements | `ov_prove.py`, `check_gpu_math.py`, `power_sampler.py` | OpenVINO execution, arithmetic and power diagnostics |
| Data preparation | `split_dataset.py` | Earlier inspection dataset split |

## Offline tests

From the repository root, use `python scripts/check.py`. Only `test_policy_action_bounds.py` and `test_visual_servo.py` in this directory are included in that suite.

**Physical trial programs:** `test_policy_bounded_export.py`, `test_smolvla_bounded_export.py`, and `test_smolvla_60s.py` are historical supervised deployment tools, not ordinary unit tests. Do not auto-discover or execute them in CI.

The example configuration is for the earlier inspection task and requires local IDs and paths. It is not a ready-to-run blue-plate configuration. The shared bounds helper is not automatically wired into every historical runner; refer to the [action-limit audit](../docs/POLICY-ACTION-LIMIT-AUDIT.md) for the tested deployment and its limits. Normalized joint bounds alone do not establish collision clearance or grasp success.
