# Simulation coordinate correction — 2026-11-14

The user requested station-normalized simulation recordings and a verified
mapping for sim-to-real training. The station was unavailable for measurements.

## Findings

- Original150 accepted simulation episodes, exporter and checkpoint use absolute
  radians. Their metadata said so; they were never validated for hardware.
- The newer normalized wrapper only stretched model ranges. It did not use the
  encoder scale, zero alignment or measured signs. Relative to direct-shaft
  encoder travel, range scaling errors were +7.85% pan, +0.33% shoulder lift,
  −0.03% elbow, −8.36% wrist flex, −4.81% wrist roll, −18.25% gripper. These
  percentages compare magnitudes; they do not establish sign or zero.
- The wrapper rounded tick commands; the installed LeRobot driver truncates.
- Saved captures include provisional base direction, but no measured model-local
  joint angles. Shoulder captures moved the elbow too. Calibration alone cannot
  determine the MuJoCo zero frame or gripper aperture geometry.
- Existing simulation policy had zero successful placements in six closed-loop
  trials. Small offline loss was not evidence of sim or real task success.

## Changes

`simulation/station_coordinates.py` implements the exact normalized/tick boundary,
measured angle mapping and common model/hardware range. `fit_joint_mapping.py`
requires three measured points and an independent held-out point for each joint,
validates scale/residuals and binds the mapping to calibration and model hashes.
Test fixtures are synthetic; no fitted station mapping has been installed.

The ordinary normalized environment refuses missing calibration. The old
approximation requires an explicitly named opt-in and is not used for new
collection. The viewer defaults to radians until calibration is available.
New preflight/collection requires a measured mapping; the expert still controls
MuJoCo in radians, while the recorder converts BOTH observed states and actions
to station-normalized values. Common joint limits constrain expert targets.
Export/training reject legacy radians, mixed mappings and inconsistent evidence.
Replay converts normalized recordings back to native angles. Policy inference
checks its unit/mapping contract before consuming observations or actions.

Old datasets, checkpoints, real recordings and hardware calibration are unchanged.
No robot movement, new cloud instance, recollection or training was initiated.

## Verification and remaining gate

43 tests passed; Python syntax compilation passed. A collection CLI request
without a mapping was rejected before creating output. Evidence:
`evidence/simulation-coordinate-audit.json`.

Next: measure actual joint directions/zero offsets and gripper geometry, then
fill `simulation/joint-reference-template.json`. Follow
[the measured-coordinate runbook](../simulation/NORMALIZED-INTERFACE.md).
Camera registration, dynamics/contact realism, held-out closed-loop success
and supervised hardware validation remain separate requirements. Coordinates
are not fully established yet; do not claim that this software work guarantees
generalization or that new normalized data already exists.
