# Colored bricks to two plates

**Updated from actual camera views:** see [STATION-RECONSTRUCTION.md](STATION-RECONSTRUCTION.md).
The station uses approximately 3.2 cm-wide 2×2 blocks, a black mat and rimmed
bowls. This supersedes the small LEGO/flat-plate geometry below. The default
viewer now shows the reference arrangement; R enables randomized layouts,
V restores the reference, and C switches between both camera views/free orbit.

This is the current task, replacing the single inspection-target experiment.

```sh
cd /Users/abdullah/Desktop/junction/junction-intel-robotics/simulation
uv run python view.py --scene sorting --seed 42
```

The native viewer opens with 4–6 separate DUPLO-style bricks, with a random
subset of red, yellow, orange, purple, white and cyan. Each color occurs once
per scene, so instructions identify one object unambiguously. Colors are
shuffled between the 2x2 and 2x4 geometries. Position, yaw, target object and
destination vary by seed. Initial footprints are separated and off the plates.

- **Blue plate: left. Green plate: right**, viewed from the front/overhead camera.
- **R:** new layout and instruction.
- **B / G:** same seeded layout, instruction targeting blue / green.
- Drag to rotate and scroll to zoom. The instruction appears inside the viewer,
  is printed in the launcher terminal, and is returned as `observation['task']`.

The bowls have rim collisions and provisional dimensions; see the reconstruction
notes for the current geometry. The passive `view.py` holds the arm. To see
physical pick-and-place movement, use `run_demo.py` as described in
[TRAINING-STATUS.md](TRAINING-STATUS.md).

## Task and success

Example: **move the red brick to the blue plate**. Each episode requests one
pick-and-place. Either destination is valid depending on the instruction; there
is no fixed color-to-plate assignment. A longer sequence can chain these tasks.

Success requires the requested brick to have lifted above 5 cm, then rest near
the requested plate center at its surface height for 0.5 seconds, with no robot
or other-brick contact. The other plate, wrong brick, or pushing without a lift
does not count. The center tolerance is deliberately conservative (±15 mm).

## Real recordings and training

Use **sorting-recording-plan.csv**, not the old inspection recording sheet.
It schedules 50 episodes with 25 per plate, both brick sizes and all six colors,
40 train / 10 validation. The first 48 slots cover each color × size × plate
combination twice; the last two add another red 2x2 example for each plate.
These 50 are a starting dataset for a more varied task, not a guarantee of
generalization. Collect more examples of combinations that fail in evaluation.

Place 4–6 distinct-colored distractors in view for each demonstration. Vary
positions so the policy cannot memorize a location. Each episode's recorded
task text must match its CSV instruction; retain the exact task text in the
dataset. For each color and size demonstrate **both** plates. Do not assign
each color permanently to one plate. Review successful grasp, lift, placement,
release and retreat before marking a slot accepted. Keep camera setup and
30 FPS recording requirements from RECORDING-50.md. Hold out precise layouts.

```sh
uv run python audit_dataset.py /path/to/sorting-dataset --task-mode sorting
uv run python prepare_training.py --dataset /path/to/sorting-dataset \
  --task-mode sorting --manifest sorting-recording-plan.csv
```

The preparation tool checks accepted manifest instructions against actual
episode labels. It rejects the old inspection-task dataset for sorting training.
Those seven episodes remain intact and can still be audited in inspection mode.
The old recipe requires `--task-mode inspection --manifest recording-plan.csv`.

A contact-based expert has generated 150 accepted randomized sorting
demonstrations, followed by a 2,000-step SmolVLA fine-tune. See [TRAINING-STATUS.md](TRAINING-STATUS.md) for the pilot,
limitations and moving viewer. Hardware deployment remains unverified.
