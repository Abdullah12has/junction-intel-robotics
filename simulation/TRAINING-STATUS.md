> Coordinates (2026-11-14): read [NORMALIZED-INTERFACE.md](NORMALIZED-INTERFACE.md). Collection, export and training use station-normalized units through a fixed-scale nominal mapping (signs checked on the station; 3.6-unit gripper gap at grasp). 146 of the 150 radian episodes are converted in `outputs/dataset-150-normalized`. The radian artifacts described below are unchanged.

# Simulation pick-and-place pilot

The MuJoCo expert controls the six SO101 actuator targets. Objects move through
contacts and gravity; the expert never teleports or attaches a held block.
Accepted recordings require a lift with both jaws contacting the requested
block, release, and a stable placement in the instructed bowl.

## View movement

```sh
uv run python run_demo.py --replay --output outputs/dataset-150-raw
```

This loops the 150 accepted scripted demonstrations. It is not trained VLA
inference. Press **3** for free orbit, drag to rotate and scroll to zoom.
**1/2** select fixed training cameras. Fixed cameras intentionally do not orbit.

## Data and training

**Completed: 150 verified episodes collected before fine-tuning.**
`outputs/preflight-150.json` identifies 150 physical successes among 562 attempts,
covering 88 preflight scene seeds. The final dataset uses 89 scene seeds after
replacing one failed red-to-blue recording with a successful new scene.
The final `outputs/dataset-150-raw` passed its audit: **150 episodes, 54,474
frames per camera, 25 episodes per color and 75 per destination**.
There are 82 2×2 and 68 2×4 pickups.
The earlier `outputs/random-pilot-01` contains a separate 14-episode pilot. Two 640×480 cameras are recorded
at 30 FPS alongside state, actuator targets, task text and contact traces.
The expert selects successful scenarios; its 150/562 preflight result does not
demonstrate broad grasp reliability. Failed attempts are excluded from training.

`export_lerobot.py` exports accepted episodes only. It holds out entire scene
seeds: 120 training episodes, 30 validation episodes. Normalization statistics
are computed only from the training episodes. `train_simulation.py` rejects
incomplete data, failed grasps, scene leakage and validation-derived statistics. Simulation actions are
absolute radians, not Studio's hardware action convention. These demonstrations
and weights must remain separate from the real inspection dataset and policy.
Hardware action mapping and sim-to-real performance remain unverified.

Training setup: SmolVLA base, two cameras, 2,000 optimizer steps with batch 32,
simulation-only pilot on a dedicated
Verda RTX PRO 6000 spot instance ($1.418/hour compute). Fine-tuning completed all 2,000 steps on 120 episodes / 43,573 frames. The 30 held-out
episodes contain 10,901 frames across 19 scene seeds. Both splits cover all
12 color/destination instructions. Final training loss: **0.021**. Held-out
flow-matching loss: **0.0644069**, measured on 960 sampled frames spanning all
30 validation episodes. This is offline prediction loss, not a closed-loop
pick-and-place success rate. Six held-out learned-policy trials were completed: one verified grasp and
zero verified stable placements. The planned 12-trial evaluation was interrupted
when the user requested real-arm deployment. Hardware deployment remains blocked
by the missing measured simulation-to-real joint mapping.

Local artifacts under `outputs/`:
- `dataset-150-raw/`: original recordings, physical contact traces and verification.
- `dataset-150-lerobot.tar.gz`: complete training-format dataset.
- `smolvla-150-final.tar.gz`: final checkpoint, optimizer state, processors and recipe.
- `train-150.log`, `training-summary.json`, `heldout-evaluation.json`: run evidence.

All three local archives match their remote SHA256 checksums; the checkpoint
archive contains model weights, preprocessing/postprocessing state, training
configuration and optimizer state. See `outputs/backup-checksums.json`.
The dedicated GPU instance was deleted after backup verification. Its recovery
OS disk remains detached; storage charges continue. Compute release is verified
in `outputs/release-verification.json`.

Recovery disk: `2d4e8a6a-85e8-4fa2-a050-b24de1b9eb4e`. The model is under
`/var/lib/physicalai-trainer/junction-simulation-training/smolvla-150/` inside
the retained trainer container volume. The old instance IP is no longer valid.
Recreate compute only for another authorized training/evaluation run.
See [Verda deletion semantics](https://docs.verda.com/cpu-and-gpu-instances/shutdown-hibernate-and-delete/).

## Model assumptions

The simulated base is rotated −90° to align arm reach with the station's forward
direction, preserving its shoulder pivot and original joint limits. The real
calibration is unchanged. Block footprints match the user's approximate 32 mm
2×2 width, but mass, friction, camera extrinsics and bowl dimensions remain
estimates. The second camera is currently a fixed close view.

```sh
uv run python preflight.py --episodes 150 --output outputs/new-preflight.json
uv run python collect.py --schedule outputs/new-preflight.json \
  --episodes 150 --attempts 150 --output outputs/new-random-recordings
uv run python verify_recordings.py outputs/new-random-recordings
uv run pytest -q
```

## Reproducibility correction

The original reset randomized friction on the previous target and retained
model friction across episodes. This was corrected after recording. The
150-episode dataset preserves its original source snapshot, source hashes and
per-episode friction reconstructed from each recorder’s exact reset sequence.
Replay restores those friction values. New recordings save friction directly;
new resets produce the same scene regardless of the previous target.

## Closed-loop evaluation and real-arm request

`outputs/vla-rollouts-01/summary.json` contains all six completed trials;
`attempt-0005/overhead.mp4` shows the orange-to-green grasp and failed stable
placement. Every completed trial ran for 20 simulation seconds. The trained
checkpoint used its default 50-step action chunks with only joint-limit clipping;
no expert actions were substituted. Raw policy outputs, executed commands,
camera videos and contact traces are saved. A seventh interrupted attempt is
not counted. The temporary inference GPU was released and the recovery disk retained.

Read-only station checks found an active Studio session owning both arms and
confirmed normalized hardware commands. `station.json` still has a null
`real_to_sim_mapping`. No hardware commands, session changes, calibration edits
or motor-limit changes were made. Deployment requires measured joint signs and
zero offsets, matching camera setup, and a validated controller conversion.
