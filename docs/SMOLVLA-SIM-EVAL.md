# Run the co-trained SmolVLA in MuJoCo

15 November 2026. Simulation only: no robot, Studio, serial device or cloud GPU is needed.

## Run on this Mac

Existing environments: `simulation/.venv` runs MuJoCo; `/Users/abdullah/Desktop/junction/.venv-vla` runs the policy on Apple MPS. Do not reinstall or resync them for this command.

```bash
cd /Users/abdullah/Desktop/junction/junction-intel-robotics/simulation
MODEL=/Users/abdullah/Desktop/junction/station-backups/2026-11-15-smolvla-cotrain/run-01-step015000
DYLD_LIBRARY_PATH=/Users/abdullah/.local/share/uv/python/cpython-3.12.12-macos-aarch64-none/lib \
  .venv/bin/mjpython rollout_policy.py \
  --local /Users/abdullah/Desktop/junction/.venv-vla/bin/python "$MODEL/contract.json" \
  --checkpoint "$MODEL/pretrained_model" \
  --normalized --raw outputs/dataset-v5-raw \
  --fresh-seed 20000 --horizon 900 \
  --output outputs/eval-cotrain-04 --loop-results
```

Use a **new output directory** each run; the script refuses an existing one. The latest run used `outputs/eval-cotrain-03`; don't delete it. The Python library path fixes the Mac viewer's `libpython` launch error. `--local` currently hardcodes MPS in `PolicyClient`; do not use this command on Intel/Linux without making device selection configurable.

## What it does

- Loads the 15,000-step co-trained SmolVLA checkpoint; its file checksums were verified against `final.sha256`.
- Checks the checkpoint's normalized action units and mapping hash against `StationCoordinates` before execution. Never substitute calibration to bypass this check.
- Tests 12 colour/destination tasks at seeds 20000–20011, up to 900 steps each (30 simulated seconds).
- Supplies two camera images: `camera1 = overhead`, `camera2 = wrist`. Here `wrist` means the **fixed close-view camera**, not the third gripper-mounted camera.
- Requests 50-action chunks and executes them closed-loop. Inference is synchronous; wall-clock time exceeds simulated time. Recent MPS chunks took about 0.3 seconds.
- Shows live policy actions, then loops recordings if `--loop-results` is present. Close the viewer to stop. Keys: 1 overhead, 2 close, 3 orbit. Avoid reset keys, joint sliders or dragging the arm during evaluation.

## Fixes retained in the working tree

- Claude added checkpoint-contract loading and local-worker support in `policy_worker.py` / `rollout_policy.py`.
- Converted radian commands are clipped to existing model limits to avoid float rounding just outside a limit.
- Added a two-second viewer synchronization period **before** the first trial reset. The earlier GUI run showed an abrupt near-zero state reset not reproduced by replaying its saved actions without the viewer. With the synchronization period, subsequent trials have completed without that failure; exact GUI reset cause is not fully established.
- An episode error is recorded as a failed trial rather than aborting every remaining scene. Encoder range checks, joint limits and grasp/success criteria remain active.
- Summary protocol text now reflects the actual horizon and fresh-seed mode.

These changes are uncommitted. Preserve other agents' uncommitted calibration/scene work.

## Results and limits

Latest run stopped when the viewer closed during trial 7: six full 900-step trials plus one interrupted trial, zero accepted placements. `complete` is false. This is not a completed 12-scene result. No policy worker was left running by the evaluator.

Read `outputs/eval-cotrain-03/summary.json`: `complete`, `trials_completed`, `successes`, and per-trial `accepted`, `physical_grasp_verified`, `max_target_height`, `final_target_xy_distance_to_plate`, `clipped_action_frames`, `error`, `inference_seconds`.

Each `attempt-NNNN/` contains camera MP4s, `trajectory.npz`, raw policy actions, a trace and metadata. `schedule.json`, inference logs and `evaluation_provenance.json` are at the run root. Early runs `eval-cotrain-01` and `-02` were interrupted and are not complete evaluations.

This schedule tests single-block instructions, **not all blocks to blue**. Fresh scenes may pair square bricks with green or rectangles with blue, unlike the synthetic training rule; distinguish this distribution shift when interpreting results. A completed trial is not a successful grasp. Use the recorded success fields and video evidence; do not weaken criteria to make the model pass.
