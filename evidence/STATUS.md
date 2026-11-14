# Verified station status

Updated 14 November 2026, Helsinki. Historical statements are preserved in STATUS-HISTORICAL.md and superseded here.

| Area | Current evidence |
|---|---|
| Dataset | 7 episodes, 4,366 frames, two cameras, one inspection task. No bowl-task demonstrations verified. |
| ACT training | Completed one full GPU epoch, 545 steps. Model fa42ee76-bcfb-43ba-ba21-124e678ee785. |
| OpenVINO | Export exists; CPU/GPU inference on recorded inputs returned finite100×6 actions. |
| Robot trials | Actual model-driven arm motion tested. Corrected export completed30seconds. Pick-and-place success NOT established. |
| Limits | First15-unit test guard was restrictive. Longer trial exposed negative gripper targets; canonical export now bounds outputs to valid normalized joint limits. |
| NPU | System driver installed; synthetic PatchCore smoke inference verified on NPU. |
| VLA | ACT is not a VLA. Local SmolVLA training untested; no Modal login file observed at default laptop location. Authentication elsewhere is not ruled out. |
| Detector | Synthetic pipeline only. Real camera/ROI, defect data, split and accuracy unverified. |
| Orchestrator | Standalone bounded policy tests ran; run_cell.py and split_dataset.py remain unvalidated end to end. |
| Backup/Git | Dataset/model/evidence backed up to Mac station-backups/2026-11-14:239files,1.21GB,all SHA256 hashes match. Docs committed locally; Git push blocked by missing credentials and app write permission. |
| Power | No verified watts measurement; previous powercap search inconclusive. |
| Submission | Mentor answers, final rubric, video/deck and final evidence tables remain unverified. |

See ../docs/GPU-TRAINING-RECOVERY.md and JSON evidence for exact job IDs and tests. A passed execution smoke test is not a passed task-success gate.

## Verda SmolVLA pilot completed — 2026-11-14

User authorized the seven-episode inspection pilot. Trained 10 epochs/2320 steps on RTX Pro 6000 96GB, final validation loss 0.27096. Model `6540166d-c6c7-4087-ba07-ec5571c37d3b` has an OpenVINO export. Three offline observations passed finite 50x6 output checks on laptop CPU (~1.5s each). No robot trial performed. Checkpoint/export backed up to Mac; all eight files match SHA256. Compute released, disk retained. See `docs/VERDA-TRAINING.md`.

### SmolVLA physical trial result

20-second monitored GPU trial completed and stopped; no active sessions afterward. Operator reports missed/dropped brick: task success false. Raw predictions exceeded calibrated range, so trial used constrained export with original preserved. See VERDA-TRAINING.md and smolvla-robot-test.json.
