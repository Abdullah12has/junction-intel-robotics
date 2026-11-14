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
| Backup/Git | Local source/evidence edits exist; current dataset and model still need a verified off-laptop backup. Git push status must be checked against remote. |
| Power | No verified watts measurement; previous powercap search inconclusive. |
| Submission | Mentor answers, final rubric, video/deck and final evidence tables remain unverified. |

See ../docs/GPU-TRAINING-RECOVERY.md and JSON evidence for exact job IDs and tests. A passed execution smoke test is not a passed task-success gate.
