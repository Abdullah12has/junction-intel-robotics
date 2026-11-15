# Reconciliation of external audit — 14 November 2026

The audit correctly identifies major unfinished work, but its training, NPU and robot-test statements are stale.

1. VLA gap remains: ACT is not a language-conditioned VLA. Validate SmolVLA or cloud training; Modal account authentication requires the account owner. Confirm requirements against the final challenge rubric.
2. Training/export/finite-output checks and real-arm tests happened. Full pick-and-place success has not happened. Do not equate these gates.
3. Dataset coverage remains7episodes/one task versus the plan's120/three tasks. Duration17–23s is observed; duration alone does not establish that the learned policy is necessarily slow. The first robot test deliberately used10Hz; later tests used recorded30Hz.
4. Real detector remains unverified. Source CAM/ROI values are not verified calibration. A camera image cannot establish its mounting quality.
5. Mentor answers remain unrecorded in the checked evidence.
6. Bounded live policy tests are separate from run_cell.py; the orchestrator and split utility remain unvalidated.
7. Source edits and evidence exist on the Mac; this is not a backup of the155MB dataset and987MB model directory. Those require an explicit verified copy. Remote Git tip was not freshly checked in this audit.
8. STATUS.md is now rewritten with current facts; old contradictory content preserved separately. Raw training and robot JSON evidence exists. Submission-ready summaries remain incomplete.
9. Watts remain unconfirmed; an empty powercap search is not evidence that RAPL is unavailable or permission was fixed.
10. Submission assets remain unverified, not assumed complete.

Priority: preserve artifacts; secure a VLA path; capture missing task demonstrations and real inspection data; validate policy success and detector accuracy; then integrate and capture submission evidence. Do not widen motor limits to compensate for a weak policy.
