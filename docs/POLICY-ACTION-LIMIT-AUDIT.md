# Policy action limit investigation — 15 November 2026

## Findings

Audited ACT N/A0/R step 20,000 and SmolVLA N/A0/R step 2,000. Joint order is shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper. State and action mean/std values in both saved processors match the train-only dataset statistics. RGB inputs are scaled once to [0,1], as in the native trainer; ACT retains its saved visual normalization. Both camera roles are explicitly mapped.

The action targets are within calibrated normalized ranges, but 33.23% of train shoulder targets and 23.80% of elbow targets have absolute value above 95. Targets reach shoulder -100 and elbow +100. Both model architectures have unbounded outputs. Affine mean/std unnormalization does not enforce calibrated limits. A small normalized prediction error can therefore exceed physical-unit bounds.

ACT's rejected first-10-action shoulder minimum was -101.654739 on Intel XPU. Re-running the same recorded observation with training-version PyTorch 2.11 on CPU produced -101.654755. Manual unnormalization matches the saved processor. This particular violation is model extrapolation, not an Intel-specific numerical or normalization conversion defect. The replacement wrist camera remains a possible distribution shift despite the operator confirming the same mounting/framing; its effect on task behavior has not been measured.

## Deployment correction

Shared module: `cell/policy_action_bounds.py`. It projects small physical-unit prediction overshoots to interior command bounds: body [-99,99], gripper [1,99]. These are conservative software command bounds, not replacement motor calibration. Predictions more than five normalized units outside the actual calibrated normalized range, malformed/nonfinite values, or measured out-of-range state are rejected. Requests persistently directed into a boundary while that joint is near it stop after two seconds. Inward recovery requests reset the dwell monitor. The existing speed, tracking-error, camera freshness, STOP-file and trial-duration guards remain in place; their measurement check is stricter than the shared helper.

This is a correction to deployment command handling, **not a correction to learned policy behavior or proof of safe collision-free motion**. SmolVLA already had simple clipping; simple clipping alone did not prevent its unsuccessful limit-seeking behavior. The dwell stop is consequently essential.

## Verification

Five tests passed both on Mac and in the Intel motion environment: recorded ACT overshoot correction, invalid/large-output rejection, boundary dwell stop, inward recovery, and unchanged calibrated measurement-range enforcement. Recorded SmolVLA replay would stop on elbow boundary persistence at 9.11 seconds. This is sparse offline replay, not a new hardware test. ACT's valid components remain unchanged; its rejected shoulder requests become -99. No model weights or running training jobs changed.

Evidence: `evidence/action-limit-audit/data-and-statistics.json`, `act-cpu-parity.json`, `bounds-replay.json`.

## Intel deployment and state

Guarded entry points are now:

- `/home/intel/junction-cell/act-checkpoint-test/run_bounded_policy.py`
- `/home/intel/junction-cell/blue-plate-checkpoint-test/run_bounded_policy.py`

Each imports the adjacent `policy_action_bounds.py`; original runners and weights are preserved. Use the corresponding checkpoint/root, Intel backend Python for policy inference, device `xpu`, and motion-env Python for control. Do not invoke the earlier ACT runner expecting the new projection behavior. STOP files remain set. No physical test was started as part of this investigation. Before a newly authorized test, recheck live ownership, calibration, pose, and operator readiness; clear only the selected root's STOP interlock for that test. Do not change motor ranges to match network outputs.

The latest supervised SmolVLA trial from an interior starting pose was stopped manually at 26.79 seconds because it returned to the joint boundaries; hold confirmed. The ACT trial on Intel was rejected before any policy actions because of the raw shoulder overshoot; hold confirmed. Neither establishes blue-plate task success.
