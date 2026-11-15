# Improving the frozen SmolVLA policy

15 November 2026. Implementation and validation record; weights and station calibration remain unchanged.

## Scope and decision

Keep checkpoint `run-01-step015000`. First audit deployment, then compare shorter execution horizons, then add local wrist-camera alignment with measured geometry. Grasp verification is separate: pixel alignment does not establish contact, lift or successful placement. No automatic retry or transport is authorized by an `aligned` result.

The training builder and provenance contain only RealSense 243622061421 (camera1, overhead) and 243722064316 (camera2, fixed close). The third camera listed in the base config was not supplied in this co-training dataset. Do not inject the physical wrist view into the frozen policy as an untested substitute. Use it separately for visual feedback. Real demonstration instruction: `pick up all the blocks and put them on the blue plate`.

## Plan and implemented stages

1. **Deployment controls:** real runner now requires explicit verified camera device paths, checks camera ownership, checks policy coordinate contract, rejects nonfinite chunks, bounds inference waiting and checks STOP while waiting. Existing normalized position limits remain. Rate limiting and tracking-error checks run per action. Device cleanup includes camera startup/arm-claim failures. Camera path role still requires verification against live frames/device metadata; V4L identifiers differ from RealSense SDK serials.
2. **Receding horizon:** real runner and simulation evaluator execute the first 10 actions, then reobserve/reinfer. `--execute-steps 50` restores the baseline. This does not reduce model chunk size or edit weights. Synchronous inference inserts holds; latency must be measured. Do not describe it as uninterrupted 30 Hz inference.
3. **Vision diagnostics:** `cell/vision_diagnostics.py video` writes an annotated video and per-frame detections from a recording. HSV thresholds and ROI are explicit station inputs. No target, multiple targets or boundary-touching targets produce hold diagnostics. No object detector weights are trained.
4. **Measured correction:** `cell/visual_servo.py` fits a local pixel/joint Jacobian from measured perturbations. A bounded damped-inverse controller holds gripper opening, requires fresh images, restricts pose and pixel domain, and requires three aligned frames. It controls local 2-D alignment only; it cannot infer depth or choose a safe approach height.
5. **Calibration and physical acceptance:** pending operator-assisted reference capture, local perturbation measurements and independent validation. `fit` always writes `validated: false`. No guessed calibration template is installed. `replay` only writes proposed commands. The visual controller is deliberately not wired into automatic real-arm transport before this gate.
6. **Grasp verification / hybrid transport:** pending calibrated depth or separately validated tracking of block motion relative to gripper. A visible block or closed gripper alone is not grasp proof. Final system should hold on unverified lift, with operator-supervised retry; the standalone `GraspVerifier` now requires repeated measured 3-D co-motion and revokes permission on a drop/stale evidence. Live depth/target tracking and hybrid transport integration remain pending.

## Files and commands

Use the existing Mac `.venv-vla` for NumPy/OpenCV tests and diagnostics:

```bash
cd /Users/abdullah/Desktop/junction
.venv-vla/bin/python -m unittest discover -s junction-intel-robotics/cell -p test_visual_servo.py -v
.venv-vla/bin/python junction-intel-robotics/cell/vision_diagnostics.py --help
```

Video config JSON keys: `hsv_low`, `hsv_high` (OpenCV hue 0–179, saturation/value 0–255), `roi` as x/y/width/height, optional `min_area`, `max_area`. Hue wrap is supported for red. Choose the target and ROI explicitly, excluding same-color plates and other blocks. Color thresholds alone do not establish object identity through occlusion.

Calibration samples NPZ: `joint_deltas`, `pixel_deltas`, `validation_joint_deltas`, `validation_pixel_deltas`; all relative to the same reference with stationary target. At least twice as many fit samples as controlled joints and four independent validation samples. Metadata JSON keys: `camera_id`, `image_size` [width,height], `reference_state` [six normalized joints], `reference_uv` [grasp-center pixel], `joints` (2–5 arm joint names, no gripper), `pose_radius` (normalized units), `pixel_radius`, `tolerance` (pixels), `mapping_sha256`. Never infer a grasp-center pixel just from the image center.

`fit samples.npz metadata.json draft.json` produces a draft. Review measured geometry, residuals and independently observed correction direction before marking it validated. The loader rejects wrong camera/resolution/mapping, poor residuals and ill-conditioned Jacobians. `replay samples.npz calibration.json proposals.json --camera-id ... --mapping-sha256 ... --image-size 640 480` uses state[N,6], target_uv[N,2], timestamps[N]. It never actuates hardware.

The real runner requires deploying `visual_servo.py` beside it. Its Mac wrapper requires `CAMERA1` and `CAMERA2` environment variables with verified laptop `/dev/v4l/by-id/...` paths. Do not deploy over another agent's changes without comparing. A separate staged copy is on the laptop at `/home/intel/junction-cell/frozen-policy-tools-20261115/`; it has not replaced or started the live policy runner.

## Evaluation

Eight unit tests pass: closed-loop convergence in a known linear camera model, measured Jacobian recovery, rank failure, stale/repeated frames, pose/capture bounds, ambiguous vision, invalid actions/rates, and multi-frame alignment. This is not an SO101 physical success test.

`simulation/compare_execution_horizons.py` runs paired headless learned-policy trials on the same scene and random seed, preserving joint limits, camera mapping and model normalization. It reports success, raw-command clipping, maximum block height and inference latency. The first 150-step smoke comparison had no lift/success for either horizon; clipped frames were 76/150 for 50 actions versus 65/150 for 10. This does not establish improved grasping. A longer paired run is recorded separately.

Physical readiness was confirmed by the operator. A read-only wrist snapshot passed and showed cyan and orange blocks plus open fingers. Reference positioning is pending; no calibration motion or visual correction has been executed in this implementation turn.

### Full paired simulation result

`simulation/outputs/horizon-comparison-02`: same red-to-blue scene (seed 20003), same checkpoint and policy seed, up to 900 steps. Baseline 50 actions: no lift or success, 153 clipped frames. Execute-10: success at step 811, maximum block height 0.0813 m, 231 clipped frames, average inference 0.270 s. Replaying saved actions with unchanged physics also passed the recorder-equivalent bilateral-jaw-contact-above-6-cm and placement checks (`contact-replay.json`). One paired scene is encouraging evidence, not a measured reliability rate or real-arm result. More clipping remains a concern; no limits were relaxed.

### Physical calibration attempt

Operator confirmed centered cyan target and hands clear. `capture_servo_calibration.py` captured the current image and normalized joint state, then refused all perturbations because elbow flex was 99.1833/100 (shoulder lift -97.0900). No calibration perturbations occurred; the driver held the present pose and disconnected with holding torque. Laptop evidence: `/home/intel/junction-cell/servo-measurements-01/measurements.json` and `reference.png`. Operator repositioning away from joint limits is pending. Existing joint calibration and limits were preserved. A single-frame cyan detector check succeeded at approximately (377,369) pixels; this is not a validated correction calibration.

### Second scene and checkpoint audit

Seed 20005 failed under both horizons, with no lift: 486 clipped frames for execute-50 and 346 for execute-10. Across these two red-to-blue scenes, execute-10 passed one and execute-50 passed none; the sample is too small to estimate reliability. Compact evidence is in `evidence/frozen-policy-improvement/`. The checkpoint audit verified all seven original manifest hashes, finite positive normalization scales, six-joint dimensions and the two training camera keys. `camera3` is an unused base-config entry. No weights changed.

Teleoperation was enabled through Studio at operator request to reposition for calibration. Backend session `58e36234f8994acc8adbdcef6ff43dcc` reported teleop and the UI switch was checked; query live state before use. Do not run the calibration script while this session owns the arm or cameras. Stop through Studio first when the operator confirms hands clear.

### Repositioning follow-up

At operator confirmation, teleoperation was switched off and session `58e36234f8994acc8adbdcef6ff43dcc` stopped through the API. Both arms disconnected and all camera publishers released. The second calibration attempt captured `servo-measurements-02/reference.png` but rejected the partially cut-off cyan target before any perturbation. Operator was asked to expose the full cyan block and remove the neighboring orange block. The arm is holding; teleoperation is OFF. Recheck state on continuation.

Third capture: pose now has margin (pan -11.85, lift -14.02, elbow 60.07, flex 50.83, roll -47.61, gripper 27.43). The image still clips the cyan object at bottom/right; no perturbation was issued. Detection now uses the full wrist frame rather than the earlier narrow ROI, preserving image-edge rejection and single-component checks. Operator shown actual frame and asked to move the block upper-left into full view. Evidence on laptop: `servo-measurements-03/`.

### Measured-motion calibration method

After the operator authorized assistant-controlled repositioning, the collector was revised to fit pixel changes against **measured** joint changes. Requests of 0.4 normalized units did not move enough; a 1.0-unit request moved pan about 0.43 units. This is consistent with deadband/tracking residual, not proof of its mechanical cause. An exact commanded-pose arrival is not required to estimate a measured local Jacobian. The collector records target error rather than assuming the target was reached. It rejects actual motion under 0.15 units, wrong direction, excursion beyond 1.5 units of the reference, non-commanded-joint motion over 0.3 units, and return drift over 0.75 units. No motor calibration, torque limits, normalized ranges or speed limits were changed. These are local calibration measurement checks, not evidence of autonomous grasp safety.

### Physical movement result (01:48 Helsinki)

Assistant-controlled small pan perturbations produced measured encoder movement and target pixel displacement. Pan moved from -12.279 to -12.796 (target center x 464.0 → 472.1), then to -11.848 (x 459.1); the gripper stayed at 27.433. The subsequent shoulder-lift request from -13.316 to -14.316 produced no measured movement. The collector stopped and held the measured pose; it did not continue or validate a two-axis fit. Last pose: [-11.935, -13.316, 60.980, 50.487, -47.660, 27.433]. Cause of insufficient small-command response remains unproven (deadband/stiction/driver behavior are hypotheses). No automatic visual correction enabled. Measurements are backed up under `station-backups/2026-11-15-servo-calibration/servo-measurements-06`; compact evidence is `evidence/frozen-policy-improvement/physical-calibration-attempt.json`.

### Persistent control and software correction (15 November)

A separate `app/tools/motion_server.py` process acquired the follower after our calibration attempt and issued substantial pose changes. Our direct diagnostic connection was correctly rejected by the device-ownership guard; no competing connection or motion was attempted. Its read-only state showed shoulder lift tracking within 0.45 normalized units at a different pose, disproving a completely nonresponsive joint but not identifying the small-command issue. That server already preserves a persistent connection and has an optional outer integral correction; we did not modify it or its guards. Exclusive ownership was requested before further physical work. Earlier reference images/poses are now stale.

Fixed a software defect in our new local alignment controller: each command was previously based on measured state, so small corrections could repeatedly remain inside servo deadband. Commands now accumulate from the previous bounded target, preserving it once aligned instead of resetting to sagged measured state. A 1.5-unit command/measurement gap guard stops a stalled joint. Two added tests cover convergence with deadband and bounded failure with a stationary joint; all 10 tests pass. This fix is validated in a numerical plant, not yet on hardware.

### Existing wrist controller discovered

Before any further physical calibration, read `/home/intel/intel-junction-2026/docs/PICK-ANYWHERE.md`. The laptop already has measured `app/tools/wrist_calib.json`, `jaw_calib.json`, `scan_calib.json` and `run/servo_corr.json`, plus `pick_any.py` and `kin.py`. A backup of those four JSON files is under `station-backups/2026-11-15-servo-calibration/existing-station-fit/`. Wrist fit RMS is reported as 1.99 pixels at its recorded pose; jaw target is (379.1,267.5), not the center of the image or our temporary cyan reference point. These are existing calibration records, not freshly validated guarantees. They must be checked at their hover height and against current jaw/camera mounting before reuse.

Important distinction: the existing wrist model maps **Cartesian millimetres to pixels**, with yaw rotation and learned correction. Our draft maps **normalized joint changes to pixels**. Never copy one matrix into the other. The existing controller applies differential IK, checks reachability, tracks target identity and uses a persistent motion server. Its whole sort loop follows GOOD/DEFECT plate routing, so do not launch it unchanged for the user's all-blocks-to-blue task. Its gripper-width-only held check is weaker than independent measured lift verification; preserve that distinction when integrating with the frozen policy.
