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
