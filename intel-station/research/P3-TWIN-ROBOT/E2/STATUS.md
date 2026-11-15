PARTIAL 2026-11-14T20:50Z E2 P3-TWIN-ROBOT: so101_twin_v2.xml built (collision, actuators); G1 PASS on v2; grasp NOT yet working (0/10 holds); jaw-gap table not valid yet; gripper law conflict open.

## Built
- sim/twin/robot/so101_twin_v2.xml (new; v1 unchanged). Generator: work/P3-TWIN-ROBOT/E2/tools/make_v2.py.
  - 17 collision geoms (group 3, convex hulls of the v1 meshes). Arm links contype 2 conaffinity 1; gripper_link and moving jaw contype 1 conaffinity 1. Visual geoms unchanged (group 2, contype 0).
  - 3 excludes: world-base_link (base sits above the mat; mesh bottom not measured), wrist_link-gripper_link, gripper_link-moving jaw.
  - 6 <position> actuators, kp/kv/forcerange from sim/twin/dynamics/twin_dynamics_v1.json (P3-TWIN-DYN, published). armature and frictionloss set on joints from the same file; damping 0.
  - ctrl units: radians. Arm ctrl = q = k_j * goal (urdf_zero law, the law the DYN fit used). Gripper ctrl = -0.174533 + k_g*(goal - 0.52) (units.py closed_at_empty_reading). DYN gripper gains were fit under the urdf_zero law, so this mixes laws; open question for G3.
- sim/twin/tests/test_grasp.py (new, plain runner; pytest not installed in sim-env).

## Checks (outputs/)
- check_v2.json: v1 0 collision, 0 actuators; v2 17 collision, 6 actuators, 3 excludes. G1 on v2: max 1.36e-8 mm over 200 samples (gate 0.5 mm) PASS.
- grasp_results.json and logs/test_grasp.log (run 2026-11-14 ~20:50Z):
  - PASS test_g1_v2_fixed_sample.
  - PASS test_empty_close_reads_empty_value (closing on nothing reads inside 0.50-0.60).
  - FAIL test_brick_grasp_hold_nominal: the hover IK converges (0.004 mm), but the settled TCP sits 18.6 mm off target. Descent stages land at 20.2 mm (target 26) and 7.2 mm (target 8). Brick not lifted.
  - FAIL test_contact_stability_10_seeds: held 0/10.
  - FAIL test_jaw_gap_is_monotone: the hull gap is -13.61 mm at every reading. This is not a valid gap measurement; the jaw-gap function needs debugging before any implied openings are stated.

## Open items (not resolved in E2)
1. Gripper law conflict. A linear map cannot satisfy closed-on-nothing 0.52, held 1.57-2.29 for a 31.7 mm brick, and open 24.6 together. The URDF span gives 0.0191986 rad/unit; the tick scale gives 0.0234910. The closed_at_empty_reading law is used. The held readings imply about 2 mm of jaw travel (E1 estimate), which does not fit a 31.7 mm brick.
2. Position error under gravity and the position actuators: about 4-18 mm TCP error at the stages. Needs a feedback or settle-time study; DYN shoulder_lift and elbow in-sample RMSE is 0.82 units (poor fit).
3. Approach axis assumed to be TCP site z (Z_DOWN). Not checked against the jaw geometry.
4. Mat is a 1.2 m box at the measured MAT_Z (placeholder extent). Brick mass is the 20.07 g solid upper bound; friction 0.35 (assumed, not measured).
5. Arm-to-TCP difference from V1 (15.6 mm median, 30 mm max) remains open for G3; the plugin's unit law is not modelled here.

## DC call budget
- About 14 Desktop Commander calls in this pass. Remaining budget per coordinator: 150 for E2, so about 136 left (this count may be approximate). Still to do: README v2 section (1 call), this STATUS (done).
