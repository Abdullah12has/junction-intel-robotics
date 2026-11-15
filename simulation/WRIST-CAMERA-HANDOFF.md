# Third wrist camera — 2026-11-14

User confirmed there is a separate third real camera on the wrist and requested a simulated counterpart. The simulated camera now exists as `wrist_mounted`, parented to the SO-101 `gripper` body after the wrist-roll joint. It moves with wrist flex and roll, not with the moving jaw.

Existing sorting cameras retain their names and transforms: `overhead` and `wrist` (the latter is the fixed close view being calibrated by Claude). Do not apply the fixed close-camera fit to `wrist_mounted`.

Changes are in `so101_env.py` (camera and rendered observation `observation.images.wrist_mounted`) and `smoke.py` (snapshot and viewer selection). In the native viewer, 1 = overhead, 2 = close, 3 = orbit, 4 = wrist-mounted; C cycles all four. Restart the viewer to load the new scene.

The initial local pose is provisional: position [0.10, 0, -0.04] metres, xyaxes [0, 1, 0, -0.669, 0, 0.743], vertical FOV 75 degrees. These values are a starting simulation view, not a measurement of the third real camera. Device serial, intrinsics, exact physical mounting body and camera-to-body extrinsics remain unverified. No real camera was opened or hardware moved for this change.

Validation rendered all three feeds, moved pan and wrist roll, and verified that the new camera moved while both fixed camera transforms stayed constant. Compared a model with the added camera against the same scene without it: body masses/inertias, joint ranges and actuator limits were identical. Preview frames are in ignored `outputs/wrist-mounted/home.png` and `moved.png`.

No edits were made to Claude's joint mapping, hardware calibration, station scene, station configuration or camera-fitting script. Existing two-camera dataset collectors/exporters were not changed: the third feed is exposed by the environment but is not yet wired into saved training datasets. Next: identify the real third feed, calibrate its intrinsics and rigid mount using synchronized images and joint states, then explicitly extend collection/export/training schemas and collect matching real demonstrations. Do not claim sim-to-real camera matching yet.
