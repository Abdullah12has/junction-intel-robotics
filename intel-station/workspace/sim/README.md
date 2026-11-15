# sim/ — digital twin of the SO-101 DUPLO sorting cell (MuJoCo 4.0.1)

Environment: `envs/sim-env` (MuJoCo 4.0.1, numpy, scipy, opencv-headless, pyyaml, pillow); render with `MUJOCO_GL=egl`.
Platform decision: MuJoCo through Intel's Physical AI MuJoCo plugin (`repos/physicalai/packages/physicalai-mujoco-plugin`),
so Studio, runtime, policies and datasets see the twin through the same robot interface as the real arm.

| folder | content | verification gate |
|---|---|---|
| `twin/robot` | SO-101 model with our kinematics and the real driver's units | G1 kinematics vs `app/tools/kin.py` |
| `twin/dynamics` | joint actuator parameters identified from the 30 Hz joint logs | G2 joint tracking vs recorded episodes |
| `twin/cameras` | top / close RealSense and wrist camera models fitted from recordings | G3 reprojection |
| `twin/scene` | mat, plates, bricks, lighting measured from depth and images | G4/G5 rendered vs real |
| `twin/tests` | gate tests | |
