"""Convert a so101-mujoco camera calibration into this simulation's world frame.

so101-mujoco fits camera extrinsics from depth (table plane + arm mesh, about 5 mm),
which is far better than the colour fit in fit_cameras.py. Its calibration checked in
on 2026-11-14 predates a rearrangement of the rig, so it is NOT applied here; after
its `calibrate_model` / `fit_captures` is rerun at the station, run:

    uv run python import_so101_mujoco_cameras.py ../../so101-mujoco/embodiments/so101/camera_extrinsics.yaml

and copy the printed entries into fitted_cameras.json.
"""
import json
import re
import sys
from pathlib import Path

import mujoco
import numpy as np

# sim_world = R @ task + T. Fitted from both kinematic chains' tool points over 200
# random poses (max residual 0.46 mm): same arm model, same joint zeros.
R_TASK_TO_SIM = np.array([[0., 1, 0], [-1, 0, 0], [0, 0, 1]])
T_TASK_TO_SIM = np.array([0.0181, 0.0231, -0.0325])
NAMES = {'overview': 'overhead', 'front': 'wrist'}   # their view name -> our camera name
FOVY_DEG = 43.3   # D415 colour at 640x480: fy = 604.4 px in their recorded intrinsics


def convert(text):
    out = {}
    for theirs, ours in NAMES.items():
        numbers = [float(x) for x in re.findall(r'-\s+(-?\d+\.\d+)', text[text.index(f'  {theirs}:'):])[:12]]
        rotation, translation = np.array(numbers[:9]).reshape(3, 3), np.array(numbers[9:])
        # task <- camera (OpenCV axes: x right, y down, z forward) to MuJoCo (x right, y up, looks along -z)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (R_TASK_TO_SIM @ rotation @ np.diag([1., -1, -1])).ravel())
        out[ours] = {'pos': (R_TASK_TO_SIM @ translation + T_TASK_TO_SIM).round(4).tolist(),
                     'quat_wxyz': quat.round(5).tolist(), 'fovy_deg': FOVY_DEG}
    return out


if __name__ == '__main__':
    print(json.dumps(convert(Path(sys.argv[1]).read_text()), indent=2))
