"""Sanity-check the joint mapping against real recorded episodes. No hardware access.

Maps every recorded real joint state into the model and checks two physical
facts: the grasp point never goes through the table, and each time the gripper
starts closing the grasp point is at brick height. Wrong shoulder/elbow/wrist
signs or zeros fail this. It does not constrain pan, wrist roll or gripper.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from expert import PickPlaceExpert
from sorting_env import BrickSortingEnv
from station_coordinates import StationCoordinates


def check(dataset, mapping=None):
    table = pa.concat_tables([pq.read_table(f, columns=['observation.state', 'episode_index'])
                              for f in sorted(dataset.glob('data/*/*.parquet'))])
    state = np.array(table['observation.state'].to_pylist())
    episode = np.array(table['episode_index'].to_pylist())
    env = BrickSortingEnv()
    try:
        c = StationCoordinates(env.limits, mapping=mapping)
        radians = np.clip(c.continuous_ticks(state)*c.slope+c.offset, env.limits[:, 0], env.limits[:, 1])
        z = np.empty(len(radians))
        for i, q in enumerate(radians):
            env.data.qpos[env.qadr] = q
            mujoco.mj_kinematics(env.model, env.data)
            body = env.data.body('gripper')
            z[i] = (body.xpos + body.xmat.reshape(3, 3) @ PickPlaceExpert.grasp_point)[2]
    finally:
        env.close()
    closes = []
    for k in np.unique(episode):
        rows = np.flatnonzero(episode == k)
        closing = np.flatnonzero(np.diff(state[rows, 5]) < -1.5)
        if len(closing):
            closes.append(float(z[rows[closing[0]]]))
    report = {'frames': len(z), 'episodes': int(len(np.unique(episode))), 'min_grasp_point_z_m': float(z.min()),
              'grasp_point_z_at_first_close_m': closes}
    # Table is z=0; the station's bricks are about 19 mm tall.
    report['passed'] = bool(z.min() > -.005 and closes and all(0 < h < .04 for h in closes))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('dataset', type=Path, help='Real LeRobot dataset directory in station-normalized units')
    p.add_argument('--mapping', type=Path)
    a = p.parse_args()
    report = check(a.dataset, a.mapping)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 1)
