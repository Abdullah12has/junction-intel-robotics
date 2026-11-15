"""Convert accepted radian recordings to station-normalized units without recollecting.

The mapping is affine per joint, so states and actions convert exactly. The one
lossy step: the expert's gripper close target (-0.08 rad) is below the real
gripper's closed position, so gripper actions are raised to the common-range
minimum. Every converted episode is then replayed through physics and kept only
if the placement still succeeds. Videos and traces are hard-linked, not copied.
"""
import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np

from sorting_env import BrickSortingEnv
from station_coordinates import StationCoordinates


def replay_succeeds(env, meta, radians):
    env.reset(meta['seed'], target_color=meta['target_color'], destination=meta['destination'],
              reference=meta['reference_layout'])
    env.model.geom_friction[:] = np.asarray(meta['geom_friction'])
    mujoco.mj_forward(env.model, env.data)
    for action in radians:
        # float32 storage can round a limit outward by one ULP.
        _, _, done, _, _ = env.step(np.clip(action, env.limits[:, 0], env.limits[:, 1]))
        if done:
            return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path('outputs/dataset-150-raw'))
    p.add_argument('--output', type=Path, default=Path('outputs/dataset-150-normalized'))
    p.add_argument('--mapping', type=Path, help='Joint mapping JSON; default is the nominal mapping')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    env = BrickSortingEnv(horizon=1800)
    c = StationCoordinates(env.limits, mapping=a.mapping)
    summary, dropped = [], []
    try:
        for metafile in sorted(a.input.glob('attempt-*/metadata.json')):
            meta = json.loads(metafile.read_text())
            if not meta['accepted']:
                continue
            if meta['action_units'] != 'absolute_joint_radians':
                raise ValueError(f'{metafile.parent}: already converted')
            with np.load(metafile.parent/'trajectory.npz') as arrays:
                state = c.from_model(arrays['state']).astype(np.float32)
                action = c.from_model(arrays['action'])
            raised = int(np.sum(action[:, 5] < c.command_low[5]))
            action[:, 5] = np.maximum(action[:, 5], c.command_low[5])
            action = action.astype(np.float32)
            try:
                radians = c.to_model(action)  # rejects anything else outside the common range
                if np.any(state < c.low) or np.any(state > c.high):
                    raise ValueError('state outside encoder range')
                if not replay_succeeds(env, meta, radians):
                    raise ValueError('converted actions no longer place the brick')
            except ValueError as error:
                dropped.append({'attempt': metafile.parent.name, 'reason': str(error)})
                continue
            out = a.output/f'attempt-{len(summary):04d}'
            out.mkdir()
            np.savez_compressed(out/'trajectory.npz', state=state, action=action)
            for name in ('overhead.mp4', 'wrist.mp4', 'trace.json'):
                os.link(metafile.parent/name, out/name)
            meta.update(c.provenance(), attempt=len(summary), converted_from=str(metafile.parent),
                        gripper_actions_raised_to_closed=raised, replay_verified_after_conversion=True)
            (out/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
            summary.append(meta)
    finally:
        env.close()
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (a.output/'conversion.json').write_text(json.dumps(
        {'converted': len(summary), 'dropped': dropped, 'mapping_sha256': c.mapping_sha256,
         'command_low': c.command_low.tolist(), 'command_high': c.command_high.tolist()}, indent=2)+'\n')
    print(f'Converted {len(summary)}; dropped {len(dropped)}: {dropped}')


if __name__ == '__main__':
    main()
