"""Fit each sim arm joint's motor response to the real arm, from real recordings. No hardware.

Real logs hold the commanded joint targets and the measured joint positions over time.
The real commands are replayed through the sim arm (free space: bricks and bowls do not
collide), and each joint's motor stiffness, motor damping, joint damping and friction
are adjusted until that joint moves like the real one. Two kinds of log are used
together: teleoperation episodes (LeRobot datasets, 30 Hz) and autonomous runs
(joints.csv with t, goal0-5, pos0-5). Every third episode of each kind is held out.
"""
import argparse
import glob
import json
from pathlib import Path

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import minimize

from sorting_env import DYNAMICS, BrickSortingEnv
from station_coordinates import StationCoordinates

JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')   # gripper not fitted:
PARAMETERS = ('kp', 'kv', 'damping', 'frictionloss')                                    # it stops on bricks the replay lacks


def load_dataset(dataset):
    table = pa.concat_tables([pq.read_table(f, columns=['observation.state', 'action', 'episode_index'])
                              for f in sorted(glob.glob(str(dataset/'data/*/*.parquet')))])
    state, action = (np.array(table[k].to_pylist()) for k in ('observation.state', 'action'))
    episode = np.array(table['episode_index'].to_pylist())
    # the state in row i was measured before the command in row i was sent
    return [(np.arange((episode == k).sum())/30, action[episode == k], state[episode == k]) for k in np.unique(episode)]


def load_csv(path):
    rows = np.loadtxt(path, delimiter=',', skiprows=1)
    return rows[:, 0]-rows[0, 0], rows[:, 1:7], rows[:, 7:13]


class Replay:
    def __init__(self):
        self.env = env = BrickSortingEnv()
        env.reset(1)
        m = self.m = env.model
        robot = m.body_rootid[m.geom_bodyid] == m.body('base').id
        for g in range(m.ngeom):     # free-space replay: only the arm, the table and the mat collide
            if not robot[g] and m.geom(g).name not in ('table', 'black_mat'):
                m.geom_contype[g] = m.geom_conaffinity[g] = 0
        self.limits = m.jnt_range[env.joint_ids].copy()
        self.coordinates = StationCoordinates(env.limits)
        self.actuator = np.array([m.actuator(j).id for j in JOINTS])
        self.dof = np.array([m.joint(j).dofadr[0] for j in JOINTS])
        # BrickSortingEnv already applies the last fit; undo it so that scale 1 is the unmodified Menagerie motor.
        applied = np.array([[DYNAMICS['joints'][j][p] for p in PARAMETERS] for j in JOINTS])
        self.base = np.column_stack((m.actuator_gainprm[self.actuator, 0], -m.actuator_biasprm[self.actuator, 2],
                                     m.dof_damping[self.dof], m.dof_frictionloss[self.dof]))/applied
        self.scales = applied.copy()

    def radians(self, normalized):
        c = self.coordinates
        return np.clip(c.continuous_ticks(normalized)*c.slope+c.offset, self.limits[:, 0], self.limits[:, 1])

    def set(self, scales):
        """scales: (5 joints, 4 parameters) multipliers."""
        self.scales = np.array(scales, float)
        value, m = self.base*self.scales, self.m
        m.actuator_gainprm[self.actuator, 0] = value[:, 0]
        m.actuator_biasprm[self.actuator, 1] = -value[:, 0]
        m.actuator_biasprm[self.actuator, 2] = -value[:, 1]
        m.dof_damping[self.dof] = value[:, 2]
        m.dof_frictionloss[self.dof] = value[:, 3]

    def error(self, episodes, delay):
        """Per-joint RMS (degrees) between sim and real joint positions under the real commands."""
        env, d = self.env, self.env.data
        squared, count = np.zeros(5), 0
        for t, goal, position in episodes:
            goal, real = self.radians(goal), self.radians(position)
            mujoco.mj_resetData(self.m, d)
            d.qpos[env.qadr] = real[0]
            d.ctrl[env.actuator_ids] = real[0]
            mujoco.mj_forward(self.m, d)
            sent = np.searchsorted(t, t[:-1]-delay, side='right')-1      # command in force during each interval
            steps = np.maximum(1, np.round(np.minimum(np.diff(t), .5)*600).astype(int))
            for i in range(1, len(t)):
                d.ctrl[env.actuator_ids] = goal[max(sent[i-1], 0)]
                mujoco.mj_step(self.m, d, nstep=steps[i-1])
                squared += (d.qpos[env.qadr][:5]-real[i][:5])**2
            count += len(t)-1
        return np.rad2deg(np.sqrt(squared/count))

    def settle(self, start, goal, seconds=3):
        """Where the sim arm comes to rest when holding `goal` from `start` (both radians)."""
        env, d = self.env, self.env.data
        mujoco.mj_resetData(self.m, d)
        d.qpos[env.qadr] = start
        d.ctrl[env.actuator_ids] = goal
        mujoco.mj_step(self.m, d, nstep=int(seconds*600))
        return d.qpos[env.qadr].copy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, nargs='*', default=[], help='Real LeRobot datasets in station-normalized units')
    p.add_argument('--csv', nargs='*', default=[], help='joints.csv logs (globs allowed)')
    p.add_argument('--static', nargs='*', default=[], help='Capture directories with pose.json for a held-out sag check, plus --waypoints')
    p.add_argument('--waypoints', type=Path)
    p.add_argument('--output', type=Path, default=Path('fitted_dynamics.json'))
    p.add_argument('--evaluations', type=int, default=70)
    a = p.parse_args()
    sources = {'teleop': [e for d in a.dataset for e in load_dataset(d) if len(e[0]) > 60],
               'autonomous': [e for pattern in a.csv for f in sorted(glob.glob(pattern)) for e in [load_csv(f)] if len(e[0]) > 60]}
    sources = {k: v for k, v in sources.items() if v}
    held_out = {k: v[::3] for k, v in sources.items()}
    fit = {k: [e for i, e in enumerate(v) if i % 3][:6] for k, v in sources.items()}     # six per kind keeps each evaluation fast
    replay = Replay()
    # Delay is a property of how each kind of log was produced, not of the motors.
    delays = {'teleop': DYNAMICS['delay_steps_30hz']/30, 'autonomous': 0.}
    for kind in sources:
        options = (1/30, 2/30, 3/30) if kind == 'teleop' else (0., .035, .07)
        delays[kind] = min(options, key=lambda s: replay.error(fit[kind], s).mean())
    print('delays (s)', delays, flush=True)

    def score(episodes):
        return np.mean([replay.error(episodes[k], delays[k]) for k in episodes], axis=0)

    start = replay.scales.copy()
    shared = {'fit': score(fit), 'held_out': score(held_out)}
    replay.set(np.ones((5, 4)))
    default = {'fit': score(fit), 'held_out': score(held_out)}
    replay.set(start)
    scales = start.copy()
    for round_number, evaluations in enumerate((a.evaluations, a.evaluations//2)):
        for j in (1, 2, 3, 0, 4):        # the gravity-loaded joints first; they disturb the others most

            def loss(x, j=j):
                trial = scales.copy()
                trial[j] = np.exp(x)
                replay.set(trial)
                return score(fit)[j]

            x0 = np.log(scales[j])
            result = minimize(loss, x0, method='Nelder-Mead', options={'xatol': .03, 'fatol': .003, 'maxfev': evaluations,
                              'initial_simplex': np.vstack([x0, x0+np.eye(4)*(.6 if round_number == 0 else .25)])})
            if result.fun < loss(x0):
                scales[j] = np.exp(result.x)
            replay.set(scales)
            print(f'round {round_number+1} {JOINTS[j]}: {result.fun:.3f} deg, scales {scales[j].round(4).tolist()}', flush=True)
    fitted = {'fit': score(fit), 'held_out': score(held_out)}
    report = {'delay_steps_30hz': int(round(delays.get('teleop', DYNAMICS['delay_steps_30hz']/30)*30)),
              'joints': {j: dict(zip(PARAMETERS, scales[i].round(4).tolist())) for i, j in enumerate(JOINTS)},
              'episodes': {k: {'fit': len(fit[k]), 'held_out': len(held_out[k])} for k in sources},
              'delay_seconds_by_source': delays,
              'joint_rms_deg': {split: {'menagerie_default': default[split].round(3).tolist(), 'shared_fit': shared[split].round(3).tolist(),
                                        'per_joint_fit': fitted[split].round(3).tolist()} for split in ('fit', 'held_out')},
              'note': 'Per-joint multipliers on the Menagerie sts3215 actuator kp/kv and joint damping/frictionloss. Gripper not fitted.'}
    if a.static and a.waypoints:
        # Held-out check of a different kind: where the real arm settled when told to hold a pose.
        ways = {w['name'].replace(' ', '-'): np.array(w['ticks'], float) for w in json.loads(a.waypoints.read_text())['waypoints']}
        c, rows = replay.coordinates, []
        for directory in a.static:
            name = Path(directory).name.split('-', 1)[1]
            goal = np.clip(ways[name]*c.slope+c.offset, replay.limits[:, 0], replay.limits[:, 1])
            real = np.array(list(json.loads((Path(directory)/'pose.json').read_text())['ticks'].values()), float)*c.slope+c.offset
            sim = replay.settle(real, goal)
            rows.append({'pose': name, 'real_sag_deg': np.rad2deg(real-goal)[:5].round(2).tolist(), 'sim_sag_deg': np.rad2deg(sim-goal)[:5].round(2).tolist()})
        report['static_sag_check'] = rows
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
