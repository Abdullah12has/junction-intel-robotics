import json, sys, glob, os
import numpy as np
import pandas as pd

root = sys.argv[1]
info = json.load(open(os.path.join(root, 'meta', 'info.json')))
print('INFO_KEYS', list(info.keys()))
print('fps', info.get('fps'), 'robot_type', info.get('robot_type'), 'codebase_version', info.get('codebase_version'))
tasks = pd.read_parquet(os.path.join(root, 'meta', 'tasks.parquet'))
print('TASKS cols', list(tasks.columns), 'index', tasks.index.name, 'nrows', len(tasks))
print(tasks.reset_index().to_string())
eps = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(root, 'meta', 'episodes', 'chunk-*', 'file-*.parquet')))]
ep = pd.concat(eps, ignore_index=True)
print('EP_FILES', sorted(glob.glob(os.path.join(root, 'meta', 'episodes', 'chunk-*', 'file-*.parquet'))))
print('EP cols', list(ep.columns))
print('EP nrows', len(ep))
dcols = [c for c in ep.columns if not c.startswith('stats/')]
print(ep[dcols].to_string())
dat = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(root, 'data', 'chunk-*', 'file-*.parquet')))], ignore_index=True)
print('DATA cols', list(dat.columns), dat.shape)
print(dat.head(2).to_string())
print('data episodes', dat['episode_index'].nunique(), 'rows per ep', dat.groupby('episode_index').size().to_dict())
S = np.stack(dat['observation.state'].values)
A = np.stack(dat['action'].values)
print('state shape', S.shape, 'action shape', A.shape)
names = info['features']['observation.state']['names']
print('state names', names)
for i, n in enumerate(names):
    print('JOINT', n, 'min', round(float(S[:, i].min()), 2), 'max', round(float(S[:, i].max()), 2))
gi = names.index('gripper.pos') if 'gripper.pos' in names else len(names) - 1
for e, g in dat.groupby('episode_index'):
    idx = g.index.values
    gp = S[idx, gi]
    ts = g['timestamp'].values
    dts = np.diff(ts)
    print('EP', e, 'n', len(idx), 'tfirst', round(float(ts[0]), 3), 'tlast', round(float(ts[-1]), 3), 'dt_med', round(float(np.median(dts)), 4), 'dt_max', round(float(dts.max()), 4), 'grip_min', round(float(gp.min()), 2), 'grip_max', round(float(gp.max()), 2), 'grip_first', round(float(gp[0]), 2), 'grip_last', round(float(gp[-1]), 2), 'task_idx', int(g['task_index'].iloc[0]), 'fi_ok', bool((g['frame_index'].values == np.arange(len(idx))).all()))
