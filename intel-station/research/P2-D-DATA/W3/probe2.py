import json, sys, glob, os
import numpy as np
import pandas as pd

root = sys.argv[1]
info = json.load(open(os.path.join(root, 'meta', 'info.json')))
dat = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(root, 'data', 'chunk-*', 'file-*.parquet')))], ignore_index=True)
S = np.stack(dat['observation.state'].values).astype(float)
A = np.stack(dat['action'].values).astype(float)
for e, g in dat.groupby('episode_index'):
    idx = g.index.values
    s = S[idx]
    a = A[idx]
    print('EP', e, 'n', len(idx), 'start_state', np.round(s[0], 1).tolist(), 'end_state', np.round(s[-1], 1).tolist())
    print('GRIP_S', ' '.join(f'{x:.0f}' for x in s[::15, 5]))
    print('GRIP_A', ' '.join(f'{x:.0f}' for x in a[::15, 5]))
    print('LIFT  ', ' '.join(f'{x:.0f}' for x in s[::15, 1]))
    print('ELBOW ', ' '.join(f'{x:.0f}' for x in s[::15, 2]))
    print('PAN   ', ' '.join(f'{x:.0f}' for x in s[::15, 0]))
    print('WFLEX ', ' '.join(f'{x:.0f}' for x in s[::15, 3]))
