"""Audit saved raw simulation episodes before LeRobot export and fine-tuning."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import av
import numpy as np
from sorting_env import BrickSortingEnv, PLATES
from station_coordinates import UNITS, StationCoordinates, fingerprint


def verify(root, minimum=150):
    env=BrickSortingEnv()
    limits=env.limits.copy()
    env.close()
    rows=[];seen=set()
    for metafile in sorted(root.glob('attempt-*/metadata.json')):
        path=metafile.parent
        meta=json.loads(metafile.read_text())
        if not meta['accepted']:continue
        key=(meta['seed'],meta['target_color'],meta['destination'])
        assert key not in seen, f'Duplicate scenario {key}'
        seen.add(key)
        assert meta['success'] and meta['physical_grasp_verified']
        trace=json.loads((path/'trace.json').read_text())
        n=meta['frames']
        assert len(trace)==n and trace[-1]['success'] and not trace[-1]['gripper_contact_bodies']
        assert any(t['object_xyz'][2]>.06 and len(t['gripper_contact_bodies'])==2 for t in trace)
        assert np.max(np.abs(np.diff([t['time'] for t in trace])-1/30))<1e-7
        # same placement rule as the environment: lying on the bowl floor within 4 cm of its centre
        assert np.max(np.abs(np.array(trace[-1]['object_xyz'][:2])-PLATES[meta['destination']]))<.04
        with np.load(path/'trajectory.npz') as arrays:
            for name in ('state','action'):
                assert arrays[name].shape==(n,6) and np.isfinite(arrays[name]).all()
            actions=arrays['action']
            if meta.get('action_units') == UNITS:
                assert meta.get('state_units') == UNITS
                mapping=meta['coordinate_mapping']
                assert fingerprint(mapping)==meta['mapping_sha256']
                c=StationCoordinates(limits,mapping=mapping)
                assert meta['calibration_sha256']==c.calibration_sha256
                assert np.all(arrays['state']>=c.low-2) and np.all(arrays['state']<=c.high+2)
                # Back to native angles for the physics-limit and velocity audit; to_model rejects out-of-range.
                actions=c.to_model(actions)
            elif meta.get('action_units') != 'absolute_joint_radians':
                raise ValueError('Unknown coordinate units')
            assert (actions>=limits[:,0]-1e-5).all() and (actions<=limits[:,1]+1e-5).all()
            assert np.max(np.abs(np.diff(actions,axis=0)))<.05
        for camera in ('overhead','wrist'):
            with av.open(str(path/f'{camera}.mp4')) as video:
                stream=video.streams.video[0]
                assert stream.width==640 and stream.height==480 and stream.average_rate==30
                assert sum(1 for _ in video.decode(video=0))==n
        rows.append(dict(attempt=path.name,seed=meta['seed'],color=meta['target_color'],destination=meta['destination'],frames=n,
                         trajectory_sha256=hashlib.sha256((path/'trajectory.npz').read_bytes()).hexdigest()))
    assert len(rows)>=minimum, f'Only {len(rows)} verified episodes; require {minimum}'
    report=dict(verified_episodes=len(rows),unique_scenes=len({x['seed'] for x in rows}),frames=sum(x['frames'] for x in rows),
                colors=dict(Counter(x['color'] for x in rows)),destinations=dict(Counter(x['destination'] for x in rows)),episodes=rows)
    (root/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    return {k:v for k,v in report.items() if k!='episodes'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--minimum',type=int,default=150)
    a=p.parse_args();print(json.dumps(verify(a.root,a.minimum),indent=2))
