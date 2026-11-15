"""Convert accepted simulation recordings with the installed LeRobot writer.

Run in the CUDA trainer's existing LeRobot environment, not Studio's device
process. Produces a separate sim-only dataset; never publishes to the Hub.
"""
import argparse
import hashlib
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.io_utils import write_stats
from lerobot.utils.utils import unflatten_dict
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.video import RGBEncoderConfig, DepthEncoderConfig

from station_coordinates import UNITS, fingerprint

JOINTS=['shoulder_pan.pos','shoulder_lift.pos','elbow_flex.pos','wrist_flex.pos','wrist_roll.pos','gripper.pos']


def write_train_stats(root, training):
    rows=pq.read_table(root/'meta'/'episodes').to_pylist()
    stats=[]
    for row in rows:
        if row['episode_index'] not in training:continue
        raw=unflatten_dict({k:v for k,v in row.items() if k.startswith('stats/')})['stats']
        stats.append({feature:{name:np.asarray(value) for name,value in values.items()} for feature,values in raw.items()})
    original=root/'all_episode_stats.json'
    if not original.exists():original.write_bytes((root/'meta'/'stats.json').read_bytes())
    write_stats(aggregate_stats(stats),root)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--min-episodes',type=int,default=150)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    accepted=[]
    for f in sorted(a.input.glob('attempt-*/metadata.json')):
        meta=json.loads(f.read_text())
        if meta['accepted'] and meta['physical_grasp_verified'] and meta['success']:
            if meta.get('action_units') != UNITS or meta.get('state_units') != UNITS:
                raise ValueError(f'{f.parent}: radians recording; run convert_recordings.py first')
            if fingerprint(meta.get('coordinate_mapping')) != meta.get('mapping_sha256'):
                raise ValueError('Invalid mapping provenance')
            accepted.append((f.parent,meta))
    if len(accepted)<a.min_episodes:raise ValueError(f'Need {a.min_episodes} verified episodes before fine-tuning')
    if len({meta['mapping_sha256'] for _,meta in accepted}) != 1:
        raise ValueError('Cannot mix coordinate mappings')
    features={k:{'dtype':'float32','shape':(6,),'names':JOINTS} for k in ['action','observation.state']}
    for name in ['camera1','camera2']:
        features['observation.images.'+name]={'dtype':'video','shape':(480,640,3),'names':['height','width','channels']}
    dataset=LeRobotDataset.create(repo_id='local/junction-sim-sorting',root=a.output,fps=30,
                                  robot_type='so101_station_normalized_simulation',features=features,
                                  video_backend='pyav',streaming_encoding=True,
                                  image_writer_threads=4,
                                  # No depth features; avoid probing the unsupported default HEVC depth codec.
                                  depth_encoder=DepthEncoderConfig(vcodec='h264',pix_fmt='yuv420p',extra_options={},preset='fast'),
                                  rgb_encoder=RGBEncoderConfig(vcodec='h264',pix_fmt='yuv420p',crf=18,preset='fast'))
    provenance=[]
    for path,meta in accepted:
        with np.load(path/'trajectory.npz') as arrays:
            states,actions=arrays['state'],arrays['action']
        n=len(states)
        if states.shape!=(n,6) or actions.shape!=(n,6) or n!=meta['frames'] or not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f'Invalid trajectory in {path}')
        # States may overshoot slightly, as real readings do; commands may not.
        if np.any(states < np.array([-102]*5+[-2])) or np.any(states > 102) or np.any(actions < np.array([-100]*5+[0])) or np.any(actions > 100):
            raise ValueError('Normalized state/action outside station bounds')
        videos=[av.open(str(path/f'{camera}.mp4')) for camera in ('overhead','wrist')]
        generators=[v.decode(video=0) for v in videos]
        try:
            for i in range(n):
                images=[next(g).to_ndarray(format='rgb24') for g in generators]
                dataset.add_frame({'observation.state':states[i], 'action':actions[i],
                                   'observation.images.camera1':images[0],
                                   'observation.images.camera2':images[1], 'task':meta['task']})
            if any(next(g,None) is not None for g in generators):raise ValueError('Extra video frames')
        finally:
            for v in videos:v.close()
        dataset.save_episode()
        provenance.append({'source_attempt':path.name,'metadata':meta,
                           'trajectory_sha256':hashlib.sha256((path/'trajectory.npz').read_bytes()).hexdigest()})
        print(f'Exported {len(provenance)}: {meta["task"]} ({n} frames)',flush=True)
    dataset.finalize()
    seeds=sorted({x['metadata']['seed'] for x in provenance})
    rng=np.random.default_rng(42)
    rng.shuffle(seeds)
    held_out=set()
    remaining=max(1,round(len(provenance)*.2))
    for seed in seeds:
        count=sum(x['metadata']['seed']==seed for x in provenance)
        if count<=remaining:
            held_out.add(seed)
            remaining-=count
    if remaining:
        # For small paired pilots, round up to a whole scene.
        held_out.add(next(seed for seed in seeds if seed not in held_out))
    validation=[i for i,x in enumerate(provenance) if x['metadata']['seed'] in held_out]
    training=[i for i in range(len(provenance)) if i not in validation]
    write_train_stats(a.output,training)
    (a.output/'simulation_provenance.json').write_text(json.dumps({
        'source':'simulation_only','action_units':UNITS,'state_units':UNITS,'hardware_compatible':False,
        'joint_mapping_verified':accepted[0][1]['joint_mapping_verified'],'mapping_sha256':accepted[0][1]['mapping_sha256'],
        'coordinate_mapping':accepted[0][1]['coordinate_mapping'],
        'episodes':provenance,'validation_episodes':validation,
        'train_episodes':training,'split_by':'scene_seed','normalization_episodes':training},indent=2)+'\n')
    print(f'Finalized {len(provenance)} simulation episodes at {a.output}',flush=True)


if __name__=='__main__':main()
