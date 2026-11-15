"""Build one LeRobot dataset from real Studio episodes plus simulated episodes, for co-training.

Run in an environment with LeRobot 0.6 (the Studio backend venv on the Intel laptop). Both
sources use the station's normalized joint units and the two fixed RealSense views, under
the real camera keys. The real gripper-camera view is not included: the simulation's
gripper camera is not calibrated, and every episode must have the same views.

Real episodes are trimmed by frame range (start, and optionally end for an episode kept only up
to its last successful placement) and can be repeated
so they are not swamped by the more numerous sim frames. Originals are never modified.
"""
import argparse
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from lerobot.configs.video import DepthEncoderConfig, RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

OVERHEAD = 'observation.images.realsense 243622061421'
CLOSE = 'observation.images.realsense 243722064316'
SIM_VIDEO = {OVERHEAD: 'overhead.mp4', CLOSE: 'wrist.mp4'}
UNITS = 'so101_station_normalized_v1'


def decode(path, start=0.0, count=None):
    """RGB frames from `start` seconds, in order; `count` of them (all if None)."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if start > 0:
            container.seek(int(start/stream.time_base), stream=stream, backward=True)
        produced = 0
        for frame in container.decode(stream):
            if frame.pts is not None and float(frame.pts*stream.time_base) < start-1e-3:
                continue
            yield frame.to_ndarray(format='rgb24')
            produced += 1
            if count is not None and produced == count:
                return
    if count is not None and produced < count:
        raise ValueError(f'{path}: {produced} frames, expected {count}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--real', type=Path, required=True, help='Studio LeRobot v3 dataset directory')
    p.add_argument('--keep', required=True, help='episode:start[:end] frame ranges, e.g. 0:46,2:12:2069,3:0')
    p.add_argument('--real-repeat', type=int, default=3)
    p.add_argument('--sim', type=Path, required=True, help='Raw simulation recordings (attempt-*/)')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    info = json.loads((a.real/'meta/info.json').read_text())
    names = info['features']['action']['names']
    if info['features']['observation.state']['names'] != names or info['fps'] != 30:
        raise ValueError('Unexpected real dataset layout')
    keep = {}
    for item in a.keep.split(','):
        parts = [int(v) for v in item.split(':')]
        keep[parts[0]] = (parts[1], parts[2] if len(parts) > 2 else None)
    task = {r['task_index']: (r.get('task') or r.get('__index_level_0__'))
            for r in pq.read_table(a.real/'meta/tasks.parquet').to_pylist()}
    episodes = {r['episode_index']: r for r in pq.read_table(a.real/'meta/episodes').to_pylist()}
    data = pq.read_table(a.real/'data').to_pydict()
    index = np.array(data['episode_index'])
    sims = []
    for meta_file in sorted(a.sim.glob('attempt-*/metadata.json')):
        meta = json.loads(meta_file.read_text())
        if not meta['accepted']:
            continue
        if meta.get('action_units') != UNITS or meta.get('state_units') != UNITS:
            raise ValueError(f'{meta_file.parent}: not in station-normalized units')
        sims.append((meta_file.parent, meta))

    features = {k: {'dtype': 'float32', 'shape': (6,), 'names': names} for k in ('action', 'observation.state')}
    for key in (OVERHEAD, CLOSE):
        features[key] = {'dtype': 'video', 'shape': (480, 640, 3), 'names': ['height', 'width', 'channels']}
    out = LeRobotDataset.create(repo_id='local/cotrain-real-sim', root=a.output, fps=30, robot_type=info.get('robot_type'),
                                features=features, video_backend='pyav', streaming_encoding=True, image_writer_threads=2,
                                encoder_threads=4,
                                depth_encoder=DepthEncoderConfig(vcodec='h264', pix_fmt='yuv420p', extra_options={}, preset='fast'),
                                rgb_encoder=RGBEncoderConfig(vcodec='h264', pix_fmt='yuv420p', crf=20, preset='fast'))
    provenance = []
    for repeat in range(a.real_repeat):
        for episode, (trim, end) in sorted(keep.items()):
            rows = np.flatnonzero(index == episode)
            if end is not None:
                rows = rows[:end]
            state = np.array(data['observation.state'])[rows]
            action = np.array(data['action'])[rows]
            text = task[data['task_index'][rows[0]]]
            n = len(rows)-trim
            streams = []
            for key in (OVERHEAD, CLOSE):
                e = episodes[episode]
                path = a.real/info['video_path'].format(video_key=key, chunk_index=e[f'videos/{key}/chunk_index'],
                                                          file_index=e[f'videos/{key}/file_index'])
                streams.append(decode(path, e[f'videos/{key}/from_timestamp']+trim/30, n))
            for i, (overhead, close) in enumerate(zip(*streams)):
                out.add_frame({'observation.state': state[trim+i].astype(np.float32), 'action': action[trim+i].astype(np.float32),
                               OVERHEAD: overhead, CLOSE: close, 'task': text})
            out.save_episode()
            provenance.append({'source': 'real', 'episode': episode, 'start_frame': trim, 'end_frame': trim+n, 'repeat': repeat, 'frames': n, 'task': text})
            print(f'real episode {episode} (repeat {repeat}): {n} frames', flush=True)
    for path, meta in sims:
        with np.load(path/'trajectory.npz') as t:
            state, action = t['state'], t['action']
        streams = [decode(path/SIM_VIDEO[key], 0, len(state)) for key in (OVERHEAD, CLOSE)]
        for i, (overhead, close) in enumerate(zip(*streams)):
            out.add_frame({'observation.state': state[i].astype(np.float32), 'action': action[i].astype(np.float32),
                           OVERHEAD: overhead, CLOSE: close, 'task': meta['task']})
        out.save_episode()
        provenance.append({'source': 'sim', 'attempt': path.name, 'seed': meta['seed'], 'frames': len(state), 'task': meta['task']})
        print(f'sim {path.name}: {len(state)} frames', flush=True)
    out.finalize()
    real_frames = sum(x['frames'] for x in provenance if x['source'] == 'real')
    sim_frames = sum(x['frames'] for x in provenance if x['source'] == 'sim')
    (a.output/'cotrain_provenance.json').write_text(json.dumps({
        'units': UNITS, 'cameras': [OVERHEAD, CLOSE], 'real_dataset': str(a.real), 'real_keep': {k: list(v) for k, v in keep.items()}, 'real_repeat': a.real_repeat,
        'real_frames': real_frames, 'sim_frames': sim_frames, 'real_fraction': round(real_frames/(real_frames+sim_frames), 3),
        'episodes': provenance}, indent=1)+'\n')
    print(f'Done: {len(provenance)} episodes, real {real_frames} frames, sim {sim_frames} frames', flush=True)


if __name__ == '__main__':
    main()
