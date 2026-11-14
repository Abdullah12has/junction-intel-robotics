"""Execute the physical simulation expert and retain success-only demonstrations."""
import argparse
import hashlib
import json
from pathlib import Path
from queue import SimpleQueue
import time

import av
import mujoco
import numpy as np

from expert import PickPlaceExpert, PlanningError
from sorting_env import BrickSortingEnv
from station_coordinates import StationCoordinates, UNITS, constrain_simulation


class EpisodeRecorder:
    def __init__(self, directory, coordinates=None):
        self.coordinates = coordinates
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.states, self.actions, self.trace = [], [], []
        self.videos = {}
        for camera in ('overhead', 'wrist'):
            container = av.open(str(directory / f'{camera}.mp4'), mode='w')
            stream = container.add_stream('libx264', rate=30)
            stream.width, stream.height, stream.pix_fmt = 640, 480, 'yuv420p'
            stream.options = {'crf':'18', 'preset':'fast'}
            self.videos[camera] = (container,stream)

    def record(self, observation, action, record, env):
        state = observation['observation.state'].copy()
        target = action.copy()
        if self.coordinates is not None:
            state = self.coordinates.from_model(state)
            target = self.coordinates.from_model(target)
            # Reject unrepresentable trajectories; never clip or relabel them.
            self.coordinates.to_model(target)
            if np.any(state < self.coordinates.low) or np.any(state > self.coordinates.high):
                raise ValueError('Simulated state outside hardware encoder range')
        self.states.append(state)
        self.actions.append(target)
        self.trace.append(record)
        for camera,(container,stream) in self.videos.items():
            frame = av.VideoFrame.from_ndarray(observation[f'observation.images.{camera}'], format='rgb24')
            for packet in stream.encode(frame):container.mux(packet)

    def finish(self, env, success, error):
        for container,stream in self.videos.values():
            for packet in stream.encode():container.mux(packet)
            container.close()
        np.savez_compressed(self.directory/'trajectory.npz',
                            state=np.array(self.states,dtype=np.float32),
                            action=np.array(self.actions,dtype=np.float32))
        trace=self.trace
        grasp_verified=any(len(t['gripper_contact_bodies'])==2 and t['object_xyz'][2]>.06 for t in trace)
        accepted=bool(success and grasp_verified and trace and trace[-1]['success'])
        meta={'accepted':accepted,'success':bool(success),'physical_grasp_verified':grasp_verified,
              'frames':len(trace),'fps':30,'task':env.instruction,'target_color':env.target_color,
              'destination':env.destination,'source':'mujoco_simulation',
              'action_units':'absolute_joint_radians','hardware_mapping_verified':False,
              'geom_friction':env.model.geom_friction.tolist(),
              'friction_provenance':'recorded from the episode model',
              'error':error,'mujoco_version':mujoco.__version__,
              'initial_object_xyz':trace[0]['object_xyz'] if trace else None,
              'final_object_xyz':trace[-1]['object_xyz'] if trace else None}
        if self.coordinates is not None:
            meta.update(self.coordinates.provenance())
        (self.directory/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        (self.directory/'trace.json').write_text(json.dumps(trace)+'\n')
        return meta


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--episodes',type=int,default=10,help='Desired successful episodes')
    p.add_argument('--attempts',type=int,default=30)
    p.add_argument('--output',type=Path,default=Path('outputs/demonstrations'))
    p.add_argument('--viewer',action='store_true')
    p.add_argument('--reference',action='store_true')
    p.add_argument('--replay',action='store_true',help='Loop accepted recorded actions in the viewer; collect no new data')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--schedule',type=Path,help='Record only successful scenarios from a preflight JSON')
    p.add_argument('--mapping',type=Path,help='Joint mapping JSON; default is the nominal mapping')
    a=p.parse_args()
    if a.replay:
        replay(a.output)
        return
    probe=BrickSortingEnv()
    try: coordinates=StationCoordinates(probe.limits,mapping=a.mapping)
    finally: probe.close()
    a.output.mkdir(parents=True,exist_ok=False)
    env=BrickSortingEnv(render=True,horizon=1800)
    constrain_simulation(env,coordinates)
    keys=SimpleQueue()
    viewer=None
    if a.viewer:
        from mujoco import viewer as mj_viewer
        viewer=mj_viewer.launch_passive(env.model,env.data,key_callback=keys.put)
        viewer.cam.lookat[:]=[.04,-.17,.02]
        viewer.cam.distance=.65;viewer.cam.azimuth=90;viewer.cam.elevation=-60
        viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    summary=[]
    successful=0
    try:
        # Begin with the two contact-verified tasks; then cover other colors.
        colors=('red','orange','white','cyan','yellow','purple')
        schedule=[r for r in json.loads(a.schedule.read_text()) if r['success']] if a.schedule else None
        for attempt in range(min(a.attempts,len(schedule)) if schedule else a.attempts):
            if successful>=a.episodes or (viewer and not viewer.is_running()):break
            color=colors[attempt%6]
            destination=('blue','green')[(attempt//6+attempt)%2]
            seed=a.seed+attempt//6
            if schedule:
                color,destination,seed=(schedule[attempt][k] for k in ('color','destination','seed'))
            env.reset(seed,target_color=color,destination=destination,reference=a.reference)
            recorder=EpisodeRecorder(a.output/f'attempt-{attempt:04d}',coordinates)
            previous=time.monotonic()
            def on_step(obs,action,record,environment):
                nonlocal previous
                recorder.record(obs,action,record,environment)
                if viewer:
                    if not viewer.is_running():raise PlanningError('Viewer closed by user')
                    while not keys.empty():
                        key=keys.get()
                        if key in (49,50,51):
                            viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FREE if key==51 else mujoco.mjtCamera.mjCAMERA_FIXED
                            if key!=51:viewer.cam.fixedcamid=env.model.camera(('overhead','wrist')[key-49]).id
                    viewer.set_texts((None,mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                                      env.instruction+f"\n{record['phase']} | accepted: {successful}\n1 overhead | 2 close | 3 free orbit | SIMULATION ONLY",''))
                    viewer.sync()
                    time.sleep(max(0,1/30-(time.monotonic()-previous)))
                    previous=time.monotonic()
            expert=PickPlaceExpert(env,on_step=on_step)
            error=None
            try:success=expert.run()
            except PlanningError as exc:success=False;error=str(exc)
            meta=recorder.finish(env,success,error)
            meta.update(attempt=attempt,seed=seed,reference_layout=a.reference)
            (recorder.directory/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
            summary.append(meta)
            successful+=int(meta['accepted'])
            (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            print(json.dumps(meta),flush=True)
    finally:
        if viewer:viewer.close()
        env.close()
    print(f'Accepted {successful}/{len(summary)} attempts. Failures remain marked and are excluded by export.',flush=True)


def replay(directory):
    from mujoco import viewer as mj_viewer
    accepted=[e for e in json.loads((directory/'summary.json').read_text()) if e['accepted']]
    if not accepted:raise ValueError('No successful episodes to replay')
    env=BrickSortingEnv(horizon=1800)
    keys=SimpleQueue()
    try:
        with mj_viewer.launch_passive(env.model,env.data,key_callback=keys.put) as viewer:
            viewer.cam.lookat[:]=[.04,-.17,.02]
            viewer.cam.distance=.65;viewer.cam.azimuth=90;viewer.cam.elevation=-60
            viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FREE
            while viewer.is_running():
                for episode in accepted:
                    if not viewer.is_running():break
                    env.reset(episode['seed'],reference=episode['reference_layout'],
                              target_color=episode['target_color'],destination=episode['destination'])
                    path=directory/f"attempt-{episode['attempt']:04d}"
                    metadata=json.loads((path/"metadata.json").read_text())
                    if "geom_friction" in metadata:
                        env.model.geom_friction[:]=np.asarray(metadata["geom_friction"])
                    coordinates = None
                    if metadata.get('action_units') == UNITS:
                        coordinates=StationCoordinates(env.limits,mapping=metadata['coordinate_mapping'])
                    elif metadata.get('action_units') != 'absolute_joint_radians':
                        raise ValueError('Unknown replay coordinates')
                    with np.load(path/'trajectory.npz') as data:actions=data['action']
                    trace=json.loads((path/'trace.json').read_text())
                    for index,action in enumerate(actions):
                        if not viewer.is_running():break
                        start=time.monotonic()
                        if coordinates is not None:action=coordinates.to_model(action)
                        while not keys.empty():
                            key=keys.get()
                            if key in (49,50,51):
                                viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FREE if key==51 else mujoco.mjtCamera.mjCAMERA_FIXED
                                if key!=51:viewer.cam.fixedcamid=env.model.camera(('overhead','wrist')[key-49]).id
                        # Float32 dataset storage can round a limit outward by
                        # one ULP; reject anything larger than numerical noise.
                        clipped=np.clip(action,env.limits[:,0],env.limits[:,1])
                        if np.max(np.abs(clipped-action))>1e-6:raise ValueError('Recorded action outside model limits')
                        _,_,done,_,_=env.step(clipped)
                        viewer.set_texts((None,mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                            env.instruction+f"\n{trace[index]['phase']} | recorded expert replay\n1 overhead | 2 close | 3 free orbit | SIMULATION ONLY",''))
                        viewer.sync()
                        time.sleep(max(0,1/30-(time.monotonic()-start)))
                        if done:break
    finally:env.close()


if __name__=='__main__':main()
