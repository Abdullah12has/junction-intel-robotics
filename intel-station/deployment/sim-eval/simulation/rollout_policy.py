"""Closed-loop SmolVLA evaluation. No expert actions or object-state policy inputs."""
import argparse
import base64
import io
import json
from pathlib import Path
from queue import SimpleQueue, Queue, Empty
from threading import Thread
import subprocess
import shlex
import time

import mujoco
import numpy as np
from collect import EpisodeRecorder
from sorting_env import BrickSortingEnv, PLATES
from station_coordinates import StationCoordinates, UNITS


class PolicyClient:
    def __init__(self,host,output,checkpoint=None,dataset=None,local=None):
        self.log=(output/'inference-stderr.log').open('w')
        self.stdout_log=(output/'inference-stdout.log').open('w')
        if local:   # local=(python, contract json): run the worker on this machine (Apple GPU)
            command=[local[0],'-u',str(Path(__file__).with_name('policy_worker.py')),'--checkpoint',checkpoint,'--contract',local[1],'--device','mps']
        else:
            remote='docker exec -i physicalai-trainer-b963d065-34f1-4740-8448-4bf9fdca2abd /opt/venv/bin/python -u /var/lib/physicalai-trainer/junction-simulation-training/policy_worker.py'
            if checkpoint is not None:remote+=' --checkpoint '+shlex.quote(checkpoint)
            if dataset is not None:remote+=' --dataset '+shlex.quote(dataset)
            command=['ssh','-T','-i',str(Path.home()/'.ssh/junction_verda_ed25519'),'-o','BatchMode=yes','-o','ServerAliveInterval=15',f'root@{host}',remote]
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
        self.lines=Queue()
        def read_stdout():
            for line in self.process.stdout:self.lines.put(line)
            self.lines.put(None)
        Thread(target=read_stdout,daemon=True).start()
        self.ready=self.read(180)
        if not self.ready.get('ready'):raise RuntimeError(self.ready)
        print(json.dumps(self.ready),flush=True)

    def read(self,timeout=90):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            try:line=self.lines.get(timeout=max(0,deadline-time.monotonic()))
            except Empty:break
            if not line:raise RuntimeError('GPU worker disconnected; see inference logs')
            self.stdout_log.write(line);self.stdout_log.flush()
            if line.startswith('VLA_RESPONSE '):
                result=json.loads(line[len('VLA_RESPONSE '):])
                if 'error' in result:raise RuntimeError(result['error'])
                return result
        raise TimeoutError('GPU inference worker timed out')

    def predict(self,obs,reset,seed):
        buffer=io.BytesIO()
        np.savez_compressed(buffer,state=obs['observation.state'],camera1=obs['observation.images.overhead'],camera2=obs['observation.images.wrist'])
        request={'observation':base64.b64encode(buffer.getvalue()).decode(),'task':obs['task'],'reset':reset,'seed':seed,'action_units':self.ready['action_units'],'mapping_sha256':self.ready.get('mapping_sha256')}
        self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
        return self.read()

    def close(self):
        if self.process.stdin:self.process.stdin.close()
        try:self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:self.process.terminate();self.process.wait(timeout=5)
        self.log.close();self.stdout_log.close()


def configure_view(viewer):
    viewer.cam.lookat[:]=[.04,-.17,.02];viewer.cam.distance=.65
    viewer.cam.azimuth=90;viewer.cam.elevation=-60;viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FREE


def display(viewer,keys,env,label):
    if not viewer.is_running():raise RuntimeError('Viewer closed')
    while not keys.empty():
        key=keys.get()
        if key in (49,50,51):
            viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FREE if key==51 else mujoco.mjtCamera.mjCAMERA_FIXED
            if key!=51:viewer.cam.fixedcamid=env.model.camera(('overhead','wrist')[key-49]).id
    viewer.set_texts((None,mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                     env.instruction+'\n'+label+'\n1 overhead | 2 close | 3 free orbit | SIMULATION ONLY',''))
    viewer.sync()


def reset_episode(env,meta):
    env.reset(meta['seed'],target_color=meta['target_color'],destination=meta['destination'],reference=meta['reference_layout'])
    if meta.get('geom_friction') is not None:env.model.geom_friction[:]=np.asarray(meta['geom_friction'])
    mujoco.mj_forward(env.model,env.data)
    return env.observe()


def trace_frame(env,info):
    jaws={env.model.body('gripper').id,env.model.body('moving_jaw_so101_v1').id}
    contacts=set()
    for c in env.data.contact:
        if c.geom1 in env.brick_geoms and env.model.geom_bodyid[c.geom2] in jaws:contacts.add(int(env.model.geom_bodyid[c.geom2]))
        if c.geom2 in env.brick_geoms and env.model.geom_bodyid[c.geom1] in jaws:contacts.add(int(env.model.geom_bodyid[c.geom1]))
    return dict(phase='trained_vla',time=float(env.data.time),object_xyz=env.data.body(env.brick_body_name).xpos.tolist(),
                gripper_contact_bodies=sorted(contacts),success=info['success'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--host')
    p.add_argument('--local',nargs=2,metavar=('PYTHON','CONTRACT'),help='Run the policy worker here with this Python and contract JSON')
    p.add_argument('--fresh-seed',type=int,help='One unseen scene per task from this seed on, instead of the held-out recordings');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--raw',type=Path,default=Path('outputs/dataset-150-raw'))
    p.add_argument('--holdout',type=Path,default=Path('outputs/heldout-evaluation.json'))
    p.add_argument('--horizon',type=int,default=600);p.add_argument('--loop-results',action='store_true')
    p.add_argument('--checkpoint',help='Remote checkpoint directory')
    p.add_argument('--dataset',help='Remote training dataset directory')
    p.add_argument('--normalized',action='store_true',help='Checkpoint uses station-normalized units')
    p.add_argument('--mapping',type=Path,help='Joint mapping JSON; default is the nominal mapping')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    all_rows=json.loads((a.raw/'summary.json').read_text())
    schedule=[];tasks=set()
    if a.fresh_seed is not None:
        for meta in all_rows:
            if meta['task'] not in tasks:
                tasks.add(meta['task'])
                schedule.append((None,dict(task=meta['task'],seed=a.fresh_seed+len(schedule),target_color=meta['target_color'],
                                           destination=meta['destination'],reference_layout=False,geom_friction=None)))
    for i in ([] if a.fresh_seed is not None else json.loads(a.holdout.read_text())['validation_episodes']):
        meta=all_rows[i]
        if meta['task'] not in tasks:
            schedule.append((i,meta));tasks.add(meta['task'])
    assert len(schedule)==12,'Expected all 12 color/destination tasks in holdout'
    (a.output/'schedule.json').write_text(json.dumps([dict(source_episode=i,metadata=m) for i,m in schedule],indent=2))
    client=PolicyClient(a.host,a.output,a.checkpoint,a.dataset,a.local)
    env=BrickSortingEnv(render=True,horizon=a.horizon)
    coordinates=StationCoordinates(env.limits,mapping=a.mapping) if a.normalized or a.mapping else None
    expected=UNITS if coordinates else 'absolute_joint_radians'
    if client.ready.get('action_units')!=expected or client.ready.get('mapping_sha256')!=(coordinates.mapping_sha256 if coordinates else None):
        client.close();env.close();raise ValueError('Checkpoint/simulator coordinate contract mismatch')
    from mujoco import viewer as mj_viewer
    keys=SimpleQueue();results=[]
    try:
        with mj_viewer.launch_passive(env.model,env.data,key_callback=keys.put) as viewer:
            configure_view(viewer)
            for trial,(source_index,meta) in enumerate(schedule):
                obs=reset_episode(env,meta)
                recorder=EpisodeRecorder(a.output/f'attempt-{trial:04d}',coordinates)
                raw_actions=[];latencies=[];clipped=0;success=False;error=None
                try:
                    for step in range(a.horizon):
                        start=time.monotonic()
                        if step%client.ready['n_action_steps']==0:
                            display(viewer,keys,env,f'LIVE TRAINED VLA | trial {trial+1}/12 | predicting actions')
                            policy_obs=obs if coordinates is None else {**obs,'observation.state':coordinates.from_model(obs['observation.state']).astype(np.float32)}
                            prediction=client.predict(policy_obs,reset=step==0,seed=42+trial)
                            chunk=np.asarray(prediction['actions'],dtype=float);latencies.append(prediction['inference_seconds'])
                        raw=chunk[step%len(chunk)]
                        low,high=(env.limits[:,0],env.limits[:,1]) if coordinates is None else (coordinates.command_low,coordinates.command_high)
                        command=np.clip(raw,low,high)
                        # float rounding at a limit can land one ULP outside the model range
                        action=command if coordinates is None else np.clip(coordinates.to_model(command),env.limits[:,0],env.limits[:,1])
                        clipped+=int(np.any(np.abs(command-raw)>1e-6));raw_actions.append(raw)
                        before=obs
                        obs,_,success,truncated,info=env.step(action)
                        trace=trace_frame(env,info)
                        recorder.record(before,action,trace,env)
                        display(viewer,keys,env,f'LIVE TRAINED VLA | trial {trial+1}/12 | {step+1}/{a.horizon} steps')
                        time.sleep(max(0,1/30-(time.monotonic()-start)))
                        if success or truncated:break
                except Exception as exc:error=str(exc)
                result=recorder.finish(env,success,error)
                np.save(recorder.directory/'raw_policy_actions.npy',np.asarray(raw_actions,dtype=np.float32))
                final=np.asarray(result['final_object_xyz']) if result['final_object_xyz'] else np.zeros(3)
                result.update(attempt=trial,seed=meta['seed'],reference_layout=meta['reference_layout'],source_episode=source_index,
                              controller='trained_smolvla',checkpoint=client.ready['checkpoint'],policy_seed=42+trial,clipped_action_frames=clipped,
                              max_target_height=max((x['object_xyz'][2] for x in recorder.trace),default=0),
                              final_target_xy_distance_to_plate=float(np.linalg.norm(final[:2]-PLATES[meta['destination']])),
                              inference_seconds=latencies,sim_horizon_seconds=a.horizon/30)
                (recorder.directory/'metadata.json').write_text(json.dumps(result,indent=2)+'\n')
                results.append(result)
                (a.output/'summary.json').write_text(json.dumps({'complete':len(results)==12,'trials':results,
                    'successes':sum(r['accepted'] for r in results),'trials_completed':len(results),
                    'protocol':'One held-out scene per color/destination; 20 simulation seconds; checkpoint default action chunks; joint-limit clipping only; synchronous inference.'},indent=2)+'\n')
                print(json.dumps({k:result[k] for k in ('attempt','task','accepted','physical_grasp_verified','max_target_height','clipped_action_frames','error')}),flush=True)
                if error:raise RuntimeError(error)
            client.close();client=None
            print('Closed-loop evaluation complete. Replaying actual learned-policy actions.',flush=True)
            if a.loop_results:
                while viewer.is_running():
                    for row in results:
                        reset_episode(env,row)
                        with np.load(a.output/f"attempt-{row['attempt']:04d}"/'trajectory.npz') as saved:actions=saved['action']
                        for action in actions:
                            if not viewer.is_running():return
                            start=time.monotonic()
                            if coordinates is not None:action=coordinates.to_model(action)
                            _,_,done,truncated,_=env.step(np.clip(action,env.limits[:,0],env.limits[:,1]))
                            display(viewer,keys,env,f"RECORDED VLA ATTEMPT {row['attempt']+1}/12 | {'SUCCESS' if row['accepted'] else 'FAILED'}")
                            time.sleep(max(0,1/30-(time.monotonic()-start)))
                            if done or truncated:break
    finally:
        if client:client.close()
        env.close()

if __name__=='__main__':main()
