"""Paired, headless learned-policy smoke evaluation; no hardware or expert control."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from rollout_policy import PolicyClient
from sorting_env import BrickSortingEnv
from station_coordinates import StationCoordinates


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--python',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--contract',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=300)
    p.add_argument('--seed',type=int,default=20003)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    client=PolicyClient(None,a.output,a.checkpoint,local=(a.python,a.contract))
    env=BrickSortingEnv(render=True,horizon=a.steps);coordinates=StationCoordinates(env.limits)
    results=[]
    try:
        if client.ready.get('mapping_sha256')!=coordinates.mapping_sha256:raise ValueError('Mapping mismatch')
        for execute in (50,10):
            obs=env.reset(a.seed,target_color='red',destination='blue',reference=False)
            raw=[];latencies=[];success=False;clipped=0;heights=[];start=time.monotonic()
            for step in range(a.steps):
                if step % execute==0:
                    policy_obs={**obs,'observation.state':coordinates.from_model(obs['observation.state']).astype(np.float32)}
                    reply=client.predict(policy_obs,reset=step==0,seed=42)
                    chunk=np.asarray(reply['actions'],float);latencies.append(reply['inference_seconds'])
                    if chunk.shape != (50,6) or not np.isfinite(chunk).all():raise ValueError('Bad policy output')
                target=chunk[step % execute];raw.append(target)
                bounded=np.clip(target,coordinates.command_low,coordinates.command_high)
                clipped+=int(np.any(np.abs(bounded-target)>1e-6))
                obs,_,success,truncated,info=env.step(np.clip(coordinates.to_model(bounded),env.limits[:,0],env.limits[:,1]))
                heights.append(float(env.data.body(env.brick_body_name).xpos[2]))
                if success or truncated:break
            row=dict(execute_steps=execute,seed=a.seed,task=env.instruction,steps=step+1,success=bool(success),
                     clipped_frames=clipped,max_height_m=max(heights),inference_mean_s=float(np.mean(latencies)),wall_seconds=time.monotonic()-start)
            results.append(row);np.save(a.output/f'actions-{execute}.npy',raw)
            (a.output/'summary.json').write_text(json.dumps({'scope':'paired single-scene smoke; not a success-rate estimate','trials':results},indent=2)+'\n')
            print(json.dumps(row),flush=True)
    finally:
        client.close();env.close()

if __name__=='__main__':main()
