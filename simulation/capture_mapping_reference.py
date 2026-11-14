"""Read the existing Studio shared store without connecting to or commanding devices.

Run on the Intel laptop with the backend venv. No calibration/register writes.
"""
import argparse
import asyncio
import hashlib
import json
import mmap
from pathlib import Path
import sys
import time
import urllib.request

import numpy as np
from PIL import Image
import websockets

BACKEND=Path('/home/intel/Documents/physical-ai-studio/application/backend')
sys.path.insert(0,str(BACKEND/'src'))
from runtime_session.store import FeatureSpec, ImageSpec, _layout
from robots.features import sanitize_camera_name

PROJECT='ab767473-d6fd-47ff-812e-8239c455dabb'
FOLLOWER='e5600356-6296-496e-8344-17995e419baf'
NAMES=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')

def get(resource):return json.load(urllib.request.urlopen(f'http://127.0.0.1:7860/api/projects/{PROJECT}/{resource}',timeout=5))

async def capture(output):
    async with websockets.connect('ws://127.0.0.1:7860/api/runtime/sessions/ws') as ws:
        sessions=json.loads(await asyncio.wait_for(ws.recv(),5))['sessions']
    matching=[s for s in sessions if s['project_id']==PROJECT and FOLLOWER in s['robot_ids']]
    if len(matching)!=1:raise RuntimeError('Need one existing Studio follower session; no session will be created')
    session=matching[0]
    robot=next(r for r in get('robots') if r['id']==FOLLOWER)
    if robot['payload']['unit']!='normalized':raise ValueError('Expected normalized Studio state')
    cameras={sanitize_camera_name(c['name']):c for c in get('cameras') if c['id'] in session['camera_ids']}
    spec=FeatureSpec(tuple(n+'.pos' for n in NAMES),{k:ImageSpec(c['payload']['height'],c['payload']['width']) for k,c in cameras.items()})
    vectors,images,size=_layout(spec)
    paths=list(Path('/dev/shm').glob('pai-*-'+session['session_id'][:18]))
    if len(paths)!=1 or paths[0].stat().st_size!=size:raise RuntimeError('Shared store does not match live camera/joint layout')
    output.mkdir(parents=True,exist_ok=False)
    samples=[]
    with paths[0].open('rb') as file,mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ) as memory:
        def read(slot):
            for _ in range(10):
                before=np.frombuffer(memory,dtype=np.float64,count=1,offset=slot.stamp_index*8)[0].item()
                array=np.ndarray(slot.shape,slot.dtype,buffer=memory,offset=slot.offset).copy()
                after=np.frombuffer(memory,dtype=np.float64,count=1,offset=slot.stamp_index*8)[0].item()
                if before==after and np.isfinite(after) and 0<=time.monotonic()-after<.5:return array,after
            raise RuntimeError('Stale or changing shared-store sample')
        for _ in range(30):
            values,stamp=read(vectors['state'])
            if not np.isfinite(values).all():raise ValueError('Non-finite joint state')
            samples.append({'state':values.tolist(),'monotonic_timestamp':stamp})
            await asyncio.sleep(1/30)
        camera_times={}
        for name,slot in images.items():
            values,stamp=read(slot);Image.fromarray(values).save(output/(name+'.png'));camera_times[name]=stamp
    values=np.array([s['state'] for s in samples])
    calibration=robot['payload']['calibration']
    summary={'capture_kind':'unlabelled_reference','read_only':True,'serial_device_opened':False,
             'session_id':session['session_id'],'robot_serial':robot['payload']['serial_number'],'joint_order':list(NAMES),
             'calibration':calibration,'calibration_sha256':hashlib.sha256(json.dumps(calibration,sort_keys=True).encode()).hexdigest(),
             'mean_normalized':values.mean(axis=0).tolist(),'range_normalized':np.ptp(values,axis=0).tolist(),
             'stable_within_0_3_normalized':bool(np.max(np.ptp(values,axis=0))<.3),
             'samples':samples,'camera_timestamps':camera_times,'physical_model_angles_rad':None}
    (output/'capture.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('samples','calibration')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    asyncio.run(capture(a.output))
