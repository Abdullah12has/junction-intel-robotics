"""Read-only wall-clock log of the live Studio session's joint state and action, for syncing the gripper camera.

Attaches to whichever follower session exists (re-attaching when it changes) and appends one CSV row per new
sample: wall time, the store's own stamps, state[6], action[6]. No serial access, no commands.
Stops after 4 hours or when /home/intel/junction-cell/recording/STOP exists.
"""
import asyncio, json, mmap, sys, time
from pathlib import Path
import numpy as np
import websockets
sys.path.insert(0, '/home/intel/Documents/physical-ai-studio/application/backend/src')
from runtime_session.store import FeatureSpec, ImageSpec, VECTOR_KEYS, _layout
from robots.features import sanitize_camera_name
import urllib.request
R = Path('/home/intel/junction-cell/recording'); P = 'ab767473-d6fd-47ff-812e-8239c455dabb'; F = 'e5600356-6296-496e-8344-17995e419baf'
N = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')
get = lambda r: json.load(urllib.request.urlopen(f'http://127.0.0.1:7860/api/projects/{P}/{r}', timeout=5))
END = time.time()+4*3600
out = (R/'joints.csv').open('a')
if out.tell() == 0: out.write('wall,session,'+','.join(f'{k}_stamp' for k in VECTOR_KEYS)+','+','.join(f'{k}{i}' for k in VECTOR_KEYS for i in range(6))+'\n')

async def session():
    async with websockets.connect('ws://127.0.0.1:7860/api/runtime/sessions/ws') as ws:
        s = [x for x in json.loads(await asyncio.wait_for(ws.recv(), 5))['sessions'] if x['project_id'] == P and F in x['robot_ids']]
    return s[0] if len(s) == 1 else None

def follow(s):
    cams = {sanitize_camera_name(c['name']): c for c in get('cameras') if c['id'] in s['camera_ids']}
    spec = FeatureSpec(tuple(n+'.pos' for n in N), {k: ImageSpec(c['payload']['height'], c['payload']['width']) for k, c in cams.items()})
    vectors, _, size = _layout(spec)
    paths = list(Path('/dev/shm').glob('pai-*-'+s['session_id'][:18]))
    if len(paths) != 1 or paths[0].stat().st_size != size: return 0
    rows = 0; last = None
    with paths[0].open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mem:
        while time.time() < END and not (R/'STOP').exists() and paths[0].exists():
            stamps = [np.frombuffer(mem, dtype=np.float64, count=1, offset=vectors[k].stamp_index*8)[0].item() for k in VECTOR_KEYS]
            if stamps != last:
                values = [np.ndarray(vectors[k].shape, vectors[k].dtype, buffer=mem, offset=vectors[k].offset).copy() for k in VECTOR_KEYS]
                again = [np.frombuffer(mem, dtype=np.float64, count=1, offset=vectors[k].stamp_index*8)[0].item() for k in VECTOR_KEYS]
                if again == stamps:
                    out.write(f'{time.time():.4f},{s["session_id"][:8]},'+','.join(f'{x:.4f}' for x in stamps)+','+','.join(f'{x:.3f}' for v in values for x in v)+'\n')
                    last = stamps; rows += 1
                    if rows % 150 == 0: out.flush()
            time.sleep(.008)
    out.flush(); return rows

while time.time() < END and not (R/'STOP').exists():
    try:
        s = asyncio.run(session())
        if s: print(time.strftime('%H:%M:%S'), 'session', s['session_id'][:8], s['environment_name'], 'rows', follow(s), flush=True)
    except Exception as e: print(time.strftime('%H:%M:%S'), 'retry:', repr(e)[:120], flush=True)
    time.sleep(1)
