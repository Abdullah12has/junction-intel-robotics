"""Read-only: log follower state + both RealSense frames for camera fitting.

State and the session's camera come from the live session's shared store; the
overhead camera (not in this session) comes from Studio's preview socket.
No serial access, no commands.  usage: sweep.py OUT SECONDS
"""
import asyncio, json, mmap, sys, time, urllib.parse, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
import websockets
sys.path.insert(0, '/home/intel/Documents/physical-ai-studio/application/backend/src')
from runtime_session.store import FeatureSpec, ImageSpec, _layout
from robots.features import sanitize_camera_name
P = 'ab767473-d6fd-47ff-812e-8239c455dabb'; F = 'e5600356-6296-496e-8344-17995e419baf'; OVERHEAD = '243622061421'
N = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')
get = lambda r: json.load(urllib.request.urlopen(f'http://127.0.0.1:7860/api/projects/{P}/{r}', timeout=5))

async def main(out, seconds):
    async with websockets.connect('ws://127.0.0.1:7860/api/runtime/sessions/ws') as ws:
        sessions = json.loads(await asyncio.wait_for(ws.recv(), 5))['sessions']
    match = [s for s in sessions if s['project_id'] == P and F in s['robot_ids']]
    if len(match) != 1: raise RuntimeError('Need exactly one existing follower session')
    session = match[0]; allcams = get('cameras')
    cams = {sanitize_camera_name(c['name']): c for c in allcams if c['id'] in session['camera_ids']}
    spec = FeatureSpec(tuple(n+'.pos' for n in N), {k: ImageSpec(c['payload']['height'], c['payload']['width']) for k, c in cams.items()})
    vectors, images, size = _layout(spec)
    paths = list(Path('/dev/shm').glob('pai-*-'+session['session_id'][:18]))
    if len(paths) != 1 or paths[0].stat().st_size != size: raise RuntimeError('Shared store layout mismatch')
    over = next(c for c in allcams if c.get('fingerprint', {}).get('serial') == OVERHEAD)
    if over['id'] in session['camera_ids']: raise RuntimeError('Overhead is in the session; read it from the store instead')
    out.mkdir(parents=True, exist_ok=False); rows = []
    url = 'ws://127.0.0.1:7860/api/cameras/ws?camera=' + urllib.parse.quote(json.dumps(over))
    with paths[0].open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mem:
        def read(slot):
            for _ in range(20):
                a = np.frombuffer(mem, dtype=np.float64, count=1, offset=slot.stamp_index*8)[0].item()
                arr = np.ndarray(slot.shape, slot.dtype, buffer=mem, offset=slot.offset).copy()
                b = np.frombuffer(mem, dtype=np.float64, count=1, offset=slot.stamp_index*8)[0].item()
                if a == b: return arr
            raise RuntimeError('Unstable shared-store read')
        async with websockets.connect(url, max_size=None) as ws:
            end = time.monotonic() + seconds; i = 0; last = 0
            while time.monotonic() < end:
                frame = await asyncio.wait_for(ws.recv(), 10)
                if isinstance(frame, str): raise RuntimeError(frame)
                if time.monotonic() - last < .2: continue   # keep ~5 Hz
                last = time.monotonic()
                before = read(vectors['state'])
                for name, slot in images.items(): Image.fromarray(read(slot)).save(out/f'{i:04d}-close.jpg', quality=85)
                after = read(vectors['state'])
                (out/f'{i:04d}-overhead.jpg').write_bytes(frame)
                rows.append({'i': i, 'state': after.tolist(), 'moved': float(np.max(np.abs(after-before)))}); i += 1
    (out/'states.json').write_text(json.dumps(rows))
    s = np.array([r['state'] for r in rows]); print(len(rows), 'frames; state min', np.round(s.min(0), 1).tolist(), 'max', np.round(s.max(0), 1).tolist())

asyncio.run(main(Path(sys.argv[1]), float(sys.argv[2])))
