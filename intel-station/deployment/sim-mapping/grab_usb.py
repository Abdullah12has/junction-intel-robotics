"""Save one frame from each non-RealSense project camera via Studio's preview socket. Read-only."""
import asyncio, json, sys, urllib.parse, urllib.request
from pathlib import Path
import websockets
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
P = 'ab767473-d6fd-47ff-812e-8239c455dabb'
cams = [c for c in json.load(urllib.request.urlopen(f'http://127.0.0.1:7860/api/projects/{P}/cameras', timeout=5)) if c['driver'] != 'realsense']
async def main():
    for c in cams:
        try:
            async with websockets.connect('ws://127.0.0.1:7860/api/cameras/ws?camera=' + urllib.parse.quote(json.dumps(c)), max_size=None) as ws:
                for _ in range(5):
                    frame = await asyncio.wait_for(ws.recv(), 10)
                    if isinstance(frame, str): raise RuntimeError(frame)
                (out/(c['id'][:8]+'.jpg')).write_bytes(frame); print(c['name'], c['id'][:8], len(frame))
        except Exception as e: print(c['name'], 'FAILED', repr(e)[:150])
asyncio.run(main())
