"""Save one frame per RealSense through Studio's supported preview socket. Read-only."""
import asyncio, json, sys, urllib.parse, urllib.request
from pathlib import Path
import websockets
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
P = 'ab767473-d6fd-47ff-812e-8239c455dabb'
cams = [c for c in json.load(urllib.request.urlopen(f'http://127.0.0.1:7860/api/projects/{P}/cameras', timeout=5)) if c['driver'] == 'realsense' and c['fingerprint']['serial'] in sys.argv[2:]]
async def main():
    for c in cams:
        url = 'ws://127.0.0.1:7860/api/cameras/ws?camera=' + urllib.parse.quote(json.dumps(c))
        async with websockets.connect(url, max_size=None) as ws:
            frame = None
            for _ in range(5):
                frame = await asyncio.wait_for(ws.recv(), 10)
                if isinstance(frame, str): raise RuntimeError(frame)
            (out/(c['fingerprint']['serial']+'.jpg')).write_bytes(frame); print(c['name'], len(frame))
asyncio.run(main())
