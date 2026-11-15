"""List Studio runtime sessions (and optionally stop them). Opus tool, I-51: Hassan handed the robot to Opus."""
import asyncio
import json
import sys
import urllib.request

import websockets

BASE = "127.0.0.1:7860"


async def list_sessions():
    async with websockets.connect(f"ws://{BASE}/api/runtime/sessions/ws", open_timeout=5) as ws:
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
    return msg.get("sessions", [])


def stop(session_id):
    req = urllib.request.Request(f"http://{BASE}/api/runtime/sessions/{session_id}/stop", method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status


if __name__ == "__main__":
    sessions = asyncio.run(list_sessions())
    print(json.dumps(sessions, indent=1)[:4000])
    if len(sys.argv) > 1 and sys.argv[1] == "--stop-all":
        for s in sessions:
            sid = s.get("id") or s.get("session_id")
            print("stopping", sid, "->", stop(sid))
