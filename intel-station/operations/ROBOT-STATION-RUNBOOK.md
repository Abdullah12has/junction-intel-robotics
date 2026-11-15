# Robot station runbook

Read the handoff first. Commands below are examples for this station, not automatic permission to start physical movement in a new task.

## Connect from the Mac

```bash
ssh -i ~/.ssh/junction_intel_ed25519 -o IdentitiesOnly=yes intel@192.168.202.209
```

Check for an existing listener before creating another UI tunnel:

```bash
lsof -nP -iTCP:3000 -sTCP:LISTEN
ssh -fN -i ~/.ssh/junction_intel_ed25519 -o IdentitiesOnly=yes \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 127.0.0.1:3000:127.0.0.1:3000 intel@192.168.202.209
curl --noproxy '*' -fsS http://127.0.0.1:3000/api/health
```

Open http://localhost:3000. UI API base defaults to localhost:3000. The tunnel exposes the UI and its `/api` WebSocket/HTTP proxy. It does not require exposing backend port 7860 to the venue LAN.

## Start services on the Intel laptop, only if stopped

```bash
ss -ltnp | grep -E '3000|7860'

cd /home/intel/Documents/physical-ai-studio/application/backend
nohup /home/intel/.local/bin/uv run --extra xpu physicalai-studio serve \
  > /tmp/junction-backend.log 2>&1 < /dev/null &

cd /home/intel/Documents/physical-ai-studio/application/ui
export PATH="/home/intel/.local/opt/node-v24.22.3-linux-x64/bin:$PATH"
nohup npm run start -- --host 127.0.0.1 \
  > /tmp/junction-ui.log 2>&1 < /dev/null &
```

Do not overwrite `.env`, resync dependencies, restart active sessions, or start duplicate services without a reason. Stopping/reloading a runtime session can change motor torque state; coordinate with the person at the arms.

## Read-only diagnostics on laptop

```bash
lsusb
ls -l /dev/serial/by-id/
fuser /dev/ttyACM* 2>/dev/null
tail -60 /tmp/junction-backend.log
tail -30 /tmp/junction-ui.log
curl -fsS http://127.0.0.1:7860/api/health
curl -fsS http://127.0.0.1:7860/api/hardware/cameras
curl -fsS http://127.0.0.1:7860/api/robots/catalog/SO101_Follower/discover
curl -fsS http://127.0.0.1:7860/api/projects/ab767473-d6fd-47ff-812e-8239c455dabb/robots
curl -fsS http://127.0.0.1:7860/api/projects/ab767473-d6fd-47ff-812e-8239c455dabb/environments
```

When Studio owns a serial port, do not connect a separate LeRobot bus. With no owner and no active session, diagnostic reads used `FeetechMotorsBus.connect(handshake=False)`, `read_calibration`, `sync_read('Present_Position', normalize=False)`, voltage and temperature reads. Disconnect with `disable_torque=False` for a read-only diagnostic so it does not silently change torque. Do not invoke homing/reset/configuration methods during a check.

## Live sessions and stopping

Preferred: switch **Teleoperate off** in the existing environment tab. It stops following; do not assume that means motors are de-energized. Studio's runtime-session menu also offers session stop. Keep the active control tab open while using it.

The read-only WebSocket `ws://127.0.0.1:7860/api/runtime/sessions/ws` sends current sessions on connection. To inspect from the backend directory:

```bash
.venv/bin/python - <<'PY'
import asyncio
import websockets
async def main():
    async with websockets.connect('ws://127.0.0.1:7860/api/runtime/sessions/ws') as ws:
        print(await asyncio.wait_for(ws.recv(), 5))
asyncio.run(main())
PY
```

For a requested stop, get the current session ID first, then:

```bash
curl -fsS -X POST http://127.0.0.1:7860/api/runtime/sessions/SESSION_ID/stop
```

Never reuse an old session ID from a log. The session stop endpoint tears down device connections. If software control fails during dangerous motion, the person at the station must use the available physical power disconnect.

## Environment and camera changes

Environment schema is in `application/backend/src/schemas/environment.py`; update endpoint is `PUT /api/projects/{project_id}/environments/{environment_id}`. Fetch the environment from the list endpoint, preserve its robot mapping and other cameras, and append the requested camera ID only if absent. The detail endpoint returns expanded relations, so do not blindly PUT its response as the write schema.

Pause teleoperation before reloading an environment. Refresh the existing control tab after saving configuration; verify camera preview and device status before resuming authorized teleoperation. Adding an environment camera does not attach it physically or establish its viewpoint.

## Teleoperation protocol, for diagnosis

UI environment pages manage a WebSocket at `/api/projects/{project_id}/runtime/ws?hz=20`. Commands include `load_environment` and `set_control`; use the supported UI to leave the user visible start/stop controls. Source of truth: `application/backend/src/schemas/runtime_session.py` and `application/ui/src/features/runtime-session/cells/robot-cell.tsx`.

At the observed commit the UI requests `{type: 'teleop', hz: 100, grace_s: 2}`. The code comment says 3 seconds, but the actual UI value is 2. Do not rely on the comment. Loading robots configures registers and enables follower torque even before teleoperation is enabled.

## Logs and temporary files

- `/tmp/junction-backend.log`, `/tmp/junction-ui.log`: service logs.
- `/tmp/junction-backend-install.log`, `/tmp/junction-ui-install.log`: installation logs.
- `/tmp/junction-system-install.log`, `/tmp/junction-gpu-install.log`: apt installation logs.
- Camera snapshots were captured under `/tmp/arm-camera-<serial>.jpg` on laptop and Mac. They contain people/workspace imagery; do not publish them.
- No permanent automated calibration script was installed.
