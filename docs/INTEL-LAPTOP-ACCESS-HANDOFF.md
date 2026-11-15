# Intel laptop access handoff

Last checked **15 November 2026, 00:54 Helsinki**. This is a real robot station. Read [ROBOT-STATION-HANDOFF.md](ROBOT-STATION-HANDOFF.md) and [ROBOT-STATION-RUNBOOK.md](ROBOT-STATION-RUNBOOK.md) before controlling devices.

## Addresses and live status

| Item | Value |
|---|---|
| Laptop user | `intel` |
| Hostname | `intel-HP-OmniBook-Ultra-Laptop-14t-kd000` |
| Tailscale IP (preferred) | `100.112.85.77` |
| Current Wi-Fi/LAN IP | `10.0.41.74` (may change with DHCP/network) |
| This Mac's Tailscale IP | `100.80.212.69` |
| SSH | Port 22; successful at last check |
| Studio UI on laptop | `127.0.0.1:3000`; listening at last check |
| Studio backend on laptop | `127.0.0.1:7860`; **not listening at last check**, health request failed |
| Mac Studio tunnel | `127.0.0.1:3000`; existing SSH listener at last check |
| Temporary alternate tunnel | Mac port 3002 was used during troubleshooting; **not listening at last check** |

Older addresses `192.168.202.209` and `172.20.10.8` are historical. The former remains the SSH host-key alias, not the current destination. Tailscale and LAN connections were intermittently dropping during earlier recordings. A successful UI page load does not prove the backend or runtime WebSocket is healthy.

## SSH from this Mac

Private key already exists locally at `/Users/abdullah/.ssh/junction_intel_ed25519`. Do not copy its contents into this repo or messages. No password is needed for this key-based connection.

```bash
ssh -i ~/.ssh/junction_intel_ed25519 \
  -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=5 \
  -o ServerAliveInterval=10 -o ServerAliveCountMax=3 \
  -o HostKeyAlias=192.168.202.209 -o StrictHostKeyChecking=yes \
  intel@100.112.85.77
```

If Tailscale stalls and both machines share the venue network, replace only the destination with `intel@10.0.41.74`. Discover current connectivity with `tailscale status` and `tailscale ping --timeout=5s --c=2 100.112.85.77`. Do not delete known-host entries or disable verification to bypass a host-key mismatch. An agent on another machine needs its own authorized access; the key path above is specific to this Mac.

## Browser and tunnel

On the Intel laptop, open `http://localhost:3000`. On the Mac, that URL reaches the laptop only through SSH forwarding. The Mac launch service `com.junction.studio-tunnel` normally maintains the port-3000 tunnel.

First inspect `lsof -nP -iTCP:3000 -sTCP:LISTEN`. Only if no tunnel is listening, create one:

```bash
ssh -fN -i ~/.ssh/junction_intel_ed25519 \
  -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=5 \
  -o HostKeyAlias=192.168.202.209 -o StrictHostKeyChecking=yes \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:3000:127.0.0.1:3000 intel@100.112.85.77
```

Check `curl --max-time 5 http://localhost:3000/api/health` on the Mac and `curl --max-time 5 http://127.0.0.1:7860/api/health` on the laptop. Port 3002, if needed, is simply an alternate local forwarding port; keep all robot control in one active browser session. Prefer the laptop's own browser when Wi-Fi disrupts live camera previews.

## Studio locations and startup

- Repository: `/home/intel/Documents/physical-ai-studio` (preserve its upstream `AGENTS.md`).
- Backend: `application/backend`, existing `.venv/bin/physicalai-studio` and `.venv/bin/python`.
- UI: `application/ui`, Node at `/home/intel/.local/opt/node-v24.22.3-linux-x64/bin`.
- Latest restart logs: `/tmp/junction-backend-cli-restart.log`, `/tmp/junction-ui-cli-restart.log`.
- Older logs: `/tmp/junction-backend-recording-recovery.log`, `/tmp/junction-backend.log`, `/tmp/junction-ui.log`. Check timestamps; older logs can mislead.
- Data root: `/home/intel/.local/share/physicalai/`; finalized datasets in `datasets/`, active recording mutations in `cache/`, training snapshots in `snapshots/`.
- Other agent's work: `/home/intel/intel-junction-2026/`; station tools/recording also under `/home/intel/junction-cell/`. Do not kill their processes indiscriminately.

Inspect `ss -ltnp` and existing processes before startup. If authorized to start an absent backend, use the existing environment rather than resyncing dependencies:

```bash
cd /home/intel/Documents/physical-ai-studio/application/backend
nohup .venv/bin/physicalai-studio serve > /tmp/junction-backend-restart.log 2>&1 < /dev/null &
```

If the UI is absent:

```bash
cd /home/intel/Documents/physical-ai-studio/application/ui
export PATH="/home/intel/.local/opt/node-v24.22.3-linux-x64/bin:$PATH"
nohup npm run start -- --host 127.0.0.1 > /tmp/junction-ui-restart.log 2>&1 < /dev/null &
```

Do not run `uv sync`, overwrite `.env`, reset calibration, or restart an active recording as routine troubleshooting. GPU PyTorch dependencies were repaired separately; upstream pins may undo that work. Preserve running process environment when restarting a configured service. PID values change; identify them live.

## Runtime ownership and devices

List runtime sessions read-only from the backend directory:

```bash
.venv/bin/python - <<'PY'
import asyncio, websockets
async def main():
    async with websockets.connect('ws://127.0.0.1:7860/api/runtime/sessions/ws') as ws:
        print(await asyncio.wait_for(ws.recv(), 5))
asyncio.run(main())
PY
```

For an authorized stop, use `POST /api/runtime/sessions/<LIVE_SESSION_ID>/stop`. Closing a recording session finalizes its dataset cache and stops its control workers; confirm the operator has finished first. Never delete or manually finalize a live cache. Loading an environment can enable motor torque; teleoperation is a separate control.

Use `fuser -v /dev/ttyACM* /dev/video*` and stable `/dev/serial/by-id/` / `/dev/v4l/by-id/` names. Never open an arm serial device already owned by Studio. Camera node numbers can change.

| Device | Identity |
|---|---|
| Follower | Serial `5B3D040939`; Studio robot `e5600356-6296-496e-8344-17995e419baf` |
| Leader | Serial `5B90148378`; Studio robot `d361855b-0f89-4b7c-ab27-2c12d6341ca5` |
| Overhead | RealSense `243622061421`; camera `b6e5da09-28f2-4208-9d15-526580074c75` |
| Fixed close view | RealSense `243722064316`; camera `1d80a0dd-8f62-4f23-ae29-2f1fe406dde9` |
| Actual wrist camera | `USB2.0_CAM1`, VID:PID `05a3:9230`; camera `4ff86d63-48bc-4a70-995d-b17aae723596` |

Both RealSense cameras were verified on USB 3.2 after an overhead first-frame timeout on USB 2.1. Do not move the calibrated camera mounts while changing cables. The separate FFmpeg wrist recorder was stopped via `/home/intel/junction-cell/recording/STOP` so Studio could own the wrist camera; do not restart it while Studio uses that device.

## Project and datasets

- Project: `ab767473-d6fd-47ff-812e-8239c455dabb` (Junction 2026).
- Three-camera environment: `3507e384-4abf-488d-be91-0d235fee2182`.
- Original six human episodes: `8a8bb05e-6f2c-44f5-bf4b-745c2028f7cf`.
- Curated five-episode copy: `e60f4087-d620-4f1d-885f-1620cefd54bd`.
- Recording URL on the laptop: `http://localhost:3000/projects/ab767473-d6fd-47ff-812e-8239c455dabb/datasets/8a8bb05e-6f2c-44f5-bf4b-745c2028f7cf/record`.
- Read-only API: `GET /api/projects/<PROJECT_ID>`, `GET /api/dataset/<DATASET_ID>/episodes`. The project response lists its datasets; `/api/projects/<PROJECT_ID>/datasets` is not a supported route in this version.

See [BLUE-PLATE-DATA-CURATION.md](BLUE-PLATE-DATA-CURATION.md) for backups, counts, review and curation. See [SMOLVLA-SIM-EVAL.md](SMOLVLA-SIM-EVAL.md) for the **separate local Mac simulator**. Running that simulator does not require connecting to the laptop. See [VERDA-TRAINING.md](VERDA-TRAINING.md) for model/checkpoint history; do not recreate billed GPU compute just to access this laptop.

The latest user Git preference is to leave commits local: do not push without a new request. Credentials and private keys must remain outside documentation and Git.
