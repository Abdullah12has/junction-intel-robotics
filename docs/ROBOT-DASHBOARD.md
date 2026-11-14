# Robot dashboard

## Open it

[Open dashboard](http://localhost:3001/projects/ab767473-d6fd-47ff-812e-8239c455dabb/dashboard) on the Mac.

This is a small extension of the existing [Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio), Apache-2.0, copied from the laptop's `junction2026` branch at `17714488fa9c7d098a056da83f7c220cf983957d`. The repository copy is in [`../dashboard/`](../dashboard/README.md); the upstream license is preserved there. Existing robot drivers, runtime commands, component library and camera renderer are reused. No second hardware-control stack was installed.

Other open-source options reviewed: [Lichtblick](https://github.com/lichtblick-suite/lichtblick) and [OnlineRobotController](https://github.com/Soappyooo/OnlineRobotController). Extending the already configured Studio avoids extra adapters and competing serial-device ownership.

## Controls

1. Stop the old Studio runtime session using the runtime-session button in the footer when ready to switch. Only one session can own an arm. The dashboard does not steal the old session.
2. Choose the environment and cameras, then **Connect arms & selected cameras**. Connecting enables holding torque; it does not start following.
3. Enable **Follow leader** to start teleoperation. **Stop following** stops tracking and holds the current pose; it is not a physical emergency stop.
4. **Disconnect arms & cameras** releases the runtime session. Closing the controlling tab also releases its session.

The page shows follower measurements, leader-derived command targets while following, differences, device status and selected camera previews. Previews/telemetry are requested at 2 Hz; teleoperation control runs at 100 Hz. Samples older than 2 seconds are hidden; camera frames older than 2.5 seconds are marked stale. A page load/reconnect never automatically connects the arms. Camera selection applies to this session and is editable before connecting.

## Start on the Mac

Use Node >=24.2 and npm >=11.14. Existing local Node 24 binary is `/Users/abdullah/.npm/_npx/387698761821791d/node_modules/node/bin/node`; it is an npm cache path and may disappear if caches are cleaned.

The Mac's login service `com.junction.studio-tunnel` forwards localhost:3000 to the laptop through Tailscale (`100.112.85.77`). It starts at login and retries after connection failures. Keep Tailscale connected on both machines. Its configuration is `~/Library/LaunchAgents/com.junction.studio-tunnel.plist`; errors go to `~/Library/Logs/junction-studio-tunnel.log`.

For manual operation only, when the service is unloaded and port 3000 is free:

```sh
ssh -N -o HostKeyAlias=192.168.202.209 -o StrictHostKeyChecking=yes -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=10 -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -i ~/.ssh/junction_intel_ed25519 -L 127.0.0.1:3000:127.0.0.1:3000 intel@100.112.85.77
```

In another terminal:

```sh
cd /Users/abdullah/Desktop/junction/junction-intel-robotics/dashboard
# Install dependencies only if needed: npm ci
PUBLIC_API_PROXY_TARGET=http://127.0.0.1:3000 PUBLIC_API_BASE_URL=http://localhost:3001 npm run start -- --host 127.0.0.1 --port 3001
```

Frontend and tunnel are loopback-bound. This URL is for the Mac, not other computers. The dashboard frontend remains a development process; the SSH tunnel is a login service. Backend startup remains documented in ROBOT-STATION-RUNBOOK.md.

## Files and validation

Feature: `dashboard/src/features/station-dashboard/`. Routing changes: `src/router.tsx`, `src/paths.ts`, robots tab navigation and project navigation. RuntimeSessionProvider now accepts an optional observation-rate prop; other pages retain their existing default.

A staging copy exists on the laptop at `/home/intel/junction-dashboard-ui`, with dependencies symlinked to the original UI. It is not the running production UI. Do not replace the original UI while a hardware session is active: a reload can release the arms.

Validation completed: TypeScript check, targeted ESLint, production build, 11 tests across dashboard/runtime-provider/project-navigation, and visible browser render against the real backend. Tests cover explicit connection, blocked start on stale telemetry, retained stop action, joint values and stale expiry. Live arm start/stop and camera feeds through this new dashboard have not been exercised because the original UI was actively teleoperating. Existing camera-session disconnect/backpressure issue remains unresolved; see ROBOT-STATION-HANDOFF.md.

## Repository packaging (2026-11-14)

Source, tests, generated API snapshots, lockfile and licence are copied into `dashboard/`. Dependencies and build artifacts are excluded. The original `studio-ui` working directory remains untouched so the existing running process is not disrupted; use the repository copy for future changes.
