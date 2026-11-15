# Robot station handoff

For Verda cloud CLI access on the Mac and the 15 November GPU availability snapshot, see [VERDA-CLI-HANDOFF.md](VERDA-CLI-HANDOFF.md). Credentials remain outside the repository.

Latest frozen-policy improvement work: [FROZEN-POLICY-IMPROVEMENT-PLAN.md](FROZEN-POLICY-IMPROVEMENT-PLAN.md). Includes shorter execution, wrist diagnostics, calibration gates and paired simulation evidence.

For laptop access, current IPs, SSH, tunnels and the 15 November service check, start with [INTEL-LAPTOP-ACCESS-HANDOFF.md](INTEL-LAPTOP-ACCESS-HANDOFF.md). It supersedes older connectivity/status observations below.

Updated: 2026-11-14, Europe/Helsinki. This is a snapshot; query live state before controlling hardware.

## Task and current progress

The user asked to set up the SO-101 arms using the official Junction guide, run teleoperation, connect the arm camera, and document the station for other agents.

Official guide: https://github.com/open-edge-platform/physical-ai-studio/blob/junction2026/JUNCTION2026.md

- Native Studio UI and backend are installed and running on the Intel laptop.
- Intel GPU computation passed an actual PyTorch XPU tensor calculation.
- Both arms have been calibrated by the people at the station; the original leader wrist-roll calibration problem was corrected.
- Teleoperation was started and verified through the UI switch and backend `switched to teleop control` log. Physical tracking accuracy was not independently verified.
- Arm camera `243722064316` was added to the saved environment. Teleoperation was paused to reload that environment; see the latest status appended below.
- No datasets were recorded, policies trained, Modal jobs started, or anomaly models installed during this setup.

## Hosts and software

| Item | Value |
|---|---|
| Local workspace | `/Users/abdullah/Desktop/junction` |
| Remote SSH | `intel@192.168.202.209` |
| Remote hostname | `intel-HP-OmniBook-Ultra-Laptop-14t-kd000` |
| Local SSH identity | `/Users/abdullah/.ssh/junction_intel_ed25519` |
| Remote repository | `/home/intel/Documents/physical-ai-studio` |
| Branch | `junction2026` |
| Observed commit | `17714488fa9c7d098a056da83f7c220cf983957d` |
| OS | Ubuntu 26.04.1, x86_64 |
| Backend | `http://127.0.0.1:7860` on laptop |
| UI | `http://localhost:3000` on laptop; same URL on Mac through SSH tunnel |
| uv | `/home/intel/.local/bin/uv` |
| Node installation | `/home/intel/.local/opt/node-v24.22.3-linux-x64` |

The remote tracked working tree was clean when checked. `.env` was copied from the backend example; no source patches or commits were made. Services were started with nohup, not installed as reboot-persistent system services.

The SSH key is dedicated to this station and has no passphrase. Do not copy its contents into notes. No laptop password is recorded here.

## Stable hardware identities

| Role | Serial | Last port | Studio ID |
|---|---|---|---|
| Leader | `5B90148378` | `/dev/ttyACM2` | `d361855b-0f89-4b7c-ab27-2c12d6341ca5` |
| Follower | `5B3D040939` | `/dev/ttyACM1` | `e5600356-6296-496e-8344-17995e419baf` |
| Arm camera, inferred from viewpoint changes | `243722064316` | RealSense D415 | `1d80a0dd-8f62-4f23-ae29-2f1fe406dde9` |
| Overhead camera, inferred from preview | `243622061421` | RealSense D415 | `b6e5da09-28f2-4208-9d15-526580074c75` |

Prefer `/dev/serial/by-id/usb-1a86_USB_Single_Serial_<SERIAL>-if00` over tty numbering. Leader role was inferred from ~5.2 V USB supply and follower from ~12.2 V external supply, matching Studio's setup checks. Both have six STS3215 motors, IDs 1–6, model number 777.

Both RealSense RGB streams passed frame acquisition at 640×480, 30 FPS. Their UVC entries appear in discovery too: do not add those as extra physical cameras. Arm camera preview currently shows mostly a laptop lid; it needs aiming at the gripper/work area. The other camera shows the table from above. Camera roles are visual inferences, not verified mounting labels.

## Project and environment

- Project: `Junction 2026`, ID `ab767473-d6fd-47ff-812e-8239c455dabb`.
- Environment: `SO101 Teleoperation`, ID `d4618b33-7572-4c4f-9d31-5ff7f17be899`.
- Follower configuration points to follower `e5600356…` and teleoperator leader `d361855b…`.
- Environment camera list now includes arm camera `1d80a0dd…`.
- UI: http://localhost:3000/projects/ab767473-d6fd-47ff-812e-8239c455dabb/environments/d4618b33-7572-4c4f-9d31-5ff7f17be899
- Chrome tab used for controls: `1118739288` (may be stale in a new session; rediscover it).

An older duplicate follower (`43baaf88-849c-49d5-af08-202f73f4f53a`) held stale calibration. It was absent from the UI at teleoperation setup. Always query current robots; do not restore the old entry. The first bad leader entry was also replaced.

## Calibration evidence and problems encountered

Initial leader homing failed on wrist-flex with `Magnitude 2056 exceeds 2047`. LeRobot resets calibration before writing offsets, so failed homing can leave partially updated hardware. The user subsequently repeated calibration.

The first saved leader wrist-roll range was only 2039–2048 (9 counts), with a live value of 2064. This was unsafe for normalized teleoperation. The user recalibrated; latest successful readings were:

| Joint | Leader range | Leader homing offset | Follower range | Follower homing offset |
|---|---|---|---|---|
| shoulder_pan | 924–3514 | -1794 | 891–3212 | -1727 |
| shoulder_lift | 777–3199 | -1363 | 822–3090 | 1829 |
| elbow_flex | 831–3020 | 1647 | 898–3102 | 1127 |
| wrist_flex | 877–3231 | 1949 | 812–3171 | 1916 |
| wrist_roll | 38–4082 | -751 | 74–3899 | -1404 |
| gripper | 2030–3321 | -53 | 2046–3577 | 304 |

These are diagnostic records, not values to blindly flash onto the motors. All leader readings were in range at the last direct check. Follower shoulder-lift read 820 (2 below minimum) and elbow 3103 (1 above maximum). Temperatures were 26–29°C leader and 33–35°C follower. All 12 motors responded.

During the first environment preview, leader gripper register/torque writes logged `comm=-6`. The subsequent clean reconnection had no such warnings before teleoperation started. Watch for recurrence. A `WebSocketDisconnect 1005` traceback appeared when closing a calibration wizard after successful save; do not confuse that old traceback with the current session state.

## Installed system changes

- Added user `intel` to `dialout`, `video`, `render`.
- Installed uv, managed Python 3.13, Node 24.22.3/npm 11.19.1, backend `uv sync --extra xpu`, UI `npm install`.
- Installed `libze1`, `libze-intel-gpu1`, `libturbojpeg0`, `ffmpeg`, `intel-opencl-icd`, `intel-ocloc` and their dependencies.
- GPU detection initially passed but execution aborted in `built_ins.cpp`; installing the OpenCL/compiler packages fixed the actual tensor calculation.
- Backend and UI remain loopback-bound. SSH forwarding was initially disabled by `00-hardening.conf`; a scoped `01-studio-forwarding.conf` permits user intel local forwarding only to `127.0.0.1:3000`.

## Remaining work

Verify the connected camera preview and aim it at the gripper/workspace. Confirm physical leader-to-follower tracking and the ability to stop at the station. Add overhead camera only when needed. Dataset collection, training credentials, Modal, inference and anomaly detection remain separate work.

Remote log wall times appear offset from API timestamps/client Helsinki time; use sequence and API UTC timestamps rather than assuming remote log clock matches local time.

## Camera-session follow-up

Adding the arm camera produced a visible feed, but two remote-browser sessions subsequently disconnected. Backend logged `Unexpected ASGI message 'websocket.send', after sending 'websocket.close'`; the session sampler reported stalls up to ~12 seconds. A teleoperation restart was acknowledged, then that session also closed. Do not interpret a cached image or switch as proof that a session is still live.

As a bandwidth mitigation, arm camera settings were changed from 640×480/30 FPS to **424×240/6 FPS**, a profile verified in the device's supported RGB profiles. The cause of the disconnects is not yet established; network backpressure is a hypothesis. This low-rate profile is for remote preview, not an approved training-data configuration. Restore suitable capture settings once local operation/network throughput is verified.

The local handoff and runbook are also copied to `/home/intel/junction-handoff/` for agents working on the laptop. The upstream repository's AGENTS.md is untouched.

### Latest verified state

The reduced-bandwidth camera connected successfully (`424x240 @ 6fps`) and its preview appears in the environment. Both arms connected without new initialization warnings. **Teleoperation is OFF** after the reconnect; it was deliberately not automatically restarted again after the two disconnects. Last observed runtime ID was `1abd852249b44c7f9e2b797465983923`; re-query it instead of assuming it persists. Long-duration camera/session stability and physical motion tracking remain unverified.

## Dashboard follow-up (2026-11-14)

The original Studio tab was subsequently observed with Teleoperate ON during dashboard development, superseding the OFF observation above. No motion command or session stop was issued during dashboard validation. Recheck live state before acting.

A separate simple dashboard is available on the Mac at http://localhost:3001/projects/ab767473-d6fd-47ff-812e-8239c455dabb/dashboard. Read [ROBOT-DASHBOARD.md](ROBOT-DASHBOARD.md) for controls, startup, source provenance and validation limits. It shares the backend through the existing port-3000 tunnel, but starts with its own empty session. It has not taken ownership of the arms. Original laptop repository/UI were not replaced.

## Third camera discovery (2026-11-14)

Detected two physical RealSense D415s plus the built-in HP 5MP webcam. HP RGB capture is `/dev/video0` (IR is a separate node); USB serial `200901010001`, bus `usb-0000:00:14.0-7`. Verified three frames at 640x480/30 with ffmpeg while the webcam was unowned. Added project camera **HP laptop webcam**, ID `25604944-56ed-48d2-adb9-440dd2067378`, using Studio-discovered USB fingerprint and default profile. It was not added to the active runtime/environment, so existing robot control was uninterrupted. Refresh camera configuration when no active session would be disrupted to see the new selection. No third external camera was detected. RealSense also appears through generic USB discovery; do not register that as an additional physical camera.

## Network change (2026-11-14)

Laptop IP is now `172.20.10.8` (previously `192.168.202.209`). SSH host identity verified against the saved old-IP key with `-o HostKeyAlias=192.168.202.209 -o StrictHostKeyChecking=yes`. Backend and UI returned HTTP 200; local port-3000 tunnel restored, and local dashboard port 3001 returned HTTP 200. No robot motion commands issued. Use the new IP and HostKeyAlias for subsequent SSH/SCP connections.

## Additional external camera detected (2026-11-14)

Third external camera now enumerates as Sonix `USB2.0_CAM1`, USB VID:PID `05a3:9230`, bus `usb-0000:00:14.0-3.4.2`, currently `/dev/video16`. Verified three frames at 640x480/30 FPS with ffmpeg after checking device ownership. Registered in Studio as **USB2.0_CAM1 external camera**, ID `4ff86d63-48bc-4a70-995d-b17aae723596`, using discovered fingerprint/profile; verified persisted configuration by GET. Total physical cameras: three external plus HP built-in. Not added to active runtime; no arm/session commands issued. Dashboard streaming for this camera remains unverified.

## Remote access setup (2026-11-14)

Current LAN IP verified by SSH is `10.0.41.74`; continue using `HostKeyAlias=192.168.202.209` with the dedicated SSH identity. Installed Tailscale from the official stable Ubuntu repository. The tailscaled service is enabled for boot and active. Account authentication is pending; no Tailscale address or cross-network access has been verified yet. Tailscale provides private authenticated network access, not a public IP. No robot session controls were changed.

### Tailscale authenticated and verified (2026-11-14)

Intel laptop Tailscale address: `100.112.85.77`. MacBook peer: `100.80.212.69`. Authenticated with a single-use key configured to expire on 2026-11-21; key contents were not saved here. Auth-key expiry is separate from device expiry. SSH from the Mac to `intel@100.112.85.77` passed with the dedicated identity and `HostKeyAlias=192.168.202.209`; backend health returned healthy through that SSH connection. Tailscaled is enabled at boot. Use this private Tailscale IP across networks while both devices are online and connected to Tailscale. Actual testing from a different physical network has not been performed. Studio remains loopback-bound; use SSH port forwarding for its UI. Existing LAN tunnels were not replaced.

### Dashboard tunnel switched to Tailscale (2026-11-14)

Installed and started Mac LaunchAgent `~/Library/LaunchAgents/com.junction.studio-tunnel.plist`, forwarding `127.0.0.1:3000` to laptop `127.0.0.1:3000` via SSH at `100.112.85.77`. Uses saved host identity, dedicated key, 15-second keepalives, three missed replies before exit, and launchd retry with 10-second throttle. Starts at Mac login. Verified tunnel health and dashboard `/api/health` return the same healthy backend instance; dashboard page returned HTTP 200. Dashboard stays at localhost:3001. Network-switch recovery is configured but not physically tested. No robot/session commands issued. See dashboard/runbook for updated connection commands.

### Hardware check awaiting leader USB (2026-11-14)

User requested arm/camera connection and a small movement test, and confirmed someone is at the station with clear workspace and power-disconnect readiness. Live discovery currently sees only follower serial `5B3D040939` at `/dev/ttyACM0`; leader `5B90148378` is absent. No motion or arm connection was initiated. Both RealSense serials passed three color frames at 424x240/6 FPS; HP webcam passed three frames at 640x480/30 FPS. External USB camera `/dev/video16` was owned by a Studio publisher worker and was not opened separately. Dashboard frontend had stopped and was restarted from the existing `studio-ui` directory on port 3001; dashboard opened with disconnected controls. Await leader USB reconnection, then recheck live ownership/session state before continuing. Stored tty paths are stale; resolve by serial before connection.

### Follower-only movement test (2026-11-14)

While leader USB remained absent, user explicitly requested moving the follower; station readiness had already been confirmed. Verified no active Studio runtime sessions and no serial owner before direct access. All six motors responded, live calibration matched saved values exactly, and positions were inside calibrated limits. Temperatures 32–35 C, supply approximately 12.2 V. All motors already had torque enabled; this was preserved.

Performed a small wrist_roll-only position ramp in 2-count increments every 150 ms, requesting +23 counts (about 2 degrees), then restoring its original goal. Measured wrist position 1675 -> 1694 -> 1682; outward travel 19 counts (1.67 degrees), final within 7 counts (0.62 degrees) of starting position. Restored original wrist goal 1678. The other five joints had zero measured change. No calibration, limits, speed, or torque settings changed. Serial connection closed without disabling torque. This confirms one wrist motor's physical encoder response, not full-arm or teleoperation validation. Leader connection and full camera/dashboard session test remain pending.

### Joystick dashboard and four-camera verification (2026-11-14)

Added bounded joint-jog controls to the Mac dashboard, default follower-only connection and all four cameras selected. Backend modified at `src/schemas/runtime_session.py`, `src/runtime_session/session.py`, plus `src/runtime_session/jog.py`; changes mirrored in local vendor checkout and `backend-patches/manual-jog.patch`. Frontend source synchronized between `studio-ui` (running) and `junction-intel-robotics/dashboard` (repository copy); OpenAPI regenerated. No motor calibration or limits modified.

Validated 13 backend jog boundary tests, 15 existing runtime-session tests, 7 UI control tests, TypeScript, targeted ESLint and production build. A brief wrist jog through the actual dashboard changed measured wrist roll from about -15.9 to -15.5 normalized units; release/Stop disarmed it. Other joint motion was not exercised through the new UI. Manual input is disabled initially and disarms on stale feedback, focus loss, page hide, Escape or error; each backend jog is finite and bounded.

All four feeds subsequently verified **Live** together with follower feedback: both RealSense D415s, HP webcam, USB2.0_CAM1. Overhead RealSense changed from 640x480/30 to 424x240/6 preview settings, matching the arm camera. Initial camera startup failed because old backend processes retained stale capture transport registrations; terminated hung old backends after graceful shutdown and started a clean backend. Current log is `/tmp/junction-backend-joystick-clean.log`; observed PID 166596 (re-query before action). Latest observed session `10c4a32fdf0a4f1f83583b6837d23701`, follower holding, joystick disabled, all four camera feeds live. Browser control tab observed 1118739400; rediscover if stale. Preserve this tab/session while using the dashboard. Do not create a competing arm session. Long-duration stability and off-site network loss remain unverified.

## NPU recovery (2026-11-14)

Missing user-space Intel NPU driver/compiler repaired with process-local official v1.38.0 libraries. Use `/home/intel/junction-cell/anomalib-env/bin/python-npu`; ordinary Python still needs the library paths. OpenVINO enumerated NPU and successfully compiled/inferred the existing PatchCore smoke model with finite outputs, execution device NPU. No firmware, kernel or Studio changes; sudo authentication failed so no system packages were installed. See [NPU-SETUP.md](NPU-SETUP.md) for paths, reproduction and limitations.

## System installation completed (2026-11-14)

After successful sudo authentication, installed `intel-driver-compiler-npu` and `intel-level-zero-npu` v1.38.0 plus six distribution dependencies. Waited for an independently initiated apt upgrade to finish; did not interrupt it or reboot. Plain Anomalib environment Python with `LD_LIBRARY_PATH` explicitly removed now lists CPU/GPU/NPU and executes the existing PatchCore smoke export on NPU with finite outputs. The `python-npu` wrapper is no longer required. This supersedes the process-local-only limitation above. Credentials were not saved. Defect accuracy and sustained performance remain unverified.

## Overhead recording-camera recovery (2026-11-14)

Recording environment repeatedly failed with overhead serial 243622061421 `CaptureTimeoutError: no publisher responded within timeout`. USB devices were present, but no publisher workers survived. Stopped the failed runtime via supported endpoint; sent TERM to old backend 166596. After it remained stuck waiting for background tasks with serial/video devices released, killed that exact old process and started the existing backend unchanged. New log `/tmp/junction-backend-recording-recovery.log`; observed PID 241038 (re-query).

Both RealSense cameras simultaneously delivered 286 frames each in roughly 10 seconds at 640x480/30 via librealsense. Overhead reports USB 2.1, arm USB 3.2. After restart, both Studio `/api/cameras/ws` streams delivered 30 valid JPEG frames each without timeout. Verification preview sockets were closed afterward. An attempted camera-only runtime test was rejected because runtime requires exactly one robot; no arms were loaded by that test. Full recording/encoding remains to be verified by the operator; no synthetic episodes were added.

## Three-camera blue-plate demonstrations (2026-11-14 evening)

See [BLUE-PLATE-DATA-CURATION.md](BLUE-PLATE-DATA-CURATION.md) for the latest recording and curation state. Six human episodes (16,943 frames) were finalized with both RealSense feeds plus USB2.0_CAM1; the original 987.7 MB dataset has a checksum-verified Mac backup. The separate cleaned copy has five episodes and 14,651 frames, imported as `e60f4087-d620-4f1d-885f-1620cefd54bd` (ALL-BLOCKS-BLUE - curated three cameras). Keep originals intact. Both RealSense cameras now report USB 3.2, superseding the USB 2.1 observation above. Studio was restarted through CLI; current logs are `/tmp/junction-backend-cli-restart.log` and `/tmp/junction-ui-cli-restart.log`. Recheck live processes/session ownership before acting.

## Recording dashboard recovery — 15 Nov 02:55 Helsinki

Restarted Studio backend/UI cleanly after stopping failed environment preview. Backend PID 2225270, UI 2225271 at that check; new logs `/tmp/junction-backend-recording-restart.log` and `/tmp/junction-ui-recording-restart.log`. Mac dashboard port 3001 had stopped; restarted from `studio-ui` with existing Node 24 and verified full rendered dashboard plus HTTP 200. Tunnel and backend health passed.

The three-camera failure was `No camera found matching` USB2.0_CAM1. USB discovery and `/dev/v4l/by-id` confirmed the Sonix wrist camera absent, even after operator reconnect. Old video16 now belongs to a RealSense node; do not remap by index alone. Both RealSense SDK serials are detected. Operator asked to try a different/direct USB port. Wrist recording remains blocked until it enumerates and its fingerprint is refreshed if needed. No camera calibration changed; no new recording started by this recovery.

Recovery verification: both RealSense cameras simultaneously started at 640x480/30 and each delivered five RGB frames; verification pipelines then closed. Wrist still absent after another-port attempt; kernel reports `usb 3-2.4-port3: connect-debounce failed` at 02:56:24. Port 3001 dashboard renders fully and is left disconnected, with only the two RealSense checkboxes selected. Saved recording environments were not changed. The optional `Top marked` MJPEG service on port 8090 was also not listening; it was not started because it exclusively owns the overhead device and would compete with raw RealSense recording. No recording was started.

### Wrist feed disconnect — 15 Nov 03:03 Helsinki

Kernel confirms Sonix USB2.0_CAM1 returned on USB 3-2.4.2 at 02:57:45, then disconnected at 03:02:35. Studio logged no frames at 03:02:36 and shut its UVC publisher down after five consecutive read failures. At 03:03, both lsusb and /dev/v4l/by-id show the wrist camera absent; both RealSense devices remain present. This is a USB-device disappearance, not merely a browser preview error. Possible cable/hub/power/camera fault remains unisolated. Saved wrist fingerprint still contains the previous bus/index; re-discover by serial after stable enumeration rather than assigning a guessed video number. No session, calibration or recording configuration changed during this diagnostic check.
