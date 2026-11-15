# Junction robot station: start here

Read [ROBOT-STATION-HANDOFF.md](ROBOT-STATION-HANDOFF.md) for hardware, session state, calibration issues and remaining work. Read [ROBOT-STATION-RUNBOOK.md](ROBOT-STATION-RUNBOOK.md) for connection, startup, diagnostics and stopping.

- These documents describe a real SO-101 station, not a simulator. Check live state before acting; session IDs and USB port numbers change.
- Do not open a serial device already owned by Studio. Use `fuser` first and prefer Studio's session controls.
- Never substitute guessed calibration or bypass motor limits. Calibration requires physical manipulation at the station.
- Keep credentials and private keys out of documentation and Git. The handoff contains only the local key path.
- Preserve the upstream repository's AGENTS.md on the Intel laptop. This file applies to this local Junction workspace.
- For TalentAdore repositories, verify the TalentAdore author and committer identity before committing. Never add AI attribution trailers. Never use `codex` in branch names.
- If the user invokes `/graphify`, read the installed graphify skill before doing anything else.

For the simple dashboard, read [ROBOT-DASHBOARD.md](ROBOT-DASHBOARD.md). Its local UI runs on port 3001 and shares the existing backend; do not connect competing sessions to the same arms.
