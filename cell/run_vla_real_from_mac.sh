#!/usr/bin/env bash
# One bounded SmolVLA trial on the real arm: policy on this Mac (MPS), arm and cameras on the Intel laptop.
# Stop early: ssh to the laptop and `touch ~/junction-cell/real-vla/STOP`.
set -euo pipefail
SECONDS_LIMIT=${1:-30}
: "${CAMERA1:?Set CAMERA1 to the verified overhead by-id path on the laptop}"
: "${CAMERA2:?Set CAMERA2 to the verified fixed-close by-id path on the laptop}"
[[ "$SECONDS_LIMIT" =~ ^[0-9]+$ ]] || exit 2
[[ "$CAMERA1" =~ ^/dev/v4l/by-id/[a-zA-Z0-9_.-]+$ && "$CAMERA2" =~ ^/dev/v4l/by-id/[a-zA-Z0-9_.-]+$ ]] || exit 2
SPEED=${2:-10}
[[ "$SPEED" =~ ^[0-9]+$ ]] || exit 2
MODEL=/Users/abdullah/Desktop/junction/station-backups/2026-11-15-smolvla-cotrain/run-01-step015000
HERE=$(cd "$(dirname "$0")" && pwd)
D=$(mktemp -d); F="$D/pipe"; mkfifo "$F"; trap 'rm -rf "$D"' EXIT
/Users/abdullah/Desktop/junction/.venv-vla/bin/python -u "$HERE/../simulation/policy_worker.py" --checkpoint "$MODEL/pretrained_model" \
  --contract "$MODEL/contract.json" --device mps < "$F" 2> /tmp/vla-real-worker.err | \
  ssh -i ~/.ssh/junction_intel_ed25519 -o IdentitiesOnly=yes -o HostKeyAlias=192.168.202.209 -o StrictHostKeyChecking=yes -o BatchMode=yes \
    intel@100.112.85.77 "cd ~/junction-cell/real-vla && ~/intel-junction-2026/envs/motion-env/bin/python -u run_vla_real.py --policy-stdio --seconds $SECONDS_LIMIT --max-speed $SPEED --execute-steps 10 --camera1 $CAMERA1 --camera2 $CAMERA2" > "$F"
