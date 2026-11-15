#!/bin/bash
# E1 P2-D-BUILD: create envs/lerobot-env pinned to lerobot 0.6.0 (uv.lock of physical-ai-studio backend)
set -u
export UV_CACHE_DIR=/home/intel/intel-junction-2026/.cache/uv
UV=/home/intel/.local/bin/uv
W=/home/intel/intel-junction-2026/work/P2-D-BUILD/E1
ENV=/home/intel/intel-junction-2026/envs/lerobot-env
LOCK=/home/intel/intel-junction-2026/repos/physical-ai-studio/application/backend/uv.lock
echo "start $(date -u +%FT%TZ)"
python3 -I "$W/tools/mkconstraints.py" "$LOCK" "$W/tools/constraints.txt" || exit 1
[ -x "$ENV/bin/python" ] || $UV venv --python 3.13 "$ENV" || exit 2
if $UV pip install --python "$ENV/bin/python" --torch-backend cpu -c "$W/tools/constraints.txt" "lerobot[dataset,feetech]==0.6.0"; then
  echo "constrained install ok"
else
  echo "constrained install failed; retrying with lerobot==0.6.0 only"
  $UV pip install --python "$ENV/bin/python" --torch-backend cpu "lerobot[dataset,feetech]==0.6.0" || exit 3
fi
echo "done $(date -u +%FT%TZ)"
