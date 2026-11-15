#!/bin/bash
# One-command build of a LeRobot v3.0 dataset from finished sort episodes.
# Usage: bash /home/intel/intel-junction-2026/app/vla/build.sh --runs RUN_DIR [RUN_DIR ...] --out OUT_DIR --variant team|full [--only-success] [--plates P [P ...]] [--max-dt SECONDS] [--zip]
# Runs build_dataset.py with the lerobot-env interpreter at nice 19, pinned to CPUs 0-3, one thread per numeric library.
# Caches stay under /home/intel/intel-junction-2026/work/P2-D-VLA1/cache.
set -eu
mkdir -p /home/intel/intel-junction-2026/work/P2-D-VLA1/cache/hf /home/intel/intel-junction-2026/work/P2-D-VLA1/cache/lerobot /home/intel/intel-junction-2026/work/P2-D-VLA1/cache/xdg /home/intel/intel-junction-2026/work/P2-D-VLA1/cache/torch
export HF_HOME=/home/intel/intel-junction-2026/work/P2-D-VLA1/cache/hf
export HF_LEROBOT_HOME=/home/intel/intel-junction-2026/work/P2-D-VLA1/cache/lerobot
export XDG_CACHE_HOME=/home/intel/intel-junction-2026/work/P2-D-VLA1/cache/xdg
export TORCH_HOME=/home/intel/intel-junction-2026/work/P2-D-VLA1/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec nice -n 19 taskset -c 0-3 /home/intel/intel-junction-2026/envs/lerobot-env/bin/python -I /home/intel/intel-junction-2026/app/vla/build_dataset.py "$@"
