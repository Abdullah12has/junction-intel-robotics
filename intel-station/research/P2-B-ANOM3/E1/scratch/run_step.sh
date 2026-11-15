#!/bin/bash
# usage: run_step.sh LOG SCRIPT [ARGS...]
# Runs one anom3 Python step on the laptop: CPU only, nice -n 19, OMP_NUM_THREADS=4, python -I -B.
# stdout and stderr go to LOG; the last line of LOG is EXIT=<code>.
LOG="$1"
shift
. /home/intel/intel-junction-2026/envs/env.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd /home/intel/intel-junction-2026/work/P2-B-ANOM3/E1 || exit 1
nice -n 19 "$PY" -I -B "$@" > "$LOG" 2>&1
echo "EXIT=$?" >> "$LOG"
