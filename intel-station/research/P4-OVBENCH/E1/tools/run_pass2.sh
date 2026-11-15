#!/bin/bash
# E1 P4-OVBENCH pass 2: anomaly graphs with native batch 2. Waits for pass 1 (ALL DONE), then runs one measured process at a time.
R=/home/intel/intel-junction-2026
E=$R/work/P4-OVBENCH/E1
T=$E/tools
PY=$R/envs/ovrun-env/bin/python
export OMP_NUM_THREADS=4
until grep -q "ALL DONE" $E/logs/ovbench_run.log; do sleep 15; done
load1() { cut -d' ' -f1 /proc/loadavg; }
gate() {
  for i in $(seq 1 45); do
    if awk -v l="$(load1)" 'BEGIN{exit !(l < 14)}'; then return 0; fi
    sleep 20
  done
  echo "gate: load still $(load1) after 15 min; running anyway"
}
run2() { # tag kind xml device [gpu_hint]
  gate
  echo "== RUN2 $1 $4 hint=${5:-none} $(date -u +%FT%TZ) load_before=$(cat /proc/loadavg)"
  timeout 1500 $PY -I -B $T/bench_ov_v2.py "$1" "$2" "$3" "$4" "$E/raw" 30 10 400 "$5"
  echo "== exit=$? $1 $4 $(date -u +%FT%TZ) load_after=$(cat /proc/loadavg)"
}
echo "== pass2 start $(date -u +%FT%TZ) load=$(cat /proc/loadavg)"
for d in CPU GPU NPU; do run2 anom_pre_close anomaly $E/models_fp16/anom_pre_close.xml $d; done
run2 anom_pre_close_f32hint anomaly $E/models_fp16/anom_pre_close.xml GPU f32
for d in CPU GPU NPU; do run2 anom_patchcore anomaly $E/models_fp16/anom_patchcore.xml $d; done
nice -n 19 $PY -I -B $T/aggregate_v2.py "$E/raw" "$E" 2>&1 | tail -n 60
echo "== PASS2 ALL DONE $(date -u +%FT%TZ)"
