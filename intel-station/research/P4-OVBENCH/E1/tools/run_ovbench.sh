#!/bin/bash
# E1 P4-OVBENCH driver. Measured runs go one at a time (one process, OMP_NUM_THREADS=4, OpenVINO CPU threads 4).
# Prep and aggregation run under nice -n 19. A measured run starts only when the 1-min load is below 14 (16 cores).
R=/home/intel/intel-junction-2026
E=$R/work/P4-OVBENCH/E1
T=$E/tools
PY=$R/envs/ovrun-env/bin/python
DATA=$R/envs/data-env/bin/python
export OMP_NUM_THREADS=4
load1() { cut -d' ' -f1 /proc/loadavg; }
gate() {
  for i in $(seq 1 45); do
    if awk -v l="$(load1)" 'BEGIN{exit !(l < 14)}'; then return 0; fi
    sleep 20
  done
  echo "gate: load still $(load1) after 15 min; running anyway"
}
prep() { nice -n 19 $PY -I -B $T/prep_fp16.py "$1" "$2" 2>&1 | tail -n 2; }
run() { # tag kind xml device [gpu_hint]
  gate
  echo "== RUN $1 $4 hint=${5:-none} $(date -u +%FT%TZ) load_before=$(cat /proc/loadavg)"
  timeout 1500 $PY -I -B $T/bench_ov.py "$1" "$2" "$3" "$4" "$E/raw" 30 10 400 "$5"
  echo "== exit=$? $1 $4 $(date -u +%FT%TZ) load_after=$(cat /proc/loadavg)"
}
echo "== start $(date -u +%FT%TZ) load=$(cat /proc/loadavg)"
prep $R/app/anomaly/anom3/models/v000/pre_close/model.xml $E/models_fp16/anom_pre_close.xml
prep $R/work/P2-B-OPT/E1/models/389da0fd_fp16/act.xml $E/models_fp16/act_fp16.xml
prep $R/models/anomaly/patchcore/weights/openvino/model.xml $E/models_fp16/anom_patchcore.xml
prep $R/models/anomaly/padim/weights/openvino/model.xml $E/models_fp16/anom_padim.xml
prep $R/models/anomaly/padim/weights/openvino_b1/model.xml $E/models_fp16/anom_padim_b1.xml
echo "== prep done $(date -u +%FT%TZ)"
for d in CPU GPU NPU; do run anom_pre_close anomaly $E/models_fp16/anom_pre_close.xml $d; done
for d in CPU GPU NPU; do run act_fp16 act $E/models_fp16/act_fp16.xml $d; done
run anom_pre_close_f32hint anomaly $E/models_fp16/anom_pre_close.xml GPU f32
for d in CPU GPU NPU; do run anom_patchcore anomaly $E/models_fp16/anom_patchcore.xml $d; done
for d in CPU GPU NPU; do run anom_padim anomaly $E/models_fp16/anom_padim.xml $d; done
for d in CPU GPU NPU; do run anom_padim_b1 anomaly $E/models_fp16/anom_padim_b1.xml $d; done
echo "== bench done $(date -u +%FT%TZ)"
nice -n 19 $PY -I -B $T/aggregate.py "$E/raw" "$E" 2>&1 | tail -n 60
echo "== perception start $(date -u +%FT%TZ) load=$(cat /proc/loadavg)"
gate
$DATA -I -B $T/perception_timing.py "$E/perception"
echo "== perception exit=$? $(date -u +%FT%TZ)"
echo "== ALL DONE $(date -u +%FT%TZ)"
