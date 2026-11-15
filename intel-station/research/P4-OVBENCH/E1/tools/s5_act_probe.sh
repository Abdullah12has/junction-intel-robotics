#!/bin/bash
# S5 (E1, P4-OVBENCH): (a) ACT FP32 on CPU rerun with OMP_NUM_THREADS=4 (the S4 CPU run was not thread-limited);
# (b) probe: ACT FP16 IR on CPU with INFERENCE_PRECISION_HINT f32 (bench_ov_v2 applies the hint on GPU only, so the probe copy applies it on any device);
# (c) aggregate_v2 with the arguments run_pass2.sh uses (S4 failed with no arguments); (d) read-only parity of the ACT outputs.
R=/home/intel/intel-junction-2026
E=$R/work/P4-OVBENCH/E1
PYO=$R/envs/ovrun-env/bin/python
FP32=$R/work/P2-B-OPT/E1/models/389da0fd/act.xml
FP16=$E/models_fp16/act_fp16.xml
OUT=$E/logs/s5_result.txt
gate() {
  for i in $(seq 1 45); do
    L=$(cut -d' ' -f1 /proc/loadavg)
    if awk -v l="$L" 'BEGIN{exit !(l+0 < 14)}'; then return 0; fi
    sleep 20
  done
  return 1
}
runc() {
  L=$1; LOG=$2; shift 2
  if gate; then
    echo "$L: start $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)"
    "$@" > "$LOG" 2>&1
    echo "$L: exit=$? at $(date -u +%FT%TZ)"
  else
    echo "$L: SKIPPED (1-min load stayed at or above 14 for 15 min)"
  fi
}
{
echo "== S5 start $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)"
runc "act_fp32 CPU, 4 threads" $E/logs/act_fp32_CPU_t4.log env OMP_NUM_THREADS=4 $PYO $E/tools/bench_ov_v2.py act_fp32 act $FP32 CPU $E/raw 30 10 400
sed 's/if DEVICE == "GPU" and GPU_HINT:/if GPU_HINT:/' $E/tools/bench_ov_v2.py > $E/tools/probe_cpu_hint.py
echo "probe hint lines: $(grep -c 'if GPU_HINT:' $E/tools/probe_cpu_hint.py)"
runc "act_fp16 CPU, f32 hint (probe)" $E/logs/act_fp16_cpu_f32_probe.log env OMP_NUM_THREADS=4 $PYO $E/tools/probe_cpu_hint.py act_fp16_cpu_f32 act $FP16 CPU $E/raw 10 5 100 f32
nice -n 19 $PYO -I -B $E/tools/aggregate_v2.py $E/raw $E > $E/logs/aggregate_v2_s5.log 2>&1
echo "aggregate_v2 exit=$? at $(date -u +%FT%TZ)"
tail -n 2 $E/logs/aggregate_v2_s5.log | cut -c1-200
echo "== parity (first member of each pair is the reference)"
$PYO -I $E/tools/parity_s5.py 2>&1
echo "== S5 DONE $(date -u +%FT%TZ)"
} > $OUT 2>&1
