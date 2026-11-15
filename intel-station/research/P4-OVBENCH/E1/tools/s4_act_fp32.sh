#!/bin/bash
# S4: ACT FP32 reference on CPU, FP32 on GPU (precision hint f32); one measured run at a time, load-gated; then parity focus.
R=/home/intel/intel-junction-2026
E=$R/work/P4-OVBENCH/E1
PYO=$R/envs/ovrun-env/bin/python
XML=$R/work/P2-B-OPT/E1/models/389da0fd/act.xml
OUT=$E/logs/s4_result.txt
gate() {
  for i in $(seq 1 45); do
    L=$(cut -d' ' -f1 /proc/loadavg)
    if awk -v l="$L" 'BEGIN{exit !(l+0 < 14)}'; then return 0; fi
    sleep 20
  done
  return 1
}
{
echo "== S4 start $(date -u +%FT%TZ) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
ls -la $XML 2>&1
gate; echo "gate before CPU run: load $(cut -d' ' -f1-3 /proc/loadavg)"
$PYO $E/tools/bench_ov_v2.py act_fp32 act $XML CPU $E/raw 30 10 400 > $E/logs/act_fp32_CPU.log 2>&1
echo "act_fp32 CPU exit=$? at $(date -u +%FT%TZ)"
gate; echo "gate before GPU run: load $(cut -d' ' -f1-3 /proc/loadavg)"
$PYO $E/tools/bench_ov_v2.py act_fp32 act $XML GPU $E/raw 30 10 400 f32 > $E/logs/act_fp32_GPU.log 2>&1
echo "act_fp32 GPU exit=$? at $(date -u +%FT%TZ)"
echo "-- bench log tails"
tail -n 3 $E/logs/act_fp32_CPU.log | cut -c1-220
tail -n 3 $E/logs/act_fp32_GPU.log | cut -c1-220
echo "-- act_fp32 raw files"
ls $E/raw | grep act_fp32
nice -n 19 $PYO $E/tools/aggregate_v2.py > $E/logs/aggregate_v2_s4.log 2>&1
echo "aggregate_v2 exit=$? at $(date -u +%FT%TZ)"
tail -n 2 $E/logs/aggregate_v2_s4.log | cut -c1-220
echo "== parity focus (CPU is the reference)"
$PYO -I $E/tools/parity_focus.py 2>&1
echo "== S4 DONE $(date -u +%FT%TZ)"
} > $OUT 2>&1
