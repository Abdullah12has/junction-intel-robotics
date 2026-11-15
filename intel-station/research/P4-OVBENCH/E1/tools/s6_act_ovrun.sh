#!/bin/bash
# S6 (E1, P4-OVBENCH): the production ACT config. The runner loads the FP32 IR in P2-B-OVRUN; measure that IR on GPU with the
# default precision hint (no hint), compare with the FP32 CPU reference, and re-run the aggregator so bench.tsv and metrics.json include it.
# Waits for S5 to finish so measured runs never overlap.
R=/home/intel/intel-junction-2026
E=$R/work/P4-OVBENCH/E1
PYO=$R/envs/ovrun-env/bin/python
OVRUN=$R/work/P2-B-OVRUN/E1/models/389da0fd/openvino/act.xml
FP32=$R/work/P2-B-OPT/E1/models/389da0fd/act.xml
OUT=$E/logs/s6_result.txt
gate() {
  for i in $(seq 1 45); do
    GL=$(cut -d' ' -f1 /proc/loadavg)
    if awk -v l="$GL" 'BEGIN{exit !(l+0 < 14)}'; then return 0; fi
    sleep 20
  done
  return 1
}
runc() {
  local LBL=$1 LOG=$2
  shift 2
  if gate; then
    echo "$LBL: start $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)"
    "$@" > "$LOG" 2>&1
    echo "$LBL: exit=$? at $(date -u +%FT%TZ)"
  else
    echo "$LBL: SKIPPED (1-min load stayed at or above 14 for 15 min)"
  fi
}
{
echo "== S6 start $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)"
for i in $(seq 1 240); do grep -q "== S5 DONE" $E/logs/s5_result.txt 2>/dev/null && break; sleep 5; done
echo "S5 finished; S6 begins $(date -u +%FT%TZ)"
echo "== runner config lines (read only)"
grep -rn "PRECISION\|INFERENCE_\|hint\|compile_model" $R/app/ovrun/ 2>/dev/null | cut -c1-200 | head -n 12
echo "== IR identity (sha256)"
sha256sum "$OVRUN" "$FP32" "$(dirname "$OVRUN")/act.bin" "$(dirname "$FP32")/act.bin"
runc "act (P2-B-OVRUN IR), GPU, default hint" $E/logs/act_ovrun_GPU.log $PYO $E/tools/bench_ov_v2.py act_ovrun act $OVRUN GPU $E/raw 20 5 300
echo "== parity: FP32 CPU reference vs runner IR on GPU"
$PYO -I $E/tools/parity_pair.py act_fp32 CPU act_ovrun GPU 2>&1
nice -n 19 $PYO -I -B $E/tools/aggregate_v2.py $E/raw $E > $E/logs/aggregate_v2_s6.log 2>&1
echo "aggregate_v2 exit=$? at $(date -u +%FT%TZ)"
tail -n 2 $E/logs/aggregate_v2_s6.log | cut -c1-200
echo "== S6 DONE $(date -u +%FT%TZ)"
} > $OUT 2>&1
