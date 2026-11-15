#!/bin/bash
# S7 (E1, P4-OVBENCH): (a) GPU runs with the f32 precision hint for the PatchCore IR (native batch 2) and the PaDiM IR (batch 1), parity only,
# to test whether the FP16 GPU output saturates; (b) parity matrix v2 over all anomaly rows (reference = CPU of the same IR); (c) a read-only
# search of P4-FUSION for the PaDiM and PatchCore IR paths it used. Waits for S6 so measured runs never overlap.
R=/home/intel/intel-junction-2026
E=$R/work/P4-OVBENCH/E1
PYO=$R/envs/ovrun-env/bin/python
OUT=$E/logs/s7_result.txt
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
echo "== S7 start $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)"
for i in $(seq 1 240); do grep -q "== S6 DONE" $E/logs/s6_result.txt 2>/dev/null && break; sleep 5; done
echo "S6 finished; S7 begins $(date -u +%FT%TZ)"
PC_XML=$(python3 -c "import json; print(json.load(open('$E/raw/anom_patchcore__CPU.json'))['xml'])")
PC_KIND=$(python3 -c "import json; print(json.load(open('$E/raw/anom_patchcore__CPU.json'))['kind'])")
PD_XML=$(python3 -c "import json; print(json.load(open('$E/raw/anom_padim__CPU.json'))['xml'])")
PD_KIND=$(python3 -c "import json; print(json.load(open('$E/raw/anom_padim__CPU.json'))['kind'])")
echo "patchcore: xml=$PC_XML kind=$PC_KIND"
echo "padim: xml=$PD_XML kind=$PD_KIND"
runc "patchcore GPU f32 hint" $E/logs/anom_patchcore_f32hint_GPU.log $PYO $E/tools/bench_ov_v2.py anom_patchcore_f32hint $PC_KIND "$PC_XML" GPU $E/raw 3 2 40 f32
runc "padim GPU f32 hint" $E/logs/anom_padim_f32hint_GPU.log $PYO $E/tools/bench_ov_v2.py anom_padim_f32hint $PD_KIND "$PD_XML" GPU $E/raw 3 2 40 f32
echo "== parity matrix v2"
$PYO -I $E/tools/parity_matrix2.py 2>&1
nice -n 19 $PYO -I -B $E/tools/aggregate_v2.py $E/raw $E > $E/logs/aggregate_v2_s7.log 2>&1
echo "aggregate_v2 exit=$? at $(date -u +%FT%TZ)"
echo "== P4-FUSION IR references (read only, timeout 60 s)"
timeout 60 grep -rn "anomaly/padim\|padim/weights\|patchcore/weights\|padim.*\.xml\|patchcore.*\.xml" $R/work/P4-FUSION/E1 --include=*.py --include=*.sh --include=*.yaml --include=*.yml --include=*.md --include=*.json --include=*.txt 2>/dev/null | cut -c1-220 | head -n 12
echo "== S7 DONE $(date -u +%FT%TZ)"
} > $OUT 2>&1
